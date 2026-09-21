import os
import json
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from environment.world_generator import WorldGrid
from environment.land_mask import LandMask
from environment.ocean_env import OceanEnvironment
from weather.weather_simulator import WeatherSimulator

from weather_repr.plot_weather import *
from weather_repr.radial_snapshot import get_radial_weather, get_radial_land

from agent.policy_agent_ll import PolicyAgent

from Helpers import *

SEED = 0

NUM_UPDATES = 300            
ROLLOUT_STEPS = 128          
GAMMA = 0.99
GAE_LAMBDA = 0.95

DEFAULT_EPISODE_LIMIT = 600  # used if the baseline can't reach a route
EPISODE_LIMIT_FACTOR = 3.0   # limit = factor * (steps straight-line needs)

GOAL_RADIUS_KM = 100.0       # actual radius = max(this, one ship step)

PROGRESS_SCALE = 10.0        # whole journey worth ~10 reward if flown straight
GOAL_BONUS = 5.0
STEP_PENALTY = 0.002

ENV_REWARD_WEIGHT = 0.0

NORMALIZE_OBS = True
RUN_BASELINE_CHECK = True
BASELINE_MAX_STEPS = 1000
PLOT_EVERY = 10
PLOT_DPI = 150
EARTH_RADIUS_KM = 6371.0

def load_training_routes(filename="training_routes.json"):

    with open(filename, "r") as f:
        routes = json.load(f)

    for route in routes:
        route["start"] = tuple(route["start"])
        route["goal"] = tuple(route["goal"])

    return routes


def to_float(x):
    """Scalar float from python float / numpy / torch tensor."""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0])
    return float(np.asarray(x).reshape(-1)[0])


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


class RunningNorm:
    """Running mean/std observation normaliser (Welford / parallel update)."""

    def __init__(self, dim, clip=10.0):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        delta = x - self.mean
        total = self.count + 1.0
        self.mean = self.mean + delta / total
        self.var = (self.var * self.count
                    + delta ** 2 * self.count / total) / total
        self.count = total

    def normalize(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-6)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


def calculate_gae(
    rewards,
    values,
    dones,
    next_value,
    gamma=0.99,
    lam=0.95
):

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    T = len(rewards)

    advantages = np.zeros(T, dtype=np.float32)

    gae = 0.0

    for t in reversed(range(T)):

        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        non_terminal = 1.0 - float(dones[t])

        delta = (
            rewards[t]
            + gamma * next_val * non_terminal
            - values[t]
        )

        gae = (
            delta
            + gamma * lam * non_terminal * gae
        )

        advantages[t] = gae

    returns = advantages + values

    return advantages, returns


_bearing_warned = False


def construct_state(
    world,
    env,
    start_position
):
    """
    Returns (agent_state, radial_weather, radial_land, goal_direction,
             dist_to_goal_km)
    NOTE: the last item is in KM (not normalised); normalise at the call site.
    """
    global _bearing_warned

    ship_lat, ship_lon = env.ship_position
    goal_lat, goal_lon = env.goal_position
    start_lat, start_lon = start_position

    radial_weather = get_radial_weather(world, ship_lat, ship_lon)

    radial_land = get_radial_land(
        world, ship_lat, ship_lon, goal_lat, goal_lon
    )

    dist_to_goal_km = geodesic_distance(
        ship_lat, ship_lon, goal_lat, goal_lon
    )

    dist_from_start_km = geodesic_distance(
        start_lat, start_lon, ship_lat, ship_lon
    )

    goal_direction = bearing_to_goal(
        ship_lat, ship_lon, goal_lat, goal_lon
    )

    if not _bearing_warned and abs(goal_direction) > 2 * math.pi + 1e-3:
        print(
            "\n[WARNING] bearing_to_goal() returned "
            f"{goal_direction:.2f} -> looks like DEGREES. sin/cos below and "
            "the policy assume RADIANS. Convert it in Helpers.\n"
        )
        _bearing_warned = True

    agent_state = np.concatenate([
        np.asarray(radial_weather).flatten(),
        np.asarray(radial_land).flatten(),
        np.array([
            dist_to_goal_km / 20000.0,
            dist_from_start_km / 20000.0,
            np.sin(goal_direction),
            np.cos(goal_direction)
        ])
    ]).astype(np.float32)

    agent_state = np.nan_to_num(
        agent_state, nan=0.0, posinf=0.0, neginf=0.0
    )

    return (
        agent_state,
        radial_weather,
        radial_land,
        goal_direction,
        dist_to_goal_km
    )


def build_observation(world, env, start_position, obs_norm=None,
                      update_norm=False):
    """construct_state +  running normalisation of the state."""
    (
        state, radial_weather, radial_land, goal_direction, dist_km
    ) = construct_state(world, env, start_position)

    if obs_norm is not None:
        if update_norm:
            obs_norm.update(state)
        state = obs_norm.normalize(state)

    return state, radial_weather, radial_land, goal_direction, dist_km


def reset_journey(env, route):
    env.reset(
        route["start"][0],
        route["start"][1],
        route["goal"][0],
        route["goal"][1]
    )
    env.ship_position = route["start"]
    env.goal_position = route["goal"]



def run_baseline_check(world, weather, routes, obs_norm):

    n = len(routes)

    envs = []
    for route in routes:
        env = OceanEnvironment(world)
        reset_journey(env, route)
        envs.append(env)

    reached_at = [None] * n
    min_km = [float("inf")] * n
    first_step_lengths = []
    radius = GOAL_RADIUS_KM

    for t in range(BASELINE_MAX_STEPS):

        active = 0

        for j, (env, route) in enumerate(zip(envs, routes)):

            if reached_at[j] is not None:
                continue

            active += 1

            (
                state, rw, rl, gdir, dkm
            ) = build_observation(
                world, env, route["start"], obs_norm, update_norm=True
            )

            min_km[j] = min(min_km[j], dkm)

            if dkm < radius:
                reached_at[j] = t
                continue

            prev = tuple(env.ship_position)

            theta = gdir
            _, _, env_done = env.step(theta, rw, dkm / 20000.0, weather)

            if t == 0:
                new = env.ship_position
                first_step_lengths.append(
                    geodesic_distance(prev[0], prev[1], new[0], new[1])
                )

            if env_done:
                reached_at[j] = t + 1

        if t == 0:
            moved = [s for s in first_step_lengths if s > 1e-3]
            if moved:
                radius = max(GOAL_RADIUS_KM, float(np.median(moved)))

        weather.update()

        if active == 0:
            break

    step_km = None
    moved = [s for s in first_step_lengths if s > 1e-3]
    if moved:
        step_km = float(np.median(moved))

    print("\nBaseline check (steer straight at goal every step):")
    print(f"  measured ship step length : "
          f"{'unknown (ship did not move!)' if step_km is None else f'{step_km:.1f} km'}")
    print(f"  goal radius used          : {radius:.1f} km")

    for j, route in enumerate(routes):
        length = geodesic_distance(
            route["start"][0], route["start"][1],
            route["goal"][0], route["goal"][1]
        )
        if reached_at[j] is not None:
            print(f"  {route['name']:<28s} reached in {reached_at[j]} steps "
                  f"(route {length:.0f} km)")
        else:
            print(f"  {route['name']:<28s} NOT reached "
                  f"(closest {min_km[j]:.0f} km, route {length:.0f} km)")

    n_ok = sum(r is not None for r in reached_at)
    if n_ok == 0:
        print(
            "\n  [!] The baseline reached NO goals. Before blaming PPO check:\n"
            "      - does env.step() move the ship at all?\n"
            "      - is theta an absolute heading (radians) or an offset?\n"
            "      - is bearing_to_goal in radians?\n"
            "      - is the goal radius / done rule in the env sane?"
        )

    return {
        "reached_at": reached_at,
        "step_km": step_km,
        "radius": radius,
    }


def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs("plots", exist_ok=True)

    TRAIN_ROUTES = load_training_routes("train_routes_selected.json")
    NUM_JOURNEYS = len(TRAIN_ROUTES)

    world = WorldGrid(
        lat_min=-90,
        lat_max=90,
        lon_min=-180,
        lon_max=180,
        resolution=1.0
    )

    print("World shape:", world.shape())

    print("Generating land mask...")
    mask_generator = LandMask()
    mask_generator.generate_mask(world)
    print("Land mask generated.")

    weather = WeatherSimulator(world)
    weather.update()
    print("Weather initialized.")


    environments = []
    for route in TRAIN_ROUTES:
        env = OceanEnvironment(world)
        reset_journey(env, route)
        environments.append(env)

    print(f"Created {len(environments)} independent journey environments.")

    route_len_km = [
        geodesic_distance(
            r["start"][0], r["start"][1], r["goal"][0], r["goal"][1]
        )
        for r in TRAIN_ROUTES
    ]

    sample_state = construct_state(
        world, environments[0], TRAIN_ROUTES[0]["start"]
    )[0]
    input_dim = len(sample_state)
    print("Policy input dimension:", input_dim)

    obs_norm = RunningNorm(input_dim) if NORMALIZE_OBS else None


    goal_radius = GOAL_RADIUS_KM
    episode_limit = [DEFAULT_EPISODE_LIMIT] * NUM_JOURNEYS

    if RUN_BASELINE_CHECK:
        base = run_baseline_check(world, weather, TRAIN_ROUTES, obs_norm)
        goal_radius = base["radius"]
        for j, steps in enumerate(base["reached_at"]):
            if steps is not None:
                episode_limit[j] = max(
                    int(EPISODE_LIMIT_FACTOR * steps), 50
                )

    print("\nEpisode limits (steps):",
          {r["name"]: episode_limit[j] for j, r in enumerate(TRAIN_ROUTES)})

    for env, route in zip(environments, TRAIN_ROUTES):
        reset_journey(env, route)

    PAgent = PolicyAgent(input_dim)
    print("Policy agent initialized.")


    episode_steps = [0] * NUM_JOURNEYS
    trajectories = [[tuple(r["start"])] for r in TRAIN_ROUTES]
    last_finished_traj = [None] * NUM_JOURNEYS
    last_finished_outcome = [None] * NUM_JOURNEYS

    total_goals = 0
    total_timeouts = 0
    warned_env_done_far = False

    for update in range(NUM_UPDATES):

        print("\n" + "=" * 70)
        print(f"TRAINING UPDATE {update + 1}/{NUM_UPDATES}")
        print("=" * 70)

        journey_buffers = [
            {
                "states": [],
                "actions": [],
                "rewards": [],
                "values": [],
                "log_probs": [],
                "dones": []
            }
            for _ in range(NUM_JOURNEYS)
        ]

        goals_this_update = 0
        timeouts_this_update = 0
        env_done_far_this_update = 0

        
        for t in range(ROLLOUT_STEPS):

            print(f"\rCollecting timestep {t + 1}/{ROLLOUT_STEPS}", end="")

            for j, route in enumerate(TRAIN_ROUTES):

                env = environments[j]
                start_position = route["start"]

                (
                    agent_state,
                    radial_weather,
                    radial_land,
                    goal_direction,
                    dist_km
                ) = build_observation(
                    world, env, start_position, obs_norm, update_norm=True
                )

                dist_norm = dist_km / 20000.0

                (
                    theta,
                    log_prob,
                    value,
                    raw_action
                ) = PAgent.act(
                    agent_state,
                    goal_direction,
                    dist_norm
                )


                _, env_reward, env_done = env.step(
                    theta,
                    radial_weather,
                    dist_norm,
                    weather
                )

                env_done = bool(env_done)

                ship_lat, ship_lon = env.ship_position
                goal_lat, goal_lon = env.goal_position
                new_km = geodesic_distance(
                    ship_lat, ship_lon, goal_lat, goal_lon
                )

                episode_steps[j] += 1

                reached = new_km < goal_radius
                terminal = reached or env_done
                timed_out = (not terminal) and (
                    episode_steps[j] >= episode_limit[j]
                )

                if env_done and not reached:
                    env_done_far_this_update += 1
                    if not warned_env_done_far:
                        warned_env_done_far = True
                        print(
                            f"\n[NOTE] env.step returned done=True while the "
                            f"ship is still {new_km:.0f} km from the goal "
                            f"(radius {goal_radius:.0f} km). Treating it as a "
                            f"terminal WITHOUT goal bonus. If this is just "
                            f"your env's own goal threshold, raise "
                            f"GOAL_RADIUS_KM."
                        )

                progress = (dist_km - new_km) / max(route_len_km[j], 1.0)
                reward = PROGRESS_SCALE * progress - STEP_PENALTY

                if reached:
                    reward += GOAL_BONUS

                if ENV_REWARD_WEIGHT != 0.0:
                    reward += ENV_REWARD_WEIGHT * float(
                        np.clip(to_float(env_reward), -10.0, 10.0)
                    )

                if timed_out:
                    next_obs = build_observation(
                        world, env, start_position, obs_norm,
                        update_norm=False
                    )[0]
                    reward += GAMMA * to_float(PAgent.get_value(next_obs))

                episode_end = terminal or timed_out

                buffer = journey_buffers[j]
                buffer["states"].append(agent_state)
                buffer["actions"].append(to_numpy(raw_action))
                buffer["rewards"].append(float(reward))
                buffer["values"].append(to_float(value))
                buffer["log_probs"].append(to_float(log_prob))
                buffer["dones"].append(episode_end)

                trajectories[j].append(tuple(env.ship_position))

                if episode_end:

                    if reached:
                        goals_this_update += 1
                        total_goals += 1
                        print(
                            f"\nJourney {j} ({route['name']}) reached goal "
                            f"after {episode_steps[j]} steps "
                            f"(update {update + 1}, t={t + 1})"
                        )
                    elif timed_out:
                        timeouts_this_update += 1
                        total_timeouts += 1

                    last_finished_traj[j] = trajectories[j]
                    last_finished_outcome[j] = (
                        "goal" if reached
                        else "timeout" if timed_out
                        else "envdone"
                    )

                    reset_journey(env, route)
                    episode_steps[j] = 0
                    trajectories[j] = [tuple(route["start"])]

            weather.update()

        print()


        all_states = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_rewards = []

        progress_fracs = []

        for j, route in enumerate(TRAIN_ROUTES):

            buffer = journey_buffers[j]
            env = environments[j]

            next_obs, _, _, _, cur_km = build_observation(
                world, env, route["start"], obs_norm, update_norm=False
            )

            progress_fracs.append(
                1.0 - cur_km / max(route_len_km[j], 1.0)
            )

            if len(buffer["states"]) == 0:
                continue

            if buffer["dones"][-1]:
                next_value = 0.0       
            else:
                next_value = to_float(PAgent.get_value(next_obs))

            advantages, returns = calculate_gae(
                rewards=buffer["rewards"],
                values=buffer["values"],
                dones=buffer["dones"],
                next_value=next_value,
                gamma=GAMMA,
                lam=GAE_LAMBDA
            )

            all_states.extend(buffer["states"])
            all_actions.extend(buffer["actions"])
            all_log_probs.extend(buffer["log_probs"])
            all_rewards.extend(buffer["rewards"])
            all_advantages.extend(advantages)
            all_returns.extend(returns)

        all_states = np.asarray(all_states, dtype=np.float32)
        all_actions = np.asarray(all_actions, dtype=np.float32)
        all_log_probs = np.asarray(all_log_probs, dtype=np.float32)
        all_advantages = np.asarray(all_advantages, dtype=np.float32)
        all_returns = np.asarray(all_returns, dtype=np.float32)

        if len(all_advantages) > 1:
            all_advantages = (
                all_advantages - all_advantages.mean()
            ) / (all_advantages.std() + 1e-8)

        num_transitions = len(all_states)

        print(
            f"Collected {num_transitions} transitions for PPO update "
            f"(max possible {NUM_JOURNEYS * ROLLOUT_STEPS})."
        )


        if num_transitions > 0:

            PAgent.update(
                states=all_states,
                raw_actions=all_actions,
                old_log_probs=all_log_probs,
                returns=all_returns,
                advantages=all_advantages
            )


        if update % PLOT_EVERY == 0 or update == NUM_UPDATES - 1:

            for j, route in enumerate(TRAIN_ROUTES):

                if last_finished_traj[j] is not None:
                    traj = last_finished_traj[j]
                    outcome = last_finished_outcome[j]
                else:
                    traj = trajectories[j]
                    outcome = "inflight"

                try:
                    plot_episode(world, environments[j], traj)

                    plt.savefig(
                        f"plots/update_{update}_journey_{j}_"
                        f"{route['name']}_{outcome}.png",
                        dpi=PLOT_DPI,
                        bbox_inches="tight"
                    )
                except Exception as e:
                    print(f"[plot warning] journey {j}: {e}")
                finally:
                    plt.close("all")


        mean_r = float(np.mean(all_rewards)) if all_rewards else 0.0

        print(f"Completed PPO update {update + 1}")
        print(f"  goals reached this update : {goals_this_update}")
        print(f"  timeouts this update      : {timeouts_this_update}")
        if env_done_far_this_update:
            print(f"  env done far from goal    : {env_done_far_this_update}")
        print(f"  mean step reward          : {mean_r:+.4f}")
        print(f"  mean progress (in-flight) : "
              f"{100.0 * float(np.mean(progress_fracs)):.1f}% of route")
        print(f"  total goals / timeouts    : {total_goals} / {total_timeouts}")
        finished = goals_this_update + timeouts_this_update
        if finished:
            print(f"  success rate this update  : "
                  f"{100.0 * goals_this_update / finished:.0f}% "
                  f"({goals_this_update}/{finished} finished episodes)")
        print("-" * 70)


if __name__ == "__main__":
    main()