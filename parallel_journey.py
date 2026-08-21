import numpy as np
import torch
import matplotlib.pyplot as plt

from environment.world_generator import WorldGrid
from environment.land_mask import LandMask
from environment.ocean_env import OceanEnvironment
from weather.weather_simulator import WeatherSimulator

from weather_repr.plot_weather import *
from weather_repr.radial_snapshot import get_radial_weather

from agent.policy_agent_ll import PolicyAgent

from Helpers import *


EARTH_RADIUS_KM = 6371.0


TRAIN_ROUTES = [

    {
        "name": "short_1",
        "start": (20.0, -50.0),
        "goal":  (35.0, -20.0)
    },

    {
        "name": "medium_1",
        "start": (10.0, -80.0),
        "goal":  (40.0, -30.0)
    },

    {
        "name": "medium_2",
        "start": (-20.0, 120.0),
        "goal":  (15.0, 160.0)
    },

    {
        "name": "long_1",
        "start": (-40.0, -150.0),
        "goal":  (40.0, -20.0)
    },

    {
        "name": "long_2",
        "start": (50.0, -170.0),
        "goal":  (-30.0, 100.0)
    },

    {
        "name": "equator_crossing",
        "start": (-25.0, -60.0),
        "goal":  (25.0, -20.0)
    },

    {
        "name": "dateline_crossing",
        "start": (10.0, 170.0),
        "goal":  (15.0, -170.0)
    },

    {
        "name": "north_1",
        "start": (45.0, -70.0),
        "goal":  (60.0, 20.0)
    },

    {
        "name": "south_1",
        "start": (-50.0, 30.0),
        "goal":  (-20.0, 120.0)
    }
]


def calculate_gae(
    rewards,
    values,
    dones,
    next_value,
    gamma=0.99,
    lam=0.95
):
    """
    Calculate GAE for ONE journey.

    rewards : [T]
    values  : [T]
    dones   : [T]

    Returns:
        advantages : [T]
        returns    : [T]
    """

    T = len(rewards)

    advantages = np.zeros(T, dtype=np.float32)

    gae = 0.0

    for t in reversed(range(T)):

        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # If episode terminated, don't bootstrap
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

    returns = advantages + np.asarray(values)

    return advantages, returns

def construct_state(
    world,
    env,
    start_position
):

    ship_lat, ship_lon = env.ship_position

    goal_lat, goal_lon = env.goal_position

    start_lat, start_lon = start_position


    radial_weather = get_radial_weather(
        world,
        ship_lat,
        ship_lon
    )


    dist_to_goal = geodesic_distance(
        ship_lat,
        ship_lon,
        goal_lat,
        goal_lon
    )


    dist_from_start = geodesic_distance(
        start_lat,
        start_lon,
        ship_lat,
        ship_lon
    )


    goal_direction = bearing_to_goal(
        ship_lat,
        ship_lon,
        goal_lat,
        goal_lon
    )


    dist_to_goal_normalized = dist_to_goal / 20000.0

    dist_from_start_normalized = dist_from_start / 20000.0

    agent_state = np.concatenate([
        radial_weather.flatten(),

        np.array([
            dist_to_goal_normalized,
            dist_from_start_normalized,

            np.sin(goal_direction),
            np.cos(goal_direction)
        ])
    ])


    return (
        agent_state,
        radial_weather,
        goal_direction,
        dist_to_goal_normalized
    )


def main():

    NUM_JOURNEYS = len(TRAIN_ROUTES)

    ROLLOUT_STEPS = 128

    NUM_UPDATES = 100

    GAMMA = 0.99

    GAE_LAMBDA = 0.95

    world = WorldGrid(
        lat_min=-90,
        lat_max=90,
        lon_min=-180,
        lon_max=180,
        resolution=1.0
    )

    print("World shape:", world.shape())


    PAgent = PolicyAgent(688)

    print("Policy agent initialized.")

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

        env.reset(
            route["start"][0],
            route["start"][1],
            route["goal"][0],
            route["goal"][1]
        )

        env.ship_position = route["start"]
        env.goal_position = route["goal"]

        environments.append(env)


    print(
        f"Created {len(environments)} independent journey environments."
    )


    journey_done = [False] * NUM_JOURNEYS
    
    all_trajectories = [] 

    for journey_id, route in enumerate(TRAIN_ROUTES):

        env = environments[journey_id]

        env.reset(
            route["start"][0],
            route["start"][1],
            route["goal"][0],
            route["goal"][1]
        )

        env.ship_position = route["start"]
        env.goal_position = route["goal"]

        all_trajectories.append(
            [env.ship_position]
        )



    for update in range(NUM_UPDATES):

        print("\n")
        print("=" * 70)
        print(
            f"TRAINING UPDATE {update + 1}/{NUM_UPDATES}"
        )
        print("=" * 70)

        all_states = []
        all_actions = []
        all_rewards = []
        all_values = []
        all_log_probs = []
        all_dones = []

        all_advantages = []
        all_returns = []

        journey_buffers = []

        for journey_id in range(NUM_JOURNEYS):

            journey_buffers.append({
                "states": [],
                "actions": [],
                "rewards": [],
                "values": [],
                "log_probs": [],
                "dones": []
            })



        for t in range(ROLLOUT_STEPS):

            print(
                f"\rCollecting timestep "
                f"{t + 1}/{ROLLOUT_STEPS}",
                end=""
            )


            for journey_id, route in enumerate(TRAIN_ROUTES):

                env = environments[journey_id]

                if journey_done[journey_id]:

                    continue


                start_position = route["start"]


                (
                    agent_state,
                    radial_weather,
                    goal_direction,
                    dist_to_goal
                ) = construct_state(
                    world,
                    env,
                    start_position
                )


                (
                    theta,
                    log_prob,
                    value,
                    raw_action
                ) = PAgent.act(
                    agent_state,
                    goal_direction,
                    dist_to_goal
                )


                next_state, reward, done = env.step(
                    theta,
                    radial_weather,
                    dist_to_goal,
                    weather
                )


                buffer = journey_buffers[journey_id]

                buffer["states"].append(
                    agent_state
                )


                buffer["actions"].append(
                    raw_action
                )

                buffer["rewards"].append(
                    reward
                )

                buffer["values"].append(
                    value
                )

                buffer["log_probs"].append(
                    log_prob
                )

                buffer["dones"].append(
                    done
                )


                all_trajectories[journey_id].append(
                    env.ship_position
                )


                if done:

                    journey_done[journey_id] = True

                    print(
                        f"\nJourney {journey_id} "
                        f"({route['name']}) reached goal "
                        f"at global timestep {t + 1}"
                    )

            weather.update()


        print()


        for journey_id, route in enumerate(TRAIN_ROUTES):

            buffer = journey_buffers[journey_id]


            if len(buffer["states"]) == 0:

                continue


            if buffer["dones"][-1]:


                next_value = 0.0

            else:

                env = environments[journey_id]

                (
                    next_agent_state,
                    _,
                    _,
                    _
                ) = construct_state(
                    world,
                    env,
                    route["start"]
                )

                next_value = PAgent.get_value(
                    next_agent_state
                )


            advantages, returns = calculate_gae(
                rewards=buffer["rewards"],
                values=buffer["values"],
                dones=buffer["dones"],
                next_value=next_value,
                gamma=GAMMA,
                lam=GAE_LAMBDA
            )


            all_states.extend(
                buffer["states"]
            )

            all_actions.extend(
                buffer["actions"]
            )

            all_rewards.extend(
                buffer["rewards"]
            )

            all_values.extend(
                buffer["values"]
            )

            all_log_probs.extend(
                buffer["log_probs"]
            )

            all_dones.extend(
                buffer["dones"]
            )

            all_advantages.extend(
                advantages
            )

            all_returns.extend(
                returns
            )


        all_states = np.asarray(
            all_states,
            dtype=np.float32
        )

        all_actions = np.asarray(
            all_actions,
            dtype=np.float32
        )

        all_log_probs = np.asarray(
            all_log_probs,
            dtype=np.float32
        )

        all_advantages = np.asarray(
            all_advantages,
            dtype=np.float32
        )

        all_returns = np.asarray(
            all_returns,
            dtype=np.float32
        )



        if len(all_advantages) > 1:

            all_advantages = (
                all_advantages
                - all_advantages.mean()
            ) / (
                all_advantages.std() + 1e-8
            )



        num_transitions = len(all_states)

        print(
            f"\nCollected {num_transitions} transitions "
            f"for PPO update."
        )

        print(
            f"Maximum possible: "
            f"{NUM_JOURNEYS * ROLLOUT_STEPS}"
        )


        if num_transitions > 0:

            PAgent.update(
                states=all_states,
                raw_actions=all_actions,
                old_log_probs=all_log_probs,
                returns=all_returns,
                advantages=all_advantages
            )


        if update % 10 == 0:

            for journey_id in range(
                min(3, NUM_JOURNEYS)
            ):

                env = environments[journey_id]

                trajectory = all_trajectories[journey_id]

                fig = plot_episode(
                    world,
                    env,
                    trajectory
                )

                plt.savefig(
                    f"plots/"
                    f"update_{update}_"
                    f"journey_{journey_id}.png",
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()


        completed = sum(journey_done)

        print(
            f"Completed PPO update {update + 1}"
        )

        print(
            f"Journeys completed: "
            f"{completed}/{NUM_JOURNEYS}"
        )

        print(
            "-" * 70
        )


if __name__ == "__main__":
    main()