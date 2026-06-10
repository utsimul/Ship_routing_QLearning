# feed data to agent and obtain its outcome

import numpy as np

from environment.world_generator import WorldGrid
from environment.land_mask import LandMask
from environment.ocean_env import OceanEnvironment
from weather.weather_simulator import WeatherSimulator

from weather_repr.radial_snapshot import get_radial_weather

from agent.random_agent import RandomAgent 

import matplotlib.pyplot as plt


EARTH_RADIUS_KM = 6371.0


def geodesic_distance(lat1, lon1, lat2, lon2):
    """
    Haversine distance in km.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return EARTH_RADIUS_KM * c


def bearing_to_goal(lat1, lon1, lat2, lon2):
    """
    Bearing from current position to goal.

    Returns angle in radians.
    """

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlon = np.radians(lon2 - lon1)

    y = np.sin(dlon) * np.cos(lat2)

    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    return np.arctan2(y, x)


def plot_episode(world, env, trajectory):

    plt.figure(figsize=(14, 7))

    plt.imshow(
        world.land_mask,
        origin="lower",
        extent=[
            world.lon_min,
            world.lon_max,
            world.lat_min,
            world.lat_max
        ],
        cmap="Greys",
        alpha=0.8
    )

    lats = [p[0] for p in trajectory]
    lons = [p[1] for p in trajectory]

    plt.plot(
        lons,
        lats,
        linewidth=2,
        label="Ship Path"
    )

    plt.scatter(
        lons[0],
        lats[0],
        s=150,
        marker="o",
        label="Start"
    )

    plt.scatter(
        lons[-1],
        lats[-1],
        s=150,
        marker="x",
        label="End"
    )

    goal_lat, goal_lon = env.goal_position

    plt.scatter(
        goal_lon,
        goal_lat,
        s=200,
        marker="*",
        label="Goal"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Agent Trajectory")
    plt.legend()
    plt.grid(True)

    plt.show()

def main():


    world = WorldGrid(
        lat_min=-90,
        lat_max=90,
        lon_min=-180,
        lon_max=180,
        resolution=1.0
    )

    print("World shape:", world.shape())

    agent = RandomAgent()


    print("Generating land mask...")

    mask_generator = LandMask()
    mask_generator.generate_mask(world)

    print("Land mask generated")

    weather = WeatherSimulator(world)

    # initialize first weather frame
    weather.update()

    print("Weather initialized")


    env = OceanEnvironment(world)

    env.reset()

    trajectory = [env.ship_position]

    print("Ship:", env.ship_position)
    print("Goal:", env.goal_position)


    NUM_STEPS = 1000
    NUM_EPISODES = 1
    


    for episode in range(NUM_EPISODES):
        step_counter = 0
        trajectory = [env.ship_position]
        actions = []
        rewards = []

        state = env.reset()
        start_position = env.ship_position

        done = False

        while not done:
            
            ship_lat, ship_lon = env.ship_position

            radial_weather = get_radial_weather(
                world,
                ship_lat,
                ship_lon
            )

            #CONSTRUCT STATE 
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

            agent_state = np.concatenate([
                radial_weather.flatten(),
                np.array([
                    dist_to_goal,
                    dist_from_start,
                    np.sin(goal_direction), #encoded as sin to avoid wrapping issues
                    np.cos(goal_direction)
                ])
            ])

            theta = agent.act(agent_state)

            next_state, reward, done = env.step(theta)
            trajectory.append(env.ship_position)

            print(
                f"Reward {reward:.2f} | "
                f"Position {next_state}"
            )

            state = next_state
            step_counter += 1

            if step_counter >= NUM_STEPS:
                print("Max steps reached, ending episode.")
                break

            # if np.random.rand() < 0.02:
            #     weather._spawn_storm()

            # weather.update()
        
        print("Episode finished")

        plot_episode(
            world,
            env,
            trajectory
        )

        env.reset()

        trajectory = [env.ship_position]

        print("New ship:", env.ship_position)
        print("New goal:", env.goal_position)

        print("Updating agent...")

        #agent.update(actions, rewards)

        print("Agent updated")

        




if __name__ == "__main__":
    main()