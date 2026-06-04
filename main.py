import numpy as np

from environment.world_generator import WorldGrid
from environment.land_mask import LandMask
from environment.ocean_env import OceanEnvironment
from weather.weather_simulator import WeatherSimulator

from weather_repr.radial_snapshot import get_radial_weather

import matplotlib.pyplot as plt

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

    for step in range(NUM_STEPS):


        if np.random.rand() < 0.02:
            weather._spawn_storm()

        weather.update()



        ship_lat, ship_lon = env.ship_position

        radial_weather = get_radial_weather(
            world,
            ship_lat,
            ship_lon
        )



        theta = np.random.uniform(0, 2 * np.pi)

        next_state, reward, done = env.step(theta)
        trajectory.append(env.ship_position)

        print(
            f"Step {step} | "
            f"Reward {reward:.2f} | "
            f"Position {next_state}"
        )

   

        if done:

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




if __name__ == "__main__":
    main()