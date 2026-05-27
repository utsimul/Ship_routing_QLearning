import numpy as np

from environment.world_generator import WorldGrid
from environment.land_mask import LandMask
from environment.ocean_env import OceanEnvironment
from weather.weather_simulator import WeatherSimulator

from weather_repr.radial_snapshot import get_radial_weather


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

        print(
            f"Step {step} | "
            f"Reward {reward:.2f} | "
            f"Position {next_state}"
        )

   

        if done:

            print("Episode finished")

            env.reset()

            print("New ship:", env.ship_position)
            print("New goal:", env.goal_position)


if __name__ == "__main__":
    main()