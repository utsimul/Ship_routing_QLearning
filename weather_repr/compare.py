from weather.weather_simulator import WeatherSimulator
from weather_repr.plot_weather import *
import matplotlib.pyplot as plt
from environment.world_generator import WorldGrid
import numpy as np
from weather_repr.pool import *


if __name__ == "__main__":

    print("compare function")
    world = WorldGrid(
        lat_min=-60,
        lat_max=60,
        lon_min=-180,
        lon_max=180,
        resolution=2.0
    )

    world.wind_u = np.zeros((world.height, world.width))
    world.wind_v = np.zeros((world.height, world.width))

    world.storm_data = [
        {"lat": 20, "lon": 70, "intensity": 3, "radius": 5},
        {"lat": -10, "lon": 120, "intensity": 2, "radius": 4}
    ]

    sim = WeatherSimulator(world)

    for _ in range(5):
        sim.update()

    plot_weather(world)

    pyramid = get_global_weather_representation(world)
    visualize_full_pyramid(pyramid, scales=[20, 10, 5])
