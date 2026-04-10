from weather.weather_simulator import WeatherSimulator
from weather_repr.plot_weather import plot_weather
import matplotlib.pyplot as plt
from environment.world_generator import WorldGrid
import numpy as np


if __name__ == "__main__":

    print("compare function")
    # --- CREATE WORLD ---
    world = WorldGrid(
        lat_min=-60,
        lat_max=60,
        lon_min=-180,
        lon_max=180,
        resolution=2.0
    )

    # --- INIT WIND ARRAYS ---
    world.wind_u = np.zeros((world.height, world.width))
    world.wind_v = np.zeros((world.height, world.width))

    # --- ADD STORMS ---
    world.storm_data = [
        {"lat": 20, "lon": 70, "intensity": 3, "radius": 5},
        {"lat": -10, "lon": 120, "intensity": 2, "radius": 4}
    ]

    # --- INIT SIMULATOR ---
    sim = WeatherSimulator(world)

    # --- RUN SIMULATION ---
    for _ in range(5):
        sim.update()

    # --- PLOT ---
    plot_weather(world)