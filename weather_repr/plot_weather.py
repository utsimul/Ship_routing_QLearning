import matplotlib.pyplot as plt

import geopandas as gpd
import numpy as np


import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np


def plot_weather(world):

    land = gpd.read_file("ne_50m_land/ne_50m_land.shp")

    lon_grid, lat_grid = np.meshgrid(world.lons, world.lats)

    u = world.wind_u
    v = world.wind_v
    speed = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Land
    land.plot(ax=ax, color="lightgray", edgecolor="black")

    # Wind heatmap
    im = ax.pcolormesh(lon_grid, lat_grid, speed,
                       shading='auto', cmap='viridis', alpha=0.6)

    plt.colorbar(im, ax=ax, label="Wind Speed")

    # Wind vectors
    step = max(1, world.height // 30)

    ax.quiver(
        lon_grid[::step, ::step],
        lat_grid[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        scale=50
    )

    # Storms
    for storm in world.storm_data:
        ax.scatter(storm['lon'], storm['lat'], color='red', s=80)

    ax.set_xlim(world.lon_min, world.lon_max)
    ax.set_ylim(world.lat_min, world.lat_max)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Weather Simulation")

    plt.show()