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

def visualize_vector_field(level, scale):
    """
    Quiver plot for wind direction
    """
    mean_u = level[0]
    mean_v = level[1]

    H, W = mean_u.shape

    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, mean_u, mean_v)
    plt.title(f"Wind Field (Scale {scale}x{scale})")
    plt.gca().invert_yaxis()
    plt.show()

def visualize_pyramid(pyramid, scales=[20, 10, 5]):
    """
    Visualize each level of the pyramid.
    Each level: (C, H, W) where C = [mean_u, mean_v, max_speed]
    """

    num_levels = len(pyramid)

    for idx, level in enumerate(pyramid):
        mean_u = level[0]
        mean_v = level[1]
        max_speed = level[2]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Scale: {scales[idx]} x {scales[idx]}")

        im0 = axes[0].imshow(mean_u)
        axes[0].set_title("Mean U (wind x)")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(mean_v)
        axes[1].set_title("Mean V (wind y)")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(max_speed)
        axes[2].set_title("Max Speed")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()

def visualize_combined(level, scale):
    mean_u = level[0]
    mean_v = level[1]
    max_speed = level[2]

    H, W = mean_u.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    plt.figure(figsize=(6, 6))
    
    # Background: speed
    plt.imshow(max_speed)
    plt.colorbar(label="Max Speed")

    # Overlay: direction
    plt.quiver(X, Y, mean_u, mean_v, color='white')

    plt.title(f"Combined Wind + Speed (Scale {scale}x{scale})")
    plt.gca().invert_yaxis()
    plt.show()

def visualize_full_pyramid(pyramid, scales=[20, 10, 5]):
    for level, scale in zip(pyramid, scales):
        visualize_pyramid(pyramid, scales)  # heatmaps
        visualize_vector_field(level, scale)        # direction
        visualize_combined(level, scale)            # combined