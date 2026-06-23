import matplotlib.pyplot as plt

import geopandas as gpd
import numpy as np


import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from weather_repr.radial_snapshot import get_radial_weather


def Plot_weather(world):

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

    print("Weather plot generated")

    return fig

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

def visualize_single_level(level, scale):
    """
    level: (C, H, W)
    """
    if level.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got shape {level.shape}")

    mean_u = level[0]
    mean_v = level[1]
    max_speed = level[2]

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Scale: {scale} x {scale}")

    im0 = axes[0].imshow(mean_u)
    axes[0].set_title("Mean U")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mean_v)
    axes[1].set_title("Mean V")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(max_speed)
    axes[2].set_title("Max Speed")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

    # Vector field
    H, W = mean_u.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    plt.figure(figsize=(6, 6))
    plt.imshow(max_speed)
    plt.quiver(X, Y, mean_u, mean_v, color='white')
    plt.title(f"Combined (Scale {scale})")
    plt.gca().invert_yaxis()
    plt.show()

def visualize_full_pyramid(pyramid, scales=[20, 10, 5]):
    for level, scale in zip(pyramid, scales):
        visualize_single_level(level, scale)


def plot_agent_view(world, agent_lat, agent_lon):

    land = gpd.read_file("ne_50m_land/ne_50m_land.shp")

    lon_grid, lat_grid = np.meshgrid(world.lons, world.lats)

    u = world.wind_u
    v = world.wind_v

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Background ---
    land.plot(ax=ax, color="lightgray", edgecolor="black")
    ax.set_facecolor("lightblue")

    step = max(1, world.height // 25)

    ax.quiver(
        lon_grid[::step, ::step],
        lat_grid[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        color="gray",
        alpha=0.4,
        scale=60
    )

    # --- Agent ---
    ax.scatter(agent_lon, agent_lat, color="red", s=120, label="Agent")

    # --- Get encoded data ---
    samples = get_radial_weather(world, agent_lat, agent_lon)

    ai, aj = world.lat_lon_to_grid(agent_lat, agent_lon)

    idx = 0
    radii = [3, 8, 15, 30]
    max_radius = max(radii)

    # --- Reconstruct sampling ---
    for r in radii:
        num_points = max(8, r * 2)

        for k in range(num_points):

            if idx >= len(samples):
                continue

            angle = 2 * np.pi * k / num_points

            i = int(ai + r * np.sin(angle))
            j = int(aj + r * np.cos(angle))

            if 0 <= i < world.height and 0 <= j < world.width:

                lat_pt, lon_pt = world.get_coordinates(i, j)

                # --- extract encoding ---
                speed, sin_w, cos_w, dist, sin_a, cos_a = samples[idx]

                # reconstruct wind vector
                uu = speed * cos_w
                vv = speed * sin_w

                # --- normalized distance ---
                norm_dist = dist / max_radius

                color = plt.cm.plasma(norm_dist)

                # --- point ---
                ax.scatter(lon_pt, lat_pt, color=color, s=30)

                # --- line from agent ---
                ax.plot(
                    [agent_lon, lon_pt],
                    [agent_lat, lat_pt],
                    color=color,
                    alpha=0.3
                )

                # --- relative direction arrow ---
                dlon = lon_pt - agent_lon
                dlat = lat_pt - agent_lat

                ax.quiver(
                    agent_lon,
                    agent_lat,
                    dlon,
                    dlat,
                    angles='xy',
                    scale_units='xy',
                    scale=1,
                    color=color,
                    alpha=0.1
                )

                # --- wind arrow ---
                ax.quiver(
                    lon_pt, lat_pt,
                    uu, vv,
                    color="black",
                    scale=60
                )

                idx += 1

    # --- Rings ---
    for r in radii:
        circle_lats = []
        circle_lons = []

        for angle in np.linspace(0, 2*np.pi, 100):
            lat = agent_lat + r * world.resolution * np.sin(angle)
            lon = agent_lon + r * world.resolution * np.cos(angle)

            circle_lats.append(lat)
            circle_lons.append(lon)

        ax.plot(circle_lons, circle_lats, linestyle="dashed", alpha=0.4)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Normalized Distance")

    ax.set_xlim(world.lon_min, world.lon_max)
    ax.set_ylim(world.lat_min, world.lat_max)

    ax.set_title("Agent Perception (Reconstructed from Encoded State)")
    ax.legend()

    plt.show()