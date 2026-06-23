import numpy as np
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

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.imshow(
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

    ax.plot(
        lons,
        lats,
        linewidth=2,
        label="Ship Path"
    )

    ax.scatter(
        lons[0],
        lats[0],
        s=150,
        marker="o",
        label="Start"
    )

    ax.scatter(
        lons[-1],
        lats[-1],
        s=150,
        marker="x",
        label="End"
    )

    goal_lat, goal_lon = env.goal_position

    ax.scatter(
        goal_lon,
        goal_lat,
        s=200,
        marker="*",
        label="Goal"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Agent Trajectory")
    ax.legend()
    ax.grid(True)

    return fig