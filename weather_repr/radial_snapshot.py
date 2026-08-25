import numpy as np

def get_radial_weather(world, lat, lon):

    ai, aj = world.lat_lon_to_grid(lat, lon)

    u = world.wind_u
    v = world.wind_v

    H, W = world.height, world.width

    result = []

    radii = [3, 8, 15, 30]

    for r in radii:

        num_points = max(8, r * 2)

        for k in range(num_points):

            angle = 2 * np.pi * k / num_points

            i = int(round(ai + r * np.sin(angle)))
            j = int(round(aj + r * np.cos(angle)))

            # longitude wraps around globe
            j = j % W

            # latitude does not wrap
            if 0 <= i < H:

                uu = u[i, j]
                vv = v[i, j]

                speed = np.sqrt(uu**2 + vv**2)
                wind_dir = np.arctan2(vv, uu)

            else:
                # outside valid latitude range
                speed = 0.0
                wind_dir = 0.0

            result.append([
                speed,
                np.sin(wind_dir),
                np.cos(wind_dir),
                float(r),
                np.sin(angle),
                np.cos(angle)
            ])

    return np.array(result, dtype=np.float32)

"""
dimensions are guarenteed because it pads out of bounds with zeros.
radii = [3, 8, 15, 30]

points =
8 + 16 + 30 + 60
= 114"""

import numpy as np


def get_radial_land(
    world,
    lat,
    lon,
    goal_lat=None,
    goal_lon=None
):
    """
    Extract radial land information around the ship using
    the exact same sampling geometry as get_radial_weather().

    Returns:
        np.ndarray of shape (114,)

    Values:
        0 = ocean
        1 = land

    The exact goal cell is treated as navigable even if the
    underlying land mask marks it as land.
    """

    ai, aj = world.lat_lon_to_grid(lat, lon)

    H, W = world.height, world.width

    result = []

    radii = [3, 8, 15, 30]

    # ======================================================
    # Convert goal to grid coordinates
    # ======================================================

    goal_i = None
    goal_j = None

    if goal_lat is not None and goal_lon is not None:

        goal_i, goal_j = world.lat_lon_to_grid(
            goal_lat,
            goal_lon
        )

    # ======================================================
    # Sample the same concentric circles as weather
    # ======================================================

    for r in radii:

        num_points = max(8, r * 2)

        for k in range(num_points):

            angle = 2 * np.pi * k / num_points

            i = int(
                round(
                    ai + r * np.sin(angle)
                )
            )

            j = int(
                round(
                    aj + r * np.cos(angle)
                )
            )

            # Longitude wraps around globe
            j = j % W

            # Latitude does not wrap
            if 0 <= i < H:

                # ==================================================
                # Is this point the goal?
                # ==================================================

                is_goal = False

                if goal_i is not None and goal_j is not None:

                    # Longitude difference must account
                    # for wrapping around the dateline.
                    j_difference = j - goal_j

                    if j_difference > W / 2:
                        j_difference -= W

                    elif j_difference < -W / 2:
                        j_difference += W

                    if (
                        i == goal_i
                        and j_difference == 0
                    ):
                        is_goal = True

                # ==================================================
                # Land value
                # ==================================================

                if is_goal:

                    # The goal itself is NOT considered
                    # an obstacle.
                    land_value = 0.0

                else:

                    land_value = float(
                        world.land_mask[i, j]
                    )

            else:

                # Outside valid latitude range.
                # Treat it as non-land for now.
                land_value = 0.0

            result.append(land_value)

    return np.asarray(
        result,
        dtype=np.float32
    )