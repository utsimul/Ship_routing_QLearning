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