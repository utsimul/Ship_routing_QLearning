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

            i = int(ai + r * np.sin(angle)) #get the x and y points from radius and angle (get x,y from polar coordinates)
            j = int(aj + r * np.cos(angle))

            if 0 <= i < H and 0 <= j < W:

                uu = u[i, j]
                vv = v[i, j]

                speed = np.sqrt(uu**2 + vv**2)
                wind_dir = np.arctan2(vv, uu)

                # relative position
                rel_angle = angle   # already relative to agent
                distance = r

                result.append([
                    speed,
                    np.sin(wind_dir),
                    np.cos(wind_dir),
                    distance,
                    np.sin(rel_angle),
                    np.cos(rel_angle)
                ])

    return np.array(result)

