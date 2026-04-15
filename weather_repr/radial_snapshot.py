import numpy as np

def get_radial_weather(world, lat, lon):

    ai, aj = world.lat_lon_to_grid(lat, lon)

    u = world.wind_u
    v = world.wind_v

    H, W = world.height, world.width

    samples_data = []
    samples_coords = []

    radii = [3, 8, 15, 30]

    for r in radii:
        num_points = max(8, r * 2)

        for k in range(num_points):
            angle = 2 * np.pi * k / num_points

            i = int(ai + r * np.sin(angle))
            j = int(aj + r * np.cos(angle))

            if 0 <= i < H and 0 <= j < W:

                # --- wind ---
                uu = u[i, j]
                vv = v[i, j]

                speed = np.sqrt(uu**2 + vv**2)
                direction = np.arctan2(vv, uu)

                # ✅ Option B (best)
                encoded = [speed, np.sin(direction), np.cos(direction)]

                samples_data.append(encoded)

                lat_pt, lon_pt = world.get_coordinates(i, j)
                samples_coords.append((lat_pt, lon_pt, uu, vv))

    return np.array(samples_data), samples_coords