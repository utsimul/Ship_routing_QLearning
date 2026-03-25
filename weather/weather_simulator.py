import math

import numpy as np


class WeatherSimulator:

    def __init__(self, world):

        self.world = world

        # load ERA5 base weather
        self.base_weather = np.load("weather/weather_base.npy")
        self.time_index = 0
        # resize if needed
        self.base_weather = self.base_weather[:world.height, :world.width]
        print(f"Base weather shape: {self.base_weather.shape}")
        

        # storm centers
        self.storm_centers = [
            [np.random.randint(0, world.height),
             np.random.randint(0, world.width)]
            for _ in range(3)
        ]

    def update(self):

        H, W = self.world.height, self.world.width

        # --- 1. RESET wind field (CRITICAL) ---
        self.world.wind_u.fill(0.0)
        self.world.wind_v.fill(0.0)

        # --- 2. BASE WEATHER (should be vector field ideally) ---
        base = self.base_weather[self.time_index]
        base = self._resize_to_world(base)

        # If base is scalar → convert to weak background wind
        self.world.wind_u += base * 0.5
        self.world.wind_v += base * 0.3

        # --- 3. APPLY STORMS (MAIN PHYSICS) ---
        for storm in self.world.storm_data:
            si, sj = self.world.lat_lon_to_grid(storm['lat'], storm['lon'])

            radius = storm['radius'] * 10   # scale to grid
            intensity = storm['intensity'] * 5

            for i in range(max(0, si - radius), min(H, si + radius)):
                for j in range(max(0, sj - radius), min(W, sj + radius)):

                    dx = j - sj
                    dy = i - si
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < radius and dist > 0:

                        angle = np.arctan2(dy, dx)

                        # Smooth decay
                        strength = intensity * np.exp(-dist / radius)

                        # Cyclonic rotation (important!)
                        self.world.wind_u[i, j] += -np.sin(angle) * strength
                        self.world.wind_v[i, j] +=  np.cos(angle) * strength

        # --- 4. ADD SMALL TURBULENCE ---
        self.world.wind_u += np.random.normal(0, 0.05, (H, W))
        self.world.wind_v += np.random.normal(0, 0.05, (H, W))

        # --- 5. TIME EVOLUTION ---
        self.time_index = (self.time_index + 1) % len(self.base_weather)

    def _resize_to_world(self, base):

        return np.kron(
            base,
            np.ones((
                self.world.height // base.shape[0] + 1,
                self.world.width // base.shape[1] + 1
            ))
        )[:self.world.height, :self.world.width]
    
    def get_weather_at_lat_lon(self, lat, lon):

        i, j = self.world.lat_lon_to_grid(lat, lon)
        u = self.world.wind_u[i, j]
        v = self.world.wind_v[i, j]

        speed = math.sqrt(u*u + v*v)
        direction = math.degrees(math.atan2(v, u))

        return {
            "speed": speed,
            "direction": direction
        }
    
    def _intensity_to_condition(self, value):

        if value < 0.2:
            return "clear"
        elif value < 0.4:
            return "cloudy"
        elif value < 0.6:
            return "rain"
        elif value < 0.8:
            return "wind"
        else:
            return "storm"