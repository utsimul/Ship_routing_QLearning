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

        # self.world.wind_u = np.zeros((self.world.height, self.world.width))
        # self.world.wind_v = np.zeros((self.world.height, self.world.width))

        # --- 1. SMOOTH TEMPORAL EVOLUTION (CRITICAL FOR RL) ---
        alpha = 0.9  # memory factor (0.8–0.98 good range)

        prev_u = self.world.wind_u.copy()
        prev_v = self.world.wind_v.copy()

        new_u = np.zeros((H, W))
        new_v = np.zeros((H, W))

        # --- 2. BASE WEATHER (should be vector field ideally) ---
        base = self.base_weather[self.time_index]
        base = self._resize_to_world(base)

        # If base is scalar → convert to weak background wind
        new_u += base * 0.5
        new_v += base * 0.3

        # --- 3. APPLY STORMS (MAIN PHYSICS) ---
        for storm in self.world.storm_data:
            si, sj = self.world.lat_lon_to_grid(storm['lat'], storm['lon'])

            radius = int(round(storm['radius']))
            intensity = storm['intensity'] * 5

            radius = int(round(storm['radius']))

            for i in range(max(0, si - radius), min(H, si + radius)):
                for j in range(max(0, sj - radius), min(W, sj + radius)):

                    dx = j - sj
                    dy = i - si
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < radius and dist > 0:

                        angle = np.arctan2(dy, dx)

                        # Smooth decay - gaussian storms
                        strength = intensity * np.exp(-(dist**2) / (2 * (radius**2)))

                        # Cyclonic rotation (important!)
                        self.world.wind_u[i, j] += -np.sin(angle) * strength
                        self.world.wind_v[i, j] +=  np.cos(angle) * strength

        # --- 4. ADD SMALL TURBULENCE ---
        self.world.wind_u = alpha * prev_u + (1 - alpha) * new_u
        self.world.wind_v = alpha * prev_v + (1 - alpha) * new_v

        # --- 5. TIME EVOLUTION ---
        if np.random.rand() < 0.2: 
            self.time_index = (self.time_index + 1) % len(self.base_weather)
    
    def _spawn_storm(self):
        lat = np.random.uniform(self.world.min_lat, self.world.max_lat)
        lon = np.random.uniform(self.world.min_lon, self.world.max_lon)

        storm = {
            'lat': lat,
            'lon': lon,
            'intensity': np.random.uniform(1.5, 4.0),   # controls wind strength
            'radius': np.random.uniform(2, 6),      # influence area (grid units)
        }

        self.world.storm_data.append(storm)

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