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

        storm_field = np.zeros((self.world.height, self.world.width))

        for cx, cy in self.storm_centers:

            for i in range(self.world.height):
                for j in range(self.world.width):

                    dist = (i-cx)**2 + (j-cy)**2
                    storm_field[i, j] += np.exp(-dist/200)

        # move storms slowly
        for center in self.storm_centers:

            center[0] += np.random.randint(-1, 2)
            center[1] += np.random.randint(-1, 2)

            center[0] = np.clip(center[0], 0, self.world.height-1)
            center[1] = np.clip(center[1], 0, self.world.width-1)

        # small random fluctuation
        noise = np.random.normal(0, 0.02, (self.world.height, self.world.width))

        base = self.base_weather[self.time_index]
        base = self._resize_to_world(base)
        
        weather = base + storm_field + noise

        weather = np.clip(weather, 0, None)

        self.world.weather = weather

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

        i = np.clip(i, 0, self.world.height - 1)
        j = np.clip(j, 0, self.world.width - 1)

        intensity = self.world.weather[i, j]

        condition = self._intensity_to_condition(intensity)

        return {
            "intensity": float(intensity),
            "condition": condition,
            "value": float(intensity)   # add this
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