import numpy as np


class WeatherSimulator:

    def __init__(self, world):

        self.world = world

        self.storm_centers = [
            [np.random.randint(0, world.height),
             np.random.randint(0, world.width)]
            for _ in range(3)
        ]

    def update(self):

        weather = np.zeros((self.world.height, self.world.width))

        for cx, cy in self.storm_centers:

            for i in range(self.world.height):
                for j in range(self.world.width):

                    dist = (i-cx)**2 + (j-cy)**2
                    weather[i, j] += np.exp(-dist/200)

        # move storms slowly
        for center in self.storm_centers:

            center[0] += np.random.randint(-1,2)
            center[1] += np.random.randint(-1,2)

            center[0] = np.clip(center[0],0,self.world.height-1)
            center[1] = np.clip(center[1],0,self.world.width-1)

        weather = np.clip(weather,0,1)

        self.world.weather = weather