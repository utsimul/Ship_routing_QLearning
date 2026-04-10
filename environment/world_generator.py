import numpy as np

class WorldGrid:

    def __init__(self,
                 lat_min,
                 lat_max,
                 lon_min,
                 lon_max,
                 resolution):

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.resolution = resolution

        self.lats = np.arange(lat_min, lat_max, resolution)
        self.lons = np.arange(lon_min, lon_max, resolution)

        self.height = len(self.lats)
        self.width = len(self.lons)

        self.land_mask = np.zeros((self.height, self.width))
        self.weather = np.zeros((self.height, self.width))
        # generates environment grid initialized to zero values
        #weather and land_mask both have index - meaning that they are location specific. 
        # #For example, weather[i][j] gives the weather condition at the location corresponding to latitudes[i] and longitudes[j].

    def get_coordinates(self, i, j):
        return self.lats[i], self.lons[j]

    def shape(self):
        return (self.height, self.width)
    
    def lat_lon_to_grid(self, lat, lon):
        i = int((lat - self.lat_min) / self.resolution)
        j = int((lon - self.lon_min) / self.resolution)

        i = np.clip(i, 0, self.height - 1)
        j = np.clip(j, 0, self.width - 1)

        return i, j
        