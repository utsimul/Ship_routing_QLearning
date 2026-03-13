import geopandas as gpd
from shapely.geometry import Point


class LandMask:

    def __init__(self, shapefile_path="ne_50m_land/ne_50m_land.shp"):

        land_gdf = gpd.read_file(shapefile_path)

        # merge all land polygons into one
        self.land = land_gdf.union_all()


    def is_land(self, lat, lon):

        point = Point(lon, lat)

        return self.land.contains(point)


    def generate_mask(self, world_grid):

        for i in range(world_grid.height):
            for j in range(world_grid.width):

                lat, lon = world_grid.get_coordinates(i, j)

                if self.is_land(lat, lon):
                    world_grid.land_mask[i, j] = 1
                else:
                    world_grid.land_mask[i, j] = 0

        return world_grid.land_mask