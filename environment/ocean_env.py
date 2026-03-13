import numpy as np


class OceanEnvironment:

    def __init__(self, world, step_scale=1.0):

        self.world = world
        self.ship_position = None
        self.goal_position = None

        # step size based on world resolution (pre decided)
        self.lat_step = world.resolution * step_scale
        self.lon_step = world.resolution * step_scale


    def reset(self):

        ocean_cells = np.argwhere(self.world.land_mask == 0)

        start = ocean_cells[np.random.choice(len(ocean_cells))]
        goal = ocean_cells[np.random.choice(len(ocean_cells))]

        start_lat, start_lon = self.world.get_coordinates(*start)
        goal_lat, goal_lon = self.world.get_coordinates(*goal)

        self.ship_position = (start_lat, start_lon)
        self.goal_position = (goal_lat, goal_lon)

        return self.ship_position


    def set_pos(self, start, goal):

        self.ship_position = start
        self.goal_position = goal

        return self.ship_position


    def latlon_to_grid(self, lat, lon):

        i = int((lat - self.world.lat_min) / self.world.resolution)
        j = int((lon - self.world.lon_min) / self.world.resolution)

        i = np.clip(i, 0, self.world.height - 1)
        j = np.clip(j, 0, self.world.width - 1)

        return i, j


    def step(self, theta):

        lat, lon = self.ship_position

        # action now yields continuous values of theta that the ship moves in -> also we assume 
        # that the distance the ship should move in is part of the state (like the agent can't decide that)
        #the agent just tells the direction the ship should move in -> and it inherently KNOWS that the agent 
        #will probably do this much based on the previous history of speeds. 
        #agent just outputs the direction component of velocity - ship decides the magnitude.
        new_lat = lat + self.lat_step * np.cos(theta)
        new_lon = lon + self.lon_step * np.sin(theta)

        # keep inside bounds
        new_lat = np.clip(new_lat, self.world.lat_min, self.world.lat_max)
        new_lon = np.clip(new_lon, self.world.lon_min, self.world.lon_max)

        grid_i, grid_j = self.latlon_to_grid(new_lat, new_lon)

        # check land collision
        if self.world.land_mask[grid_i, grid_j] == 1:

            reward = -100
            done = True

        else:

            goal_dist = np.sqrt(
                (new_lat - self.goal_position[0])**2 +
                (new_lon - self.goal_position[1])**2
            )

            if goal_dist < self.world.resolution:

                reward = 100
                done = True

            else:

                reward = -1
                done = False

        self.ship_position = (new_lat, new_lon)

        return self.ship_position, reward, done


    def get_state(self):

        return {
            "ship": self.ship_position,
            "goal": self.goal_position,
            "land_mask": self.world.land_mask,
            "weather": self.world.weather
        }