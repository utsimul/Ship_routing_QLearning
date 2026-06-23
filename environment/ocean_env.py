import numpy as np
from Helpers import *
from colorama import Fore, Back, Style, init

init(autoreset=True)


class OceanEnvironment:

    def __init__(self, world, step_scale=1.0):

        self.world = world
        self.ship_position = None
        self.goal_position = None

        # step size based on world resolution (pre decided)
        self.lat_step = world.resolution * step_scale
        self.lon_step = world.resolution * step_scale

        ocean_cells = np.argwhere(self.world.land_mask == 0)

        self.start = ocean_cells[np.random.choice(len(ocean_cells))] #here start and goal are defined
        self.goal = ocean_cells[np.random.choice(len(ocean_cells))]


    def reset(self, start_lat, start_lon, goal_lat, goal_lon):


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


    def step(self, theta, weather, dist_to_goal, weather_simulator):

        lat, lon = self.ship_position
        #print(Fore.BLUE + "taking action in direction: " + str(theta) + Fore.RESET)

        # action now yields continuous values of theta that the ship moves in -> also we assume 
        # that the distance the ship should move in is part of the state (like the agent can't decide that)
        #the agent just tells the direction the ship should move in -> and it inherently KNOWS that the agent 
        #will probably do this much based on the previous history of speeds. 
        #agent just outputs the direction component of velocity - ship decides the magnitude.
        new_lat = lat + self.lat_step * np.cos(theta)
        new_lon = lon + self.lon_step * np.sin(theta)


        # latitude does not wrap on Earth
        new_lat = np.clip(
            new_lat,
            self.world.lat_min,
            self.world.lat_max
        )

        # longitude wraps around the globe
        lon_range = (
            self.world.lon_max -
            self.world.lon_min
        )

        new_lon = (
            (new_lon - self.world.lon_min)
            % lon_range
        ) + self.world.lon_min

        grid_i, grid_j = self.latlon_to_grid(new_lat, new_lon)

        # check land collision
        if self.world.land_mask[grid_i, grid_j] == 1:

            reward = -100 #increasing this to -100 did make the agent not to collide at the end.
            done = False

            print("land collision at: ", (new_lat, new_lon))
            return self.ship_position, reward, done

        else:

            goal_dist = np.sqrt(
                (new_lat - self.goal_position[0])**2 +
                (new_lon - self.goal_position[1])**2
            )

            if goal_dist < self.world.resolution:

                done = True

        

        self.ship_position = (new_lat, new_lon)

        #---------------CALCULATE REWARDS FROM NEW POSITION -----------------
        
        new_dist_to_goal = geodesic_distance(
            new_lat,
            new_lon,
            self.goal_position[0],
            self.goal_position[1]
        )
    
        distance_reward = dist_to_goal - new_dist_to_goal
        distance_reward /= 100.0 #normalize?

        #weather reward
        weather_data = weather_simulator.get_weather_at_lat_lon(
            new_lat,
            new_lon
        )

        wind_speed = weather_data["speed"]

        wind_direction = np.radians(
            weather_data["direction"]
        )

        # ship heading unit vector
        ship_dx = np.cos(theta)
        ship_dy = np.sin(theta)

        # wind unit vector
        wind_dx = np.cos(wind_direction)
        wind_dy = np.sin(wind_direction)

        # dot product
        alignment = (
            ship_dx * wind_dx +
            ship_dy * wind_dy
        )

        # numerical safety
        alignment = np.clip(alignment, -1.0, 1.0)

        weather_cost = (
            wind_speed *
            (1.0 - alignment)
        )

        # normalize
        weather_cost /= 20.0

        #print("distance reward: ", distance_reward)
        #print("weather cost: ", weather_cost)

        reward = (
            distance_reward
            - 0.25 * weather_cost
        )

        print("reward: ", reward)

        # goal bonus
        if new_dist_to_goal < self.world.resolution:
            reward += 100.0
            done = True
        else:
            done = False

        print("updated ship position and calculated reward: " , reward)

        return self.ship_position, reward, done


    def get_state(self):

        return {
            "ship": self.ship_position,
            "goal": self.goal_position,
            "land_mask": self.world.land_mask,
            "weather": self.world.weather
        }