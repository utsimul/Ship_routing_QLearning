import numpy as np


ACTIONS = {
    0: (-1, 0),   # N
    1: (-1, 1),   # NE
    2: (0, 1),    # E
    3: (1, 1),    # SE
    4: (1, 0),    # S
    5: (1, -1),   # SW
    6: (0, -1),   # W
    7: (-1, -1)   # NW
}


class OceanEnvironment:

    def __init__(self, world):

        self.world = world
        self.ship_position = None
        self.goal_position = None

    def reset(self):

        ocean_cells = np.argwhere(self.world.land_mask == 0)

        start = ocean_cells[np.random.choice(len(ocean_cells))]
        goal = ocean_cells[np.random.choice(len(ocean_cells))]

        self.ship_position = tuple(start)
        self.goal_position = tuple(goal)

        return self.ship_position


    def set_pos(self, start, goal):

        self.ship_position = tuple(start)
        self.goal_position = tuple(goal)

        return self.ship_position


    def step(self, action):

        dx, dy = ACTIONS[action]

        x, y = self.ship_position

        while True:
            new_x = x + dx
            new_y = y + dy

            # clipping to keep inside grid
            new_x = np.clip(new_x, 0, self.world.height - 1)
            new_y = np.clip(new_y, 0, self.world.width - 1)

            diagonal_move = abs(dx) + abs(dy) == 2

            # diagonal corner correction / corner cutting avoidance
            if diagonal_move:

                adj1 = (x + dx, y)
                adj2 = (x, y + dy)

                adj1_land = (
                    0 <= adj1[0] < self.world.height and
                    0 <= adj1[1] < self.world.width and
                    self.world.land_mask[adj1] == 1
                )

                adj2_land = (
                    0 <= adj2[0] < self.world.height and
                    0 <= adj2[1] < self.world.width and
                    self.world.land_mask[adj2] == 1
                )

                if adj1_land and adj2_land:
                    # block diagonal move
                    new_x, new_y = x, y
                else:
                    break

        # check terrain 
        if self.world.land_mask[new_x, new_y] == 1:

            reward = -100
            done = True

        elif (new_x, new_y) == self.goal_position:

            reward = 100
            done = True

        else:

            # movement cost
            if diagonal_move:
                step_cost = 1.41
            else:
                step_cost = 1

            reward = -step_cost
            done = False

        self.ship_position = (new_x, new_y)

        return self.ship_position, reward, done


    def get_state(self):

        return {
            "ship": self.ship_position,
            "goal": self.goal_position,
            "land_mask": self.world.land_mask,
            "weather": self.world.weather
        }