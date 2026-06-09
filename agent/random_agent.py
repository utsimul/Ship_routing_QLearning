import numpy as np


class RandomAgent:

    def act(self, state):

        
        theta = np.random.uniform(0, 2*np.pi) # angle is in radians

        return theta