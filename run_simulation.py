import pygame
import time

from agent.random_agent import RandomAgent
from weather.weather_simulator import WeatherSimulator
from GUI.simulator_gui import runGUI
from environment.world_generator import WorldGrid
from environment.land_mask import LandMask
from environment.ocean_env import OceanEnvironment




def run(world, env):

    agent = RandomAgent()
    weather = WeatherSimulator(world)
    gui = runGUI(world, env)

    clock = pygame.time.Clock()

    running = True

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        weather.update()

        state = env.get_state()

        theta = agent.act(state)

        pos, reward, done = env.step(theta)

        gui.path.append(pos)

        gui.draw()

        if done:
            env.reset()
            gui.path.clear()

        clock.tick(5)   # simulation speed


    pygame.quit()

class SimulatorGUI:

    def __init__(self, lat_min, lat_max, lon_min, lon_max, resolution):

        self.world = WorldGrid(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        resolution=resolution
        )
        
        self.mask = LandMask()
        self.mask.generate_mask(self.world)

        self.env = OceanEnvironment(self.world)
        self.env.reset()

simGUI = SimulatorGUI(
    lat_min=0,
    lat_max=10,
    lon_min=0,
    lon_max=10,
    resolution=0.1
)

if __name__ == "__main__":
    import pygame
    pygame.init()
    run(simGUI.world, simGUI.env)