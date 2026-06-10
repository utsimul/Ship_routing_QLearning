# feed data to agent and obtain its outcome

import numpy as np

from environment.world_generator import WorldGrid
from environment.land_mask import LandMask
from environment.ocean_env import OceanEnvironment
from weather.weather_simulator import WeatherSimulator

from weather_repr.radial_snapshot import get_radial_weather

from agent.random_agent import RandomAgent 
from agent.policy_agent import PolicyAgent

import matplotlib.pyplot as plt
from Helpers import *


EARTH_RADIUS_KM = 6371.0



def main():


    world = WorldGrid(
        lat_min=-90,
        lat_max=90,
        lon_min=-180,
        lon_max=180,
        resolution=1.0
    )

    print("World shape:", world.shape())

    agent = RandomAgent()
    PAgent = PolicyAgent(688) #I printed this to find agent state shape

    print("Generating land mask...")

    mask_generator = LandMask()
    mask_generator.generate_mask(world)

    print("Land mask generated")

    weather = WeatherSimulator(world)

    # initialize first weather frame
    weather.update()

    print("Weather initialized")


    env = OceanEnvironment(world)

    env.reset()

    trajectory = [env.ship_position]

    print("Ship:", env.ship_position)
    print("Goal:", env.goal_position)


    NUM_STEPS = 1000
    NUM_EPISODES = 1
    


    for episode in range(NUM_EPISODES):
        step_counter = 0
        trajectory = [env.ship_position]
        actions = []
        rewards = []

        state = env.reset()
        start_position = env.ship_position

        done = False

        while not done:
            
            ship_lat, ship_lon = env.ship_position


            #CONSTRUCT STATE 
            ship_lat, ship_lon = env.ship_position

            goal_lat, goal_lon = env.goal_position

            start_lat, start_lon = start_position


            radial_weather = get_radial_weather(
                world,
                ship_lat,
                ship_lon
            )

            dist_to_goal = geodesic_distance(
                ship_lat,
                ship_lon,
                goal_lat,
                goal_lon
            )

            dist_from_start = geodesic_distance(
                start_lat,
                start_lon,
                ship_lat,
                ship_lon
            )

            goal_direction = bearing_to_goal(
                ship_lat,
                ship_lon,
                goal_lat,
                goal_lon
            )

            #normalize values
            dist_to_goal /= 20000.0
            dist_from_start /= 20000.0
            #20000 because earth's radius is 20015km

            agent_state = np.concatenate([
                radial_weather.flatten(),
                np.array([
                    dist_to_goal,
                    dist_from_start,
                    np.sin(goal_direction), #encoded as sin to avoid wrapping issues
                    np.cos(goal_direction)
                ])
            ])

            print("Agent state shape:", agent_state.shape)

            theta = PAgent.act(agent_state)

            next_state, reward, done = env.step(theta)
            trajectory.append(env.ship_position)

            print(
                f"Reward {reward:.2f} | "
                f"Position {next_state}"
            )

            state = next_state
            step_counter += 1

            if step_counter >= NUM_STEPS:
                print("Max steps reached, ending episode.")
                break

            # if np.random.rand() < 0.02:
            #     weather._spawn_storm()

            # weather.update()
        
        print("Episode finished")

        plot_episode(
            world,
            env,
            trajectory
        )

        env.reset()

        trajectory = [env.ship_position]

        print("New ship:", env.ship_position)
        print("New goal:", env.goal_position)

        print("Updating agent...")

        #agent.update(actions, rewards)

        print("Agent updated")

        




if __name__ == "__main__":
    main()