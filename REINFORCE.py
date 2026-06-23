#calculate formula for reward
#weather has to change after every timestep concurrently so we need to update environment after every step.

#take action in another direction when there is random collision. 
import numpy as np

from environment.world_generator import WorldGrid
from environment.land_mask import LandMask
from environment.ocean_env import OceanEnvironment
from weather.weather_simulator import WeatherSimulator

from weather_repr.plot_weather import *
from weather_repr.radial_snapshot import get_radial_weather

from agent.random_agent import RandomAgent 
from agent.policy_agent import PolicyAgent

import matplotlib.pyplot as plt
from Helpers import *


EARTH_RADIUS_KM = 6371.0

START_POS = (20.0, -50.0)
GOAL_POS  = (35.0, -20.0)

def main():


    world = WorldGrid(
        lat_min=-90,
        lat_max=90,
        lon_min=-180,
        lon_max=180,
        resolution=1.0
    )

    print("World shape:", world.shape())

    #agent = RandomAgent()
    PAgent = PolicyAgent(688) #I printed this to find agent state shape

    print("Generating land mask...")

    mask_generator = LandMask()
    mask_generator.generate_mask(world)

    print("Land mask generated")

    weather = WeatherSimulator(world)

    # initialize first weather frame
    weather.update()

    print("Weather initialized")

    fig = Plot_weather(world)
    plt.savefig("plots/weather_initial.png", dpi=300, bbox_inches="tight")
    plt.close()


    env = OceanEnvironment(world)

    env.reset(START_POS[0], START_POS[1], GOAL_POS[0], GOAL_POS[1])

    env.ship_position = START_POS
    env.goal_position = GOAL_POS

    trajectory = [env.ship_position]

    print("Ship:", env.ship_position)
    start_position_constant = env.ship_position
    print("Goal:", env.goal_position)


    NUM_STEPS = 2000
    NUM_EPISODES = 30
    trajectories = []
    


    for episode in range(NUM_EPISODES):
        
        step_counter = 0
        trajectory = [start_position_constant]
        actions = []
        rewards = []
        log_probs = []

        state = env.reset(START_POS[0], START_POS[1], GOAL_POS[0], GOAL_POS[1])
        start_position = start_position_constant
        env.ship_position = start_position


        done = False

        goal_direction = bearing_to_goal(
            env.ship_position[0],
            env.ship_position[1],
            env.goal_position[0],
            env.goal_position[1]
        )

        while not done:


            #CONSTRUCT STATE 
            ship_lat, ship_lon = env.ship_position

            goal_lat, goal_lon = env.goal_position

            start_lat, start_lon = start_position
            #print("current ship position: ", ship_lat, ship_lon)


            radial_weather = get_radial_weather(
                world,
                ship_lat,
                ship_lon
            )

            #print("radial weather first 5 values: ", radial_weather.flatten()[:5])

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


            #print("agent state min and max: ",np.min(agent_state), np.max(agent_state))
            #print(agent_state[:10])

            theta, log_prob = PAgent.act(agent_state)

            #theta = goal_direction

            next_state, reward, done = env.step(theta, radial_weather, dist_to_goal, weather) #i think i need to input weather AT THAT POINT to so that
            #the ship goes with the wind direction, but maybe it might not be that important...?
            rewards.append(reward)

            #JUST TRYING TO FIX learnin AND SEE IF THIS WORKS: 
            #rewards.append(-1*reward)
            log_probs.append(log_prob)


            trajectory.append(env.ship_position)

            # print(
            #     f"Reward {reward:.2f} | "
            #     f"Position {next_state}"
            # )

            state = next_state
            step_counter += 1

            if step_counter >= NUM_STEPS:
                print("Max steps reached, ending episode.")
                break

            # if np.random.rand() < 0.02:
            #     weather._spawn_storm()

            # weather.update()
        

        print("parameters before update")
        for name, param in PAgent.policy.named_parameters():
            print(name, param.data.norm())
        #backward pass
        PAgent.update(
            rewards,
            log_probs
        )

        if episode % 29 == 0:
            fig = plot_episode(
                world,
                env,
                trajectory
            )
            plt.savefig(f"plots/episode_{episode}.png",
                dpi=300,
                bbox_inches="tight")
            plt.close()
        
        trajectories.append(trajectory)

        env.reset(START_POS[0], START_POS[1], GOAL_POS[0], GOAL_POS[1])

        

        print("New ship:", env.ship_position)
        print("New goal:", env.goal_position)

        #calculate reward

        #print("Updating agent...")
        #backprop after every episode

        #agent.update(actions, rewards)

        #print("Agent updated")
        print("---------------------EPISODE FINISHED---------------------")

        




if __name__ == "__main__":
    main()