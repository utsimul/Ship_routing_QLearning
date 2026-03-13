from environment.land_mask import LandMask
from environment.world_generator import WorldGrid
from environment.ocean_env import OceanEnvironment


world = WorldGrid(
    lat_min=10,
    lat_max=30,
    lon_min=50,
    lon_max=75,
    resolution=0.5
)

mask = LandMask()
print("Land mask generated")

mask.generate_mask(world)
print("Land mask applied to world grid")

env = OceanEnvironment(world)
print("Ocean environment initialized")

env.reset()

print("Grid shape:", world.shape())
print("Ship:", env.ship_position)
print("Goal:", env.goal_position)