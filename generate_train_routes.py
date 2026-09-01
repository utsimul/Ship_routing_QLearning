"""I am allowing 4 different kinds of combinations of start and goal points:
    1. SEA -> SEA
    2. SEA -> COAST
    3. COAST -> SEA
    4. COAST -> COAST"""

import json
import random
import numpy as np

from environment.world_generator import WorldGrid
from environment.land_mask import LandMask


NUM_SEA_SEA = 15
NUM_SEA_COAST = 15
NUM_COAST_SEA = 15
NUM_COAST_COAST = 15

OUTPUT_FILE = "training_routes.json"

RANDOM_SEED = 42

# A route should not start and end almost at the same location.
MIN_DISTANCE_DEGREES = 10.0


def classify_points(world, land_mask):
    """
    Classify every grid cell into:

        LAND
        SEA
        COAST

    COAST means:
        - the current point itself is NOT land
        - at least one neighboring grid cell IS land

    This means every COAST point is still a valid ocean
    position for the ship.
    """

    H, W = world.height, world.width

    sea_points = []
    coast_points = []

    for i in range(H):

        for j in range(W):


            if land_mask[i, j] == 1:
                continue

            # Current point is sea.
            # Check neighboring cells.
            is_coast = False

            for di in [-1, 0, 1]:

                for dj in [-1, 0, 1]:

                    # Skip the point itself
                    if di == 0 and dj == 0:
                        continue

                    ni = i + di
                    nj = j + dj

                    # Longitude wraps around the globe
                    nj = nj % W

                    # Latitude does NOT wrap
                    if ni < 0 or ni >= H:
                        continue

                    if land_mask[ni, nj] == 1:
                        is_coast = True
                        break

                if is_coast:
                    break

            lat, lon = world.get_coordinates(i, j)

            point = (float(lat), float(lon))

            if is_coast:
                coast_points.append(point)
            else:
                sea_points.append(point)

    return sea_points, coast_points


def coordinate_distance(p1, p2):

    lat1, lon1 = p1
    lat2, lon2 = p2

    dlat = lat2 - lat1

    # Correct longitude wrapping
    dlon = abs(lon2 - lon1)

    if dlon > 180:
        dlon = 360 - dlon

    return np.sqrt(dlat ** 2 + dlon ** 2)


def generate_routes(
    start_points,
    goal_points,
    route_type,
    num_routes,
    existing_routes
):

    routes = []

    attempts = 0
    max_attempts = num_routes * 1000

    while len(routes) < num_routes and attempts < max_attempts:

        attempts += 1

        start = random.choice(start_points)
        goal = random.choice(goal_points)

        # To avoid creating an almost-zero-length journey
        if coordinate_distance(start, goal) < MIN_DISTANCE_DEGREES:
            continue

        # Avoid duplicate start-goal combinations
        duplicate = False

        for route in existing_routes + routes:

            if (
                route["start"] == list(start)
                and
                route["goal"] == list(goal)
            ):
                duplicate = True
                break

        if duplicate:
            continue

        route_number = len(existing_routes) + len(routes) + 1

        route = {
            "name": f"{route_type}_{route_number}",

            "type": route_type,

            "start": [
                start[0],
                start[1]
            ],

            "goal": [
                goal[0],
                goal[1]
            ]
        }

        routes.append(route)

    if len(routes) < num_routes:

        print(
            f"WARNING: Could only generate "
            f"{len(routes)}/{num_routes} routes for "
            f"{route_type}"
        )

    return routes



def main():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)



    world = WorldGrid(
        lat_min=-90,
        lat_max=90,
        lon_min=-180,
        lon_max=180,
        resolution=1.0
    )

    print("World shape:", world.shape())


    print("Generating land mask...")

    land_generator = LandMask()

    land_mask = land_generator.generate_mask(world)

    print("Land mask generated.")

    print("Classifying sea and coast points...")

    sea_points, coast_points = classify_points(
        world,
        land_mask
    )

    print(f"Sea points:   {len(sea_points)}")
    print(f"Coast points: {len(coast_points)}")


    all_routes = []

    # 1. SEA -> SEA
    routes = generate_routes(
        sea_points,
        sea_points,
        "sea_sea",
        NUM_SEA_SEA,
        all_routes
    )

    all_routes.extend(routes)

    # 2. SEA -> COAST
    routes = generate_routes(
        sea_points,
        coast_points,
        "sea_coast",
        NUM_SEA_COAST,
        all_routes
    )

    all_routes.extend(routes)

    # 3. COAST -> SEA
    routes = generate_routes(
        coast_points,
        sea_points,
        "coast_sea",
        NUM_COAST_SEA,
        all_routes
    )

    all_routes.extend(routes)

    # 4. COAST -> COAST
    routes = generate_routes(
        coast_points,
        coast_points,
        "coast_coast",
        NUM_COAST_COAST,
        all_routes
    )

    all_routes.extend(routes)


    with open(OUTPUT_FILE, "w") as f:

        json.dump(
            all_routes,
            f,
            indent=4
        )


    print()
    print("=" * 60)
    print("TRAINING ROUTES GENERATED")
    print("=" * 60)

    print(f"Total routes: {len(all_routes)}")

    for route_type in [
        "sea_sea",
        "sea_coast",
        "coast_sea",
        "coast_coast"
    ]:

        count = sum(
            r["type"] == route_type
            for r in all_routes
        )

        print(
            f"{route_type:15s}: {count}"
        )

    print("=" * 60)

    print(
        f"\nSaved routes to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()