import json
import os

import matplotlib.pyplot as plt
import geopandas as gpd

from environment.world_generator import WorldGrid
from environment.land_mask import LandMask


ROUTES_FILE = "training_routes.json"
SHAPEFILE = "ne_50m_land/ne_50m_land.shp"

OUTPUT_FILE = "training_routes_plot.png"


def load_routes(filename):

    with open(filename, "r") as f:
        routes = json.load(f)

    return routes

def classify_point(
    world,
    land_mask,
    lat,
    lon
):

    # Convert coordinate to grid cell
    i, j = world.lat_lon_to_grid(lat, lon)


    if i < 0 or i >= world.height:
        return "invalid"

    j = j % world.width



    if land_mask[i, j] == 1:
        return "land"


    for di in [-1, 0, 1]:

        for dj in [-1, 0, 1]:

            if di == 0 and dj == 0:
                continue

            ni = i + di
            nj = (j + dj) % world.width

            if ni < 0 or ni >= world.height:
                continue

            if land_mask[ni, nj] == 1:
                return "coast"

    return "sea"


def plot_routes():


    routes = load_routes(ROUTES_FILE)

    print(f"Loaded {len(routes)} routes.")

    world = WorldGrid(
        lat_min=-90,
        lat_max=90,
        lon_min=-180,
        lon_max=180,
        resolution=1.0
    )


    print("Generating land mask...")

    land_generator = LandMask()

    land_mask = land_generator.generate_mask(world)

    print("Land mask generated.")


    land_gdf = gpd.read_file(SHAPEFILE)


    fig, ax = plt.subplots(
        figsize=(20, 10)
    )

    # Plot land
    land_gdf.plot(
        ax=ax,
        edgecolor="black",
        linewidth=0.4
    )


    statistics = {
        "sea_sea": 0,
        "sea_coast": 0,
        "coast_sea": 0,
        "coast_coast": 0,
        "invalid": 0
    }

    for route in routes:

        route_name = route["name"]
        route_type = route["type"]

        start_lat, start_lon = route["start"]
        goal_lat, goal_lon = route["goal"]



        start_class = classify_point(
            world,
            land_mask,
            start_lat,
            start_lon
        )

        goal_class = classify_point(
            world,
            land_mask,
            goal_lat,
            goal_lon
        )



        actual_type = (
            f"{start_class}_{goal_class}"
        )

        if (
            start_class == "land"
            or goal_class == "land"
            or start_class == "invalid"
            or goal_class == "invalid"
        ):

            statistics["invalid"] += 1

            print(
                f"WARNING: {route_name}"
            )

            print(
                f"    Start: "
                f"{start_lat}, {start_lon} "
                f"-> {start_class}"
            )

            print(
                f"    Goal:  "
                f"{goal_lat}, {goal_lon} "
                f"-> {goal_class}"
            )

        elif actual_type in statistics:

            statistics[actual_type] += 1



        ax.plot(
            [start_lon, goal_lon],
            [start_lat, goal_lat],
            linewidth=0.8,
            alpha=0.5
        )



        if start_class == "coast":
            start_marker = "s"
        else:
            start_marker = "o"

        ax.scatter(
            start_lon,
            start_lat,
            marker=start_marker,
            s=45,
            edgecolor="black",
            linewidth=0.7,
            zorder=5
        )



        if goal_class == "coast":
            goal_marker = "s"
        else:
            goal_marker = "o"

        ax.scatter(
            goal_lon,
            goal_lat,
            marker=goal_marker,
            s=55,
            edgecolor="black",
            linewidth=0.7,
            zorder=5
        )


        ax.text(
            start_lon,
            start_lat,
            route_name,
            fontsize=6,
            alpha=0.7
        )


    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_title(
        "Generated Training Routes"
    )

    ax.grid(
        True,
        alpha=0.3
    )



    print()
    print("=" * 60)
    print("ROUTE VERIFICATION")
    print("=" * 60)

    print(
        f"Sea → Sea:       {statistics['sea_sea']}"
    )

    print(
        f"Sea → Coast:     {statistics['sea_coast']}"
    )

    print(
        f"Coast → Sea:     {statistics['coast_sea']}"
    )

    print(
        f"Coast → Coast:   {statistics['coast_coast']}"
    )

    print(
        f"INVALID:         {statistics['invalid']}"
    )

    print("=" * 60)



    if statistics["invalid"] == 0:

        print(
            "\n✓ ALL ROUTES HAVE VALID OCEAN START/GOAL POINTS."
        )

    else:

        print(
            "\n✗ WARNING: SOME ROUTES CONTAIN LAND POINTS."
        )


    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\nPlot saved to: {OUTPUT_FILE}"
    )

    plt.show()



if __name__ == "__main__":
    plot_routes()