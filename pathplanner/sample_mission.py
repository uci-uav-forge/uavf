import json, pyproj
from forge_pathplanner import bdc, prm, surface, polygon, rrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import RegularGridInterpolator

# From https://kespry.force.com/s/article/Finding-Your-State-Plane-Coordinate-System
CRS = {
    "Irvine": pyproj.CRS.from_epsg(2230),
    "Langley": pyproj.CRS.from_epsg(2283),
    "WGS84": pyproj.CRS.from_epsg(4326),
}

FT2METER = 3.28084
METER2FT = 0.3048


class Mission(object):
    def __init__(self, mission_json, wgs2loc, loc2wgs):
        self.mission_json = mission_json
        self.wgs2loc = wgs2loc
        self.loc2wgs = loc2wgs

    def l_get_boundaries(self):
        bounds = []
        for fz in self.mission_json["flyZones"]:
            fzbounds = []
            for bp in fz["boundaryPoints"]:
                # get latitude, longitude in WGS84
                lat84 = bp["latitude"]
                long84 = bp["longitude"]
                # Convert to local
                latloc, longloc = wgs2loc.transform(lat84, long84)
                fzbounds.append([latloc, longloc])
            # closed loop
            fzbounds.append(fzbounds[0])
            bounds.append(fzbounds)
        return np.array(bounds)

    def l_get_search_boundaries(self):
        bounds = []
        for sgp in self.mission_json["searchGridPoints"]:
            # get lat long in WGS84
            lat84 = sgp["latitude"]
            long84 = sgp["longitude"]
            # convert to local
            latloc, longloc = wgs2loc.transform(lat84, long84)
            bounds.append([latloc, longloc])
        # closed loop
        bounds.append(bounds[0])
        return np.array(bounds)

    def l_get_ordered_waypoints(self):
        waypoints = []
        for wp in self.mission_json["waypoints"]:
            lat84, long84, h = wp["latitude"], wp["longitude"], wp["altitude"]
            latloc, longloc = wgs2loc.transform(lat84, long84)
            waypoints.append([latloc, longloc, h])
        return np.array(waypoints)

    def l_get_obstacles(self):
        obstacles = []
        for obs in self.mission_json["stationaryObstacles"]:
            lat84 = obs["latitude"]
            long84 = obs["longitude"]
            radius = obs["radius"]
            height = obs["height"]
            latloc, longloc = wgs2loc.transform(lat84, long84)
            obstacles.append(
                {
                    "x": latloc,
                    "y": longloc,
                    "r": radius,
                    "h": height,
                }
            )
        return obstacles


def grid_over_poly(poly, step, buffer):
    """get grid with cell size `step` over collection of points `poly`. `buffer`
    is how much in excess of the outer perimeter the grid should be."""
    xmin, xmax = poly[:, 0].min() - buffer, poly[:, 0].max() + buffer
    ymin, ymax = poly[:, 1].min() - buffer, poly[:, 1].max() + buffer
    xrange = xmin, xmax
    yrange = ymin, ymax
    return surface.generate_xy_grid(xrange, yrange, step)


if __name__ == "__main__":
    # make figures
    fg1 = plt.figure(figsize=(9, 9))
    ax1 = fg1.add_subplot(1, 1, 1)
    fg2 = plt.figure(figsize=(9, 9))
    ax2 = fg2.add_subplot(1, 1, 1, projection="3d")
    fg3 = plt.figure(figsize=(9, 5))
    ax3 = fg3.add_subplot(1, 1, 1)

    # open mission JSON
    with open("example_mission.json", "r") as f:
        mission_json = json.load(f)

    # define translations
    wgs2loc = pyproj.Transformer.from_crs(CRS["WGS84"], CRS["Langley"])
    loc2wgs = pyproj.Transformer.from_crs(CRS["Langley"], CRS["WGS84"])

    # make mission from json
    mission = Mission(mission_json, wgs2loc, loc2wgs)

    # boundaries
    boundaries = mission.l_get_boundaries()[0]
    obstacles = mission.l_get_obstacles()
    searchgrid = mission.l_get_search_boundaries()
    waypoints = mission.l_get_ordered_waypoints()

    # plot obstacles as circles
    for o in obstacles:
        circ = Circle((o["x"], o["y"]), o["r"], fc="grey")
        ax1.add_artist(circ)

    # plot outer boundaries and search grid
    ax1.plot(boundaries[:, 0], boundaries[:, 1], lw=2, c="k", label="Outer Border")
    ax1.plot(
        searchgrid[:, 0], searchgrid[:, 1], lw=1, ls="--", c="g", label="Search Grid"
    )

    # step size -- this is the grid size, in feet
    step = 40

    # make a grid over the polygon
    X, Y = grid_over_poly(boundaries, step, 250)
    print("Made {} grid".format(X.shape))

    # reformat the list of obstacles as [((x, y), r, h), ... ]
    obs_list = []
    for o in obstacles:
        obs_list.append(((o["x"], o["y"]), o["r"], o["h"]))

    # make the ground
    Hground = surface.place_obstacles(X, Y, obs_list)

    # make the sheet
    Hsheet = surface.get_optimal_grid(
        X, Y, Hground, 50, 0.6, 0.09, 100, step, waypoints=waypoints, solver=None
    )

    # plot ground, sheet in 3d, 2d
    surface.plot_mpl3d(ax2, X, Y, Hground, Hsheet, wireframe=True)
    surface.plot_mpl_2d(ax1, X, Y, Hsheet, levels=40)

    # plot each waypoint
    dist, n, route_step = 0.0, 0, 2.5
    interp = RegularGridInterpolator((X[0, :], Y[:, 0]), Hsheet.T)
    # plot of route distance, heights
    route_d, route_h = [], []
    # scatter plot of waypoint distance, heights
    waypoint_d, waypoint_h = [], []
    for i, a in enumerate(waypoints[:-1]):
        print("Path {} of {}".format(i, waypoints.shape[0]))
        n += 1
        # next waypoint
        b = waypoints[i + 1]

        hcost = 1e5
        srrt = rrt.SheetRRT(X.shape, X, Y, Hsheet, hcost)
        vstart, vend = srrt.make(a[:2], b[:2], 150, np.linalg.norm(a[:2] - b[:2]))
        path = srrt.get_path(vstart, vend)
        ax1.plot(path[:, 0], path[:, 1])

        h_on_sheet = interp((a[0], a[1]), method="linear")
        # draw waypoints on 2d map
        ax1.text(a[0], a[1], str("p {}".format(i)))

        # calculate points for route between waypoints
        route_dist = np.linalg.norm(b[:2] - a[:2])
        route_steps = int(np.ceil(route_dist / route_step))
        route_points = np.linspace(a[:2], b[:2], num=route_steps)

        # fill array for waypoint route chart
        waypoint_d.append(dist)
        waypoint_h.append(a[2])
        # put waypoint labels on route chart
        ax3.text(dist, h_on_sheet, str("p {}".format(n)))

        # fill arrays for route chart
        for rp in route_points:
            rph = interp((rp[0], rp[1]), method="linear")
            route_d.append(dist)
            route_h.append(rph)
            dist += route_step

    # scatter plot of each waypoint on route
    ax3.scatter(waypoint_d, waypoint_h, label="Waypoints", c="r", marker="*")
    # line plot of route
    ax3.plot(route_d, route_h, c="k", label="Route")
    ax3.set_ylabel("Route Height")
    ax3.set_xlabel("Route Distance")
    # scale aspect ratio: route distance / max route height
    aspect = (max(route_d) - min(route_d)) / (max(route_h) - min(route_h))
    print(aspect)
    ax3.set_aspect(aspect / 5)
    ax3.legend()

    # Draw waypoints on 2d Plot
    ax1.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        marker="*",
        s=50,
        c="r",
        label="Waypoints",
    )

    # Draw waypoints on 3d Plot
    ax2.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        marker="*",
        s=50,
        c="r",
        label="Waypoints",
    )
    ax1.legend()

    plt.show()
