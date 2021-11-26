import json, pyproj
from forge_pathplanner import bdc, prm, surface, polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
                    "lat": latloc,
                    "lon": longloc,
                    "radius": radius,
                    "height": height,
                }
            )
        return obstacles


if __name__ == "__main__":
    with open("example_mission.json", "r") as f:
        mission_json = json.load(f)

    wgs2loc = pyproj.Transformer.from_crs(CRS["WGS84"], CRS["Irvine"])
    loc2wgs = pyproj.Transformer.from_crs(CRS["Irvine"], CRS["WGS84"])

    mission = Mission(mission_json, wgs2loc, loc2wgs)
    bds = mission.l_get_boundaries()[0]
    obs = mission.l_get_obstacles()
    sg = mission.l_get_search_boundaries()
    wp = mission.l_get_ordered_waypoints()

    fg1 = plt.figure(figsize=(9, 9))
    ax1 = fg1.add_subplot(1, 1, 1)
    fg2 = plt.figure(figsize=(9, 9))
    ax2 = fg2.add_subplot(1, 1, 1, projection="3d")

    for o in obs:
        circ = Circle((o["lat"], o["lon"]), o["radius"])
        ax1.add_artist(circ)

    ax1.plot(bds[:, 0], bds[:, 1], lw=2, c="k", label="Outer Border")
    ax1.plot(sg[:, 0], sg[:, 1], lw=1, ls="--", c="g", label="Search Grid")
    ax1.scatter(wp[:, 0], wp[:, 1], marker="*", s=50, c="r", label="Waypoints")
    ax2.scatter(
        wp[:, 0], wp[:, 1], wp[:, 2], marker="*", s=50, c="r", label="Waypoints"
    )

    outer_buffer = 250

    xmin, xmax = bds[:, 0].min() - outer_buffer, bds[:, 0].max() + outer_buffer
    ymin, ymax = bds[:, 1].min() - outer_buffer, bds[:, 1].max() + outer_buffer
    step = min(xmax - xmin, ymax - ymin) / 80
    xrange = xmin, xmax
    yrange = ymin, ymax

    X, Y = surface.generate_xy_grid(xrange, yrange, step)
    print("xyshape", X.shape, Y.shape)
    print("yrange", yrange, "\txrange", xrange)

    obs_list = []
    for o in obs:
        obs_list.append(((o["lat"], o["lon"]), o["radius"], o["height"]))

    Hground = surface.place_obstacles(X, Y, obs_list)

    ax2.set_box_aspect([xmax - xmin, ymax - ymin, Hground.max() - Hground.min()])

    print(Hground.max(), Hground.min())

    Hsheet = surface.get_optimal_grid(
        X, Y, Hground, 50, 1, 0.1, 100, step, waypoints=wp, solver=None
    )
    surface.plot_mpl3d(ax2, X, Y, Hground, Hsheet, wireframe=True)
    surface.plot_mpl_2d(ax1, X, Y, Hsheet, levels=40)

    ax1.legend()

    plt.show()
