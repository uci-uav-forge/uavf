import json, pyproj
import surface
import numpy as np
from typing import Tuple
from shapely.geometry import Polygon, Point

# From https://kespry.force.com/s/article/Finding-Your-State-Plane-Coordinate-System
CRS = {
    "Irvine": pyproj.CRS.from_epsg(2230),
    "Langley": pyproj.CRS.from_epsg(2283),
    "WGS84": pyproj.CRS.from_epsg(4326),
}

FT2METER = 3.28084
METER2FT = 0.3048


def get_xformer_from_CRS_str(from_str: str, to_str: str):
    from_crs = CRS[from_str]
    to_crs = CRS[to_str]
    return pyproj.Transformer.from_crs(from_crs, to_crs)


class Mission(object):
    def __init__(
        self,
        mission_json: str,
        wgs2loc: pyproj.Transformer,
        loc2wgs: pyproj.Transformer,
    ):
        self.mission_json = json.loads(mission_json)
        self.wgs2loc = wgs2loc
        self.loc2wgs = loc2wgs

        self.boundaries, self.alt_min, self.alt_max = self.l_get_boundaries()
        # only the first flight zone is used
        self.boundaries, self.alt_min, self.alt_max = (
            self.boundaries[0],
            self.alt_min[0],
            self.alt_max[0],
        )
        self.search_boundaries = self.l_get_search_boundaries()
        self.waypoints = self.l_get_ordered_waypoints()
        self.obstacles = self.l_get_obstacles()

    def cover_grid(
        self,
        poly: np.ndarray,
        buffer: Tuple[float, float],
        max_npoints: Tuple[int, int],
        stepxy: Tuple[float, float],
    ):
        """
        Return a grid of points covering a polygon
        """
        xmin, xmax = poly[:, 0].min() - buffer[0], poly[:, 0].max() + buffer[0]
        ymin, ymax = poly[:, 1].min() - buffer[1], poly[:, 1].max() + buffer[1]

        x_range = xmax - xmin
        y_range = ymax - ymin
        stepx, stepy = max(stepxy[0], x_range / max_npoints[0]), max(
            stepxy[1], y_range / max_npoints[1]
        )
        X, Y = np.meshgrid(np.arange(xmin, xmax, stepx), np.arange(ymin, ymax, stepy))
        gridstep = (stepx, stepy)
        return X, Y, gridstep

    def l_get_boundaries(self):
        bounds, min_heights, max_heights = [], [], []
        for fz in self.mission_json["flyZones"]:
            fzbounds = []
            for bp in fz["boundaryPoints"]:
                # get latitude, longitude in WGS84
                lat84 = bp["latitude"]
                long84 = bp["longitude"]
                # Convert to local
                latloc, longloc = wgs2loc.transform(lat84, long84)
                fzbounds.append([latloc, longloc])
            # close the loop
            fzbounds.append(fzbounds[0])
            bounds.append(np.array(fzbounds))
            max_heights.append(fz["altitudeMin"])
            min_heights.append(fz["altitudeMax"])
        return bounds, max_heights, min_heights

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


if __name__ == "__main__":
    # make mission
    with open("example_mission.json") as f:
        mission_json = f.read()

    # create transformers
    wgs2loc = get_xformer_from_CRS_str("WGS84", "Irvine")
    loc2wgs = get_xformer_from_CRS_str("Irvine", "WGS84")

    # create mission
    mission = Mission(mission_json, wgs2loc, loc2wgs)

    # get a grid covering the region
    X, Y, gridstep = mission.cover_grid(
        mission.boundaries, (10.0, 10.0), (200, 200), (40.0, 40.0)
    )
    # get terrain grid
    Hterrain = surface.place_obstacles(X, Y, mission.obstacles)

    # get planar infinite height obstacles
    O = np.empty_like(X)
    # make a polygon from the mission boundaries
    boundary_polygon = Polygon(shell=mission.boundaries)
    # create a closed world in O from the polygon
    for xy in np.ndindex(X.shape):
        xy_point = np.array([X[xy], Y[xy]])
        if not boundary_polygon.contains(Point(xy_point)):
            O[xy] = 1
        elif Hterrain[xy] >= mission.alt_max:
            O[xy] = 1
        else:
            O[xy] = 0

    # remove infinite height obstacles from the surface; we can't go over them!
    Hterrain = np.where(Hterrain >= mission.alt_max, 0, Hterrain)

    # get surface
    S = surface.get_optimal_grid(
        X,
        Y,
        Hterrain,
        buffer=10.0,
        max_dh=1.0,
        max_d2h=0.1,
        min_h=10.0,
        step=gridstep,
        waypoints=mission.waypoints,
    )

    # unpack waypoints

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # draw obstacles on the grid
    ax.pcolor(X, Y, O, cmap="binary", shading="auto")
    # draw surface on the grid as contours
    ax.contour(X, Y, S, levels=16, cmap="ocean", linewidths=0.5, antialiased=True)
    # plot the mission boundaries
    ax.plot(mission.boundaries[:, 0], mission.boundaries[:, 1], "k-")
    # plot the waypoints
    ax.plot(mission.waypoints[:, 0], mission.waypoints[:, 1], "r--")
    ax.autoscale()

    plt.show()
