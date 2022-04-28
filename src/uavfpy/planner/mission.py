import json, pyproj, time, cv2, rrtplanner, logging, sys
import numpy as np
from typing import Tuple
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import ceil

# relative imports
from . import surface


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# From https://kespry.force.com/s/article/Finding-Your-State-Plane-Coordinate-System
CRS = {
    "Irvine": pyproj.CRS.from_epsg(2230),
    "Maryland": pyproj.CRS.from_epsg(2283),
    "WGS84": pyproj.CRS.from_epsg(4326),
}

FT2METER = 3.28084
METER2FT = 0.3048


def get_xformer_from_CRS_str(from_str: str, to_str: str):
    from_crs = CRS[from_str]
    to_crs = CRS[to_str]
    return pyproj.Transformer.from_crs(from_crs, to_crs)


class Mission(object):
    """
    Object representing a mission.

    Parameters
    ----------
    mission_json : str
        Json string containing the mission. You can obtain this from the interop
        server.
    wgs2loc : pyproj.Transformer
        transform projection from WGS84 to local coordinate system
    loc2wgs : pyproj.Transformer
        transform projection from local coordinate system to WGS84
    grid_buffer : Tuple[float, float]
        the buffer to add around the outside boundaries of the misssion when
        creating the occupancygrid, in feet
    grid_max_npoints : Tuple[int, int]
        maximum number of points in the x,y direction to use when creating the
        occupancygrid. Set this to be the largest grid you want to process/plan
        over.
    grid_stepxy : Tuple[float, float]
        desired step size in the x,y direction when creating the occupancygrid. This
        step size is used only when the resulting grid size is under
        `grid_max_npoints` in the x or y direction.
    """

    def __init__(
        self,
        mission_json: str,
        wgs2loc: pyproj.Transformer,
        loc2wgs: pyproj.Transformer,
        grid_buffer: Tuple[float, float],
        grid_max_npoints: Tuple[int, int],
        grid_stepxy: Tuple[float, float],
    ):
        self.mission_json = json.loads(mission_json)
        # projection transforms
        self.wgs2loc = wgs2loc
        self.loc2wgs = loc2wgs
        # unpack mission json into list of boundaries
        self.boundaries, self.alt_min, self.alt_max = self.json_get_boundaries()
        # only the first of the boundaries is used.
        self.boundaries = self.boundaries[0]
        self.alt_min = self.alt_min[0]
        self.alt_max = self.alt_max[0]

        # unpack search zone boundaries, waypoints, and obstacles
        self.search_boundaries = self.json_get_search_boundaries()
        self.waypoints = self.json_get_ordered_waypoints()
        self.obstacles = self.json_get_obstacles()

        # get xy grid
        self.X, self.Y, self.gridstep = self.cover_grid(
            self.boundaries,
            grid_buffer,
            grid_max_npoints,
            grid_stepxy,
        )
        # get "terrain" on grid
        self.Hterrain = surface.place_obstacles(self.X, self.Y, self.obstacles)
        self.og = self.get_occupancygrid(
            self.boundaries,
            self.Hterrain,
            self.alt_max,
        )
        # now that "infinite height" obstacles have been accounted for in the occupancy
        # grid, we can remove them so that the sheet planner does not consider their
        # altitude when computing the sheet altitude.
        self.Hterrain = np.where(self.Hterrain >= self.alt_max, 0, self.Hterrain)

        # solving the surface may take a long time so we don't do it on object
        # instantiation
        self.Hsurf = None

    def get_occupancygrid(
        self,
        boundaries: np.ndarray,
        Hterrain: np.ndarray,
        alt_max: float,
    ) -> np.ndarray:
        """
        Get the occupancy grid for the mission. The occupancy grid is a grid where 0 is
        the allowable space and 1 is the forbidden space. The vehicle cannot fly outside
        of the mission bounds, or at any x,y location where the obstacle is higher than
        the maximum altitude. However, x,y positions where the obstacle is shorter than
        the maximum altitude can be flown by the vehicle. The surface planner will route
        the vehicle over those obstacles -- therefore, their x,y position is marked
        valid.

        Parameters
        ----------
        boundaries : np.ndarray
            ordered boundary points
        Hterrain : np.ndarray
            Height of the "terrain." This is the array containing a static obstacle
            height for each space in x,y. z=0 where there is no static obstacle,
            z=obs_height where there is a static obstacle.
        alt_max : float
            max altitude for this region

        Returns
        -------
        np.ndarray
            MxN array of 0s and 1s, where 1 is the forbidden space and 0 is the allowed
            space. So, for an i,j grid point corresponding to some X[ij], Y[ij] pair, we
            can query the occupancygrid returned by this function to determine if the
            vehicle is allowed there.
        """
        logging.info("Placing Obstacles...")
        # occupancy grid is 2d bool array. 1 where obstacle is present 0 where obstacle is absent
        og = np.empty_like(self.X)
        # everything outside of mission bounds is a static obstacle
        boundary_polygon = Polygon(shell=boundaries)
        # check each point
        for xy in tqdm(
            np.ndindex(self.X.shape), desc="Placing Obstacles", total=self.X.size
        ):
            xy_point = np.array([self.X[xy], self.Y[xy]])
            if not boundary_polygon.contains(Point(xy_point)):
                og[xy] = 1
            elif Hterrain[xy] >= alt_max:
                og[xy] = 1
            else:
                og[xy] = 0
        logging.info("Done!")
        return og

    def cover_grid(
        self,
        poly: np.ndarray,
        buffer: Tuple[float, float],
        max_npoints: Tuple[int, int],
        stepxy: Tuple[float, float],
    ):
        """
        get a grid of points which cover a polygon. Polygon is an ordered list of
        boundary points in the local coordinate system. buffer is the buffer around the
        polygon. max_npoints is the maximum points in both x and y direcitons. stepxy is
        the desired size of a grid cell in the x and y directions. this method will
        return a grid covering the polygon (plus the buffer) with either the desired
        step size or the maximum number of points, whichever is smaller.

        For example: if my polygon covers a region of 90m x 90m, and my grid buffer is
        (5m, 5m), the grid will be (100m x 100m). If I then specify a max number of
        points of (100 x 100), the grid (with the max number of points!) is a (100x100)
        grid with each cell being 1m x 1m. If I specify a step size of (5m x 5m),
        though, I will get a grid that is (20 x 20).

        It is recommended to create a grid with sufficient resolution to produce
        accurate paths (i.e., a grid with a cell size of 100m x 100m is useless if I
        want to plan paths to within 3m x 3m!). But not with resolution that is so large
        that computing paths will take a crazy amount of time to plan.

        Parameters
        ----------
        poly : np.ndarray
            Polygon to cover, as (M x 2) ndarray in local coordinate system.
        buffer : Tuple[float, float]
            Buffer in x and y direction.
        max_npoints : Tuple[int, int]
            Max number of grid points to produce in (x,y) direction.
        stepxy : Tuple[float, float]
            Desired cell dimensions in x and y direction.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, tuple]
            X coordinates of the grid, Y coordinates of the grid, and a single cell's size.
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

    def get_nearest_grid_idx(self, x, y):
        """
        Get index of the nearest grid point in (X, Y) to a point (x, y)

        Parameters
        ----------
        x : float
            point x
        y : float
            point y
        X : np.ndarray
            array of grid points X
        Y : np.ndarray
            array of grid points Y
        """
        ix = np.argmin(np.abs(self.X[0, :] - x))
        iy = np.argmin(np.abs(self.Y[:, 0] - y))
        return np.array([ix, iy], dtype=np.intc)

    def json_get_boundaries(self) -> Tuple[list, list, list]:
        """
        get mission boundaries. missions can, according to the schema, have multiple
        areas. Each of these areas would have a list of boundaries and a max/min height.

        Returns
        -------
        Tuple[list, list, list]
            mission boundaries. The first is a list-of-lists containing boundaries in
            local coordinates, the second is a list of max altitudes, and the third is a
            list of min altitudes.
        """
        bounds, min_heights, max_heights = [], [], []
        for fz in self.mission_json["flyZones"]:
            fzbounds = []
            for bp in fz["boundaryPoints"]:
                # get latitude, longitude in WGS84
                lat84 = bp["latitude"]
                long84 = bp["longitude"]
                # Convert to local
                latloc, longloc = self.wgs2loc.transform(lat84, long84)
                fzbounds.append([latloc, longloc])
            # close the loop
            fzbounds.append(fzbounds[0])
            bounds.append(np.array(fzbounds))
            max_heights.append(fz["altitudeMin"])
            min_heights.append(fz["altitudeMax"])
        return bounds, max_heights, min_heights

    def json_get_search_boundaries(self) -> np.ndarray:
        """
        get search boundaries in local coordinates as ordered ndarray

        Returns
        -------
        np.ndarray
            search boundaries
        """
        bounds = []
        for sgp in self.mission_json["searchGridPoints"]:
            # get lat long in WGS84
            lat84 = sgp["latitude"]
            long84 = sgp["longitude"]
            # convert to local
            latloc, longloc = self.wgs2loc.transform(lat84, long84)
            bounds.append([latloc, longloc])
        # closed loop
        bounds.append(bounds[0])
        return np.array(bounds)

    def json_get_ordered_waypoints(self) -> np.ndarray:
        """
        get the ordered waypoints in local coordinate frame as ndarray

        Returns
        -------
        np.ndarray
            ordered waypoints in local coordinate system
        """
        waypoints = []
        for wp in self.mission_json["waypoints"]:
            lat84, long84, h = wp["latitude"], wp["longitude"], wp["altitude"]
            latloc, longloc = self.wgs2loc.transform(lat84, long84)
            waypoints.append([latloc, longloc, h])
        return np.array(waypoints)

    def json_get_obstacles(self) -> list:
        """
        get obstacles from the mission json

        Returns
        -------
        list
            list of obstacles as list of dict with keys x, y, r, h
        """
        obstacles = []
        for obs in self.mission_json["stationaryObstacles"]:
            lat84 = obs["latitude"]
            long84 = obs["longitude"]
            radius = obs["radius"]
            height = obs["height"]
            latloc, longloc = self.wgs2loc.transform(lat84, long84)
            obstacles.append(
                {
                    "x": latloc,
                    "y": longloc,
                    "r": radius,
                    "h": height,
                }
            )
        return obstacles

    def get_Hsurf(self) -> np.ndarray:
        """Get the optimal surface height for each grid point

        Returns
        -------
        np.ndarray
            The surface height for each grid point

        Raises
        ------
        ValueError
            If solve_Hsurf() has not been called
        """
        if self.Hsurf is not None:
            return self.Hsurf
        else:
            hsurf_not_found_str = "You did not call solve_Hsurf on this object yet, and therefore, there is no surface object stored! Call solve_Hsurf() before attempting to access the surface object."
            raise ValueError(hsurf_not_found_str)

    def solve_Hsurf(
        self,
        buffer: float,
        max_dh: float,
        max_d2h: float,
        solve_shape: tuple = (100, 100),
        verbose: bool = True,
    ):
        """
        Solve the surface given mission parameters. This method may take some time to
        run, and may fail to run if a convergent solution cannot be found. The method
        downsamples the terrain object to size `solve_shape`, solves a convex
        optimization problem over that downscaled terrain to produce the optimal
        surface, then upscales the surface to the original terrain (and hence, occupancy
        grid) size. small values of `solve_shape` will result in faster convergence, but
        may produce quantization artifacts.

        buffer, max_dh, and max_d2h are parameters of the optimization problem; in
        general, small values of max_dh and max_d2h will cause the surface to be
        "smoother" but may also result in an infeasible problem. In particular, if
        waypoints are at different altitudes, but placed very close to one another,
        max_dh and max_d2h must be set large enough to produce a feasible path between
        the two waypoints.

        It is recommended that you experiment with these parameters for your particular
        mission to avoid solver errors.

        This method calls the method `surface.get_optimal_grid()`.

        Parameters
        ----------
        buffer : float
            vertical safety buffer between the solved sheet and the surface of objects.
        max_dh : float
            maximum dh/dxy allowed for the vehicle. A dh/dxy of 1 means that the vehicle
            can climb or descend at most 1 meter for every meter of horizontal distance
            travelled.
        max_d2h : float
            maximum d2h/dxy^2 allowed for the vehicle. Setting this smaller means that
            the vehicle can pitch up or down slower.
        solve_shape : tuple, optional
            shape of the grid over which the optimization problem is solved. Larger
            values for this grid will be more accurate, but will take longer and may
            introduce solver errors, by default (100, 100)
        verbose : bool, optional
            whether to print verbose solver information, by default True
        """

        logging.info("Solving Surface...")
        # scale down the array to get desired solve performance. Solver performance is
        # dictated by the size of solve_shape. Smaller solve_shape takes less time to
        # solve, but may suffer from quantization artifacts. Large solve_shape is more
        # accurate, but can fail to converge for some obstacles depending on the
        # rigidity of constraints.
        Xsc = cv2.resize(self.X, solve_shape, interpolation=cv2.INTER_AREA)
        Ysc = cv2.resize(self.Y, solve_shape, interpolation=cv2.INTER_AREA)
        Hterrainsc = cv2.resize(
            self.Hterrain, solve_shape, interpolation=cv2.INTER_AREA
        )

        # scale factors
        xsc, ysc = solve_shape[0] / self.X.shape[0], solve_shape[1] / self.Y.shape[1]

        # scale down the grid step
        step_sc = (self.gridstep[0] / xsc, self.gridstep[1] / ysc)
        print(self.gridstep, step_sc)

        # then solve the convex problem
        Hsurfsc = surface.get_optimal_grid(
            Xsc,
            Ysc,
            Hterrainsc,
            buffer=buffer,
            max_dh=max_dh,
            max_d2h=max_d2h,
            min_h=self.alt_min,
            step=step_sc,
            waypoints=self.waypoints,
            verbose=verbose,
        )
        # scale the surface back up using cubic interpolation
        Hsurf = cv2.resize(
            Hsurfsc, self.Hterrain.T.shape, interpolation=cv2.INTER_CUBIC
        )
        self.Hsurf = Hsurf

        logging.info("Done!")

    def compute_plan_thru_waypoints(
        self, waypoints: np.ndarray, n: int = 2500, r_rewire: float = -1.0, plot=False
    ):
        if self.Hsurf is None:
            raise ValueError(
                "You must call solve_Hsurf() before attempting to compute a plan."
            )
        if not (waypoints.shape[1] != 2 or waypoints.shape[1] != 3):
            raise ValueError(
                "Waypoints must be a 2D array with shape (n,2) or (n,3). Shape is {}".format(
                    waypoints.shape
                )
            )
        if waypoints.shape[0] < 2:
            raise ValueError("Waypoints must have at least 2 points.")

        # set r_rewire to default value
        if r_rewire < 0.0:
            r_rewire = ceil(400.0 / self.gridstep[0])
        else:
            r_rewire = ceil(r_rewire / self.gridstep[0])

        # set r_goal
        r_goal = 8

        rrts = rrtplanner.RRTStarInformed(
            self.og.T, n=n, r_rewire=r_rewire, r_goal=r_goal, pbar=False
        )
        if plot:
            ptwp_fig = plt.figure()
            ax = ptwp_fig.add_subplot(111)
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Computed Plan Through {} Waypoints".format(len(waypoints)))
        # paths are computed relative to the occupancy grid.
        og_paths = []

        logging.info("Using {} with r_rewire={} to plan.".format(type(rrts), r_rewire))
        logging.info("Computing waypoint paths...")
        t0 = time.time()

        # iterate through pairs of waypoints
        for i in tqdm(range(waypoints.shape[0] - 1)):
            # unpack waypoint pairs
            w1 = waypoints[i]
            w2 = waypoints[i + 1]
            # plans are computed by occupancygrid index, not by absolute location. so
            # xstart, xgoal correspond to cells of the occupancy grid.
            xstart = self.get_nearest_grid_idx(w1[0], w1[1])
            xgoal = self.get_nearest_grid_idx(w2[0], w2[1])
            # a tree and a goal vertex are produced for each waypoint pair
            T, gv = rrts.plan(xstart, xgoal)
            # get path from node 0 to the goal vertex for each waypoint
            vertices = rrts.route2gv(T, gv)
            # get path points. The path is an ordered list of occupancy grid indices.
            path = rrts.vertices_as_ndarray(T, vertices)
            # append to og paths
            og_paths.append(path)

            # draw plot
            if plot:
                rrtplanner.plot_rrt_lines(
                    ax, T, color_costs=False, color="silver", alpha=0.5
                )

        # now we combine each path from waypoint to waypoint into a single array
        all_paths = np.concatenate(og_paths, axis=0)

        # remove duplicates
        for i in range(all_paths.shape[0] - 1):
            if np.all(all_paths[i] == all_paths[i + 1]):
                all_paths = np.delete(all_paths, i + 1, axis=0)

        # logging
        t1 = time.time()
        logging.info(
            "Done! Total waypoint path length: {}, took {}s".format(
                all_paths.shape[0], t1 - t0
            )
        )

        # plotting
        if plot:
            rrtplanner.plot_path(ax, all_paths)
            ax.imshow(self.og, cmap="binary")
            plt.show()

        world_paths = []
        # from occupancyGrid to world coordinates
        for ogp in all_paths:
            og0 = ogp[0]
            og1 = ogp[1]
            x0 = self.X[og0[1], og0[0]]
            x1 = self.X[og1[1], og1[0]]
            y0 = self.Y[og0[1], og0[0]]
            y1 = self.Y[og1[1], og1[0]]
            world_paths.append((x0, y0))
            world_paths.append((x1, y1))
        return np.array(world_paths)

    def transform_to_wgs84(self, array: np.ndarray) -> np.ndarray:
        """
        Method to transform an (Mx2) array to WGS84 coordinates. (M x 3) arrays can be passed in but only the first two columns are touched.

        Parameters
        ----------
        array : np.ndarray
            (M x 2+) array of M points in local coordinate system

        Returns
        -------
        np.ndarray
            (M x 2) array of transformed points in WGS84 coordinates
        """
        output = []
        for xx, yy in zip(array[:, 0], array[:, 1]):
            xxt, yyt = self.loc2wgs.transform(xx, yy)
            output.append((xxt, yyt))
        return np.array(output)

    def transform_from_wgs84(self, array: np.ndarray) -> np.ndarray:
        """Method to transform an (Mx2) array from WGS84 coordinates to local coordinates. (M x 3) arrays can be passed in but only the first two columns are touched.

        Parameters
        ----------
        array : np.ndarray
            (M x 2+) array of M points in WGS84 coordinates

        Returns
        -------
        np.ndarray
            (M x 2) array of M points in local coordinates
        """
        output = []
        for xx, yy in zip(array[:, 0], array[:, 1]):
            xxt, yyt = self.wgs2loc.transform(xx, yy)
            output.append((xxt, yyt))
        return np.array(output)


if __name__ == "__main__":
    # make mission
    with open("example_mission.json") as f:
        mission_json = f.read()

    # create transformers
    wgs2loc = get_xformer_from_CRS_str("WGS84", "Irvine")
    loc2wgs = get_xformer_from_CRS_str("Irvine", "WGS84")

    # create mission
    mission = Mission(
        mission_json,
        wgs2loc,
        loc2wgs,
        grid_buffer=(2.0, 2.0),
        grid_max_npoints=(200, 200),
        grid_stepxy=(5.0, 5.0),
    )
    # solve the surface
    mission.solve_Hsurf(
        buffer=10.0, max_dh=8.0, max_d2h=0.06, solve_shape=(60, 60), verbose=False
    )

    # get the solved params
    waypointpath = mission.compute_plan_thru_waypoints(mission.waypoints, n=400)

    X, Y, O = mission.X, mission.Y, mission.og
    S = mission.get_Hsurf()

    from plots import plot_surface_2d, plot_surface_3d
    from matplotlib import pyplot as plt

    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111, projection="3d")
    # plot the mission boundaries
    ax1.plot(mission.boundaries[:, 0], mission.boundaries[:, 1], "k-")
    # plot the waypoints
    ax1.plot(mission.waypoints[:, 0], mission.waypoints[:, 1], "r--")
    plot_surface_2d(ax1, X, Y, S, levels=40)
    ax1.pcolor(X, Y, O, cmap="binary", shading="auto")
    plot_surface_3d(ax2, X, Y, mission.Hterrain, S)

    # plot waypoint path
    ax1.plot(waypointpath[:, 0], waypointpath[:, 1], "b-")
    plt.show()
