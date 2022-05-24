"""
Offline Autopilot (Manual Waypoint Generation Script)

Author: Omar Hossain

NOTE: Altitude in Mission Planner is in METERS
"""

import logging, sys

try:
    from uavfpy.planner import mission
except ModuleNotFoundError:
    logging.warning(
        "uavfpy package is available to PYTHONPATH. Maybe it is not installed? In this case, we will assume that you are running this script directly from the repository."
    )
    logging.info("Attempting import from relative path...")
    # skip up one
    sys.path.append("../")
    try:
        # import directly
        from src.uavfpy.planner import mission
    except ImportError:
        logging.exception(
            "Failed to import uavfpy! Check that uavfpy is installed or that you are running this script from the `scripts` directory."
        )


class OfflinePlanner(object):
    def __init__(
        self,
        input_fname: str,
        output_fname: str,
        gridcell: float,
        maxgrid: tuple,
        xybuffer: float,
        local_CRS: str,
        n_rrt: int,
    ):
        self.curIndex = 0

        self.input_fname = input_fname
        self.output_fname = output_fname

        # read out mission JSON
        with open(input_fname) as f:
            mission_json = f.read()

        # get transformers
        wgs2loc = mission.get_xformer_from_CRS_str("WGS84", local_CRS)
        loc2wgs = mission.get_xformer_from_CRS_str(local_CRS, "WGS84")

        # create mission, solve, and plan
        self.mission = mission.Mission(
            mission_json,
            wgs2loc,
            loc2wgs,
            # amount of "buffer" over the xy boundaries of the  mission region that the occupancygrid covers.
            grid_buffer=(xybuffer, xybuffer),
            # maximum grid shape. Very large grids can result in long planning times; this prevents a tiny step size from producing a very large grid
            grid_max_npoints=maxgrid,
            # grid cell size
            grid_stepxy=(gridcell, gridcell),
        )

        # solve the surface
        self.mission.solve_Hsurf(
            buffer=10.0, max_dh=8.0, max_d2h=0.06, solve_shape=(80, 80), verbose=True
        )

        # compute a plan through the waypoints
        self.mission.compute_plan_thru_waypoints(
            self.mission.waypoints,
            n=n_rrt,
            plot=True,
        )
        print("Waypoints Before Transformation: ")
        print(self.mission.waypoints)
        print("Waypoints After Transformation: ")
        print(self.mission.transform_to_wgs84(self.mission.waypoints))
        print("# of Distinct Waypoints: ")
        print(len(self.mission.waypoints))
        self.set_beginning()
        self.generate_mission()

    def generate_mission(self):
        waypoints = self.mission.transform_to_wgs84(self.mission.waypoints)
        self.addWaypoint("TAKEOFF", 0, 0, 6)
        for i in range(len(self.mission.waypoints)):
            self.addWaypoint(
                "WAYPOINT",
                waypoints[i][0],
                waypoints[i][1],
                self.mission.waypoints[i][2],
            )
        self.addWaypoint("RETURN", 0, 0, 6)

    def set_beginning(self):
        with open("offlineMissions/"+self.output_fname, "a") as f:
            f.write("QGC WPL 110\n")
            f.write(
                self.generate_waypoint(
                    self.curIndex, 1, 0, "BEG", 0, 0, 0, 0, 0, 0, 0, 1
                )
            )

    # curWp should be 0 if following series of waypoints
    # cordFrame relates to 0 = absolute 3 = relative coordinate frame
    def generate_waypoint(
        self,
        index,
        curWp,
        cordFrame,
        command,
        param1,
        param2,
        param3,
        param4,
        param5,
        param6,
        param7,
        autoContinue,
    ):
        if command == "WAYPOINT":
            command = 16
        elif command == "TAKEOFF":
            command = 22
        elif command == "LAND":
            command = 21
        elif command == "BEG":
            command = 0
        else:  # RETURN, default in case error
            command = 20

        wp = ""
        for i in (
            index,
            curWp,
            cordFrame,
            command,
            param1,
            param2,
            param3,
            param4,
            param5,
            param6,
            param7,
            autoContinue,
        ):
            wp = wp + str(i) + "\t"
        wp = wp[:-1] + "\n"
        return wp

    def addWaypoint(self, cmd, latt, long, altitude):
        self.curIndex += 1
        if cmd == "TAKEOFF":
            wp = self.generate_waypoint(
                self.curIndex, 0, 3, cmd, 0, 0, 0, 0, 0, 0, altitude, 1
            )
        else:
            wp = self.generate_waypoint(
                self.curIndex, 0, 3, cmd, 0, 0, 0, 0, latt, long, altitude, 1
            )
        with open(self.output_fname, "a") as f:
            f.write(wp)


from argparse import ArgumentParser

parser = ArgumentParser()

# required: input and output filenames
parser.add_argument(
    dest="interop_json",
    help="Interop Mission Definition (JSON)",
    type=str,
)
parser.add_argument(
    dest="output",
    help="Output Waypoint File for use by qGroundControl (.waypoint)",
    type=str,
)

# optional: grid, solve params
parser.add_argument(
    "-gridcell",
    "--gridcell",
    dest="gridcell",
    help="Planner occupancygrid cell size, in feet",
    default=8.0,
    type=float,
)
parser.add_argument(
    "-maxgrid",
    "--maxgrid",
    dest="maxgrid",
    help="Maximum occupancygrid shape. integer",
    default=(400, 400),
    type=int,
)
parser.add_argument(
    "-xybuffer",
    "--xybuffer",
    dest="xybuffer",
    help="Buffer in excess of mission boundary, in feet",
    default=10.0,
    type=float,
)
parser.add_argument(
    "-localCRS",
    "--localCRS",
    dest="localCRS",
    help='Local Coordinate Reference System. Allowable options are "Irvine" for the test locations or "Maryland" for the competition location.',
    default="Irvine",
)
parser.add_argument(
    "-n",
    "--n_rrt",
    dest="n_rrt",
    help="Number of RRT sample points",
    default=1000,
)
opts = parser.parse_args()

op = OfflinePlanner(
    opts.interop_json,
    opts.output,
    opts.gridcell,
    opts.maxgrid,
    opts.xybuffer,
    opts.localCRS,
    opts.n_rrt,
)
