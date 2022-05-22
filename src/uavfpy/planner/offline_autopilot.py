# Omar Hossain
# offline_autopilot.py


try:
    from uavfpy.planner.mission import Mission, get_xformer_from_CRS_str
except:
    from mission import Mission, get_xformer_from_CRS_str

class OfflinePlanner:

    # Mission Object
    mission = None


    def __init__(self, fileName, interopName):
        self.f = open(fileName + ".waypoints", "x")
        with open(interopName) as f:
            mission_json = f.read()
        wgs2loc = get_xformer_from_CRS_str("WGS84", "Irvine")
        loc2wgs = get_xformer_from_CRS_str("Irvine", "WGS84")
        self.mission = Mission(
            mission_json, 
            wgs2loc, 
            loc2wgs, 
            grid_buffer=(2.0, 2.0),
            grid_max_npoints=(200, 200),
            grid_stepxy=(5.0, 5.0),
        )
        self.mission.solve_Hsurf(
            buffer=10.0, max_dh=8.0, max_d2h=0.06, solve_shape=(60, 60), verbose=False
        )
        self.mission.compute_plan_thru_waypoints
        waypointpath = self.mission.compute_plan_thru_waypoints(self.mission.waypoints, n=400)
        print(waypointpath)
        self.set_beginning()

    def __init__(self, fileName):
        self.curIndex = 0
        self.f = open(fileName + ".waypoints", "x")
        self.set_beginning()

    def set_beginning(self):
        self.f.write("QGC WPL 110\n")
        self.f.write(self.generate_waypoint(self.curIndex, 1, 0, "BEG", 0, 0, 0, 0, 0, 0, 0, 1))

    # curWp should be 0 if following series of waypoints
    # cordFrame relates to 0 = absolute 3 = relative coordinate frame
    def generate_waypoint(self, index, curWp, cordFrame, command, param1, param2, param3, param4, param5, param6, param7, autoContinue):
        if command == "WAYPOINT":
            command = 16
        elif command == "TAKEOFF":
            command = 22
        elif command == "LAND":
            command = 21
        elif command == "BEG":
            command = 0
        else: # RETURN, default in case error
            command = 20

        wp = ""
        for i in (index, curWp, cordFrame, command, param1, param2, param3, param4, param5, param6, param7, autoContinue):
            wp = wp + str(i) + '\t'
        wp = wp[:-1] + '\n'
        return wp

    def addWaypoint(self, cmd, latt, long, altitude):
        self.curIndex+=1
        if(cmd == "TAKEOFF"):
            wp = self.generate_waypoint(self.curIndex, 0, 3, cmd, 0, 0, 0, 0, 0, 0, altitude, 1)
        else:
            wp = self.generate_waypoint(self.curIndex, 0, 3, cmd, 0, 0, 0, 0, latt, long, altitude, 1)
        self.f.write(wp)


if __name__ == "__main__":
    fileName = input("Input fileName: ")

    op = OfflinePlanner(fileName)
    op.addWaypoint("TAKEOFF", 0, 0, 100)
    op.addWaypoint("WAYPOINT", 33.64272700, -117.82523900, 100)
    op.addWaypoint("WAYPOINT", 33.64255600, -117.82468910, 100)
    op.addWaypoint("WAYPOINT", 33.64219870, -117.82518000, 100)
    op.addWaypoint("RETURN", 0, 0, 100)
