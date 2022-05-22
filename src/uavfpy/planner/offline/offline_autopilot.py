# Omar Hossain
# offline_autopilot.py


class OfflinePlanner:

    def __init__(self, f):
        self.curIndex = 0
        self.f = f
        self.f.write("QGC WPL 110\n");
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

    def addWaypoint(self, f, cmd, latt, long, altitude):
        self.curIndex+=1
        if(cmd == "TAKEOFF"):
            wp = self.generate_waypoint(self.curIndex, 0, 3, cmd, 0, 0, 0, 0, 0, 0, altitude, 1)
        else:
            wp = self.generate_waypoint(self.curIndex, 0, 3, cmd, 0, 0, 0, 0, latt, long, altitude, 1)
        f.write(wp)



if __name__ == "__main__":
    fileName = input("Input fileName: ")
    with open(fileName + ".waypoints", "x") as f:
        op = OfflinePlanner(f)
        op.addWaypoint(f, "TAKEOFF", 0, 0, 100)
        op.addWaypoint(f, "WAYPOINT", 33.64272700, -117.82523900, 100)
        op.addWaypoint(f, "WAYPOINT", 33.64255600, -117.82468910, 100)
        op.addWaypoint(f, "WAYPOINT", 33.64219870, -117.82518000, 100)
        op.addWaypoint(f, "RETURN", 0, 0, 100)
