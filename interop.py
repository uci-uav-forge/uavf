# /usr/bin/python3

from interop_client import InteropClient
import rospy, threading
from geographic_msgs.msg import GeoPointStamped, GeoPoint
from interop.msg import (
    Color,
    FlyZone,
    FlyZoneArray,
    Orientation,
    Shape,
    Object,
    ObjectType,
    GeoCylinder,
    GeoPolygonStamped,
    GeoCylinderArrayStamped,
    WayPoints,
)

# AUVSI SERVER INFO
SERVER_IP = "127.0.0.1"
SERVER_PORT = "8000"
SERVER_URL = f"http://{SERVER_IP}:{SERVER_PORT}"
USERNAME = "testadmin"
PASSWORD = "testpass"

# TOPICS
WAYPOINTS_TOPIC = "waypoints"
FLYZONES_TOPIC = "flyzones"
SEARCH_GRID_TOPIC = "search_grid_points"
AIR_DROP_BOUNDARY_TOPIC = "air_drop_boundary_points"
OFF_AXIS_ODLC_TOPIC = "off_axis_odlc_pos"
EMERGENT_LAST_KNOWN_TOPIC = "emergent_last_known_pos"
UGV_DRIVE_TOPIC = "ugv_drive_pos"
LOST_COMM_POS_TOPIC = "lost_comms_pos"
AIR_DROP_POS_TOPIC = "air_drop_pos"
MAP_CENTER_POS_TOPIC = "map_center_pos"
STATIONARY_OBSTACLES_TOPIC = "stationary_obstacles"

ODLC_TOPIC = "odlc"
MAPS_TOPIC = "map"
TELEMETRY_TOPIC = "telemetry"


def talker():
    mission_data = client.get_mission(1)

    waypoints_pub = rospy.Publisher(
        WAYPOINTS_TOPIC,
        WayPoints,
        queue_size=10,
    )
    flyzones_pub = rospy.Publisher(
        FLYZONES_TOPIC,
        FlyZone,
        queue_size=10,
    )
    search_grid_pub = rospy.Publisher(
        SEARCH_GRID_TOPIC,
        GeoPolygonStamped,
        queue_size=10,
    )
    air_drop_boundary_pub = rospy.Publisher(
        AIR_DROP_BOUNDARY_TOPIC,
        GeoPointStamped,
        queue_size=10,
    )
    off_axis_odlc_pub = rospy.Publisher(
        OFF_AXIS_ODLC_TOPIC,
        GeoPointStamped,
        queue_size=10,
    )
    emergent_last_known_pub = rospy.Publisher(
        EMERGENT_LAST_KNOWN_TOPIC,
        GeoPointStamped,
        queue_size=10,
    )
    ugv_drive_pub = rospy.Publisher(
        UGV_DRIVE_TOPIC,
        GeoPointStamped,
        queue_size=10,
    )
    lost_comms_pos_pub = rospy.Publisher(
        LOST_COMM_POS_TOPIC,
        GeoPointStamped,
        queue_size=10,
    )
    air_drop_pos_pub = rospy.Publisher(
        AIR_DROP_POS_TOPIC,
        GeoPointStamped,
        queue_size=10,
    )
    map_center_pos_pub = rospy.Publisher(
        MAP_CENTER_POS_TOPIC,
        GeoPointStamped,
        queue_size=10,
    )
    stationary_obstacles_pub = rospy.Publisher(
        STATIONARY_OBSTACLES_TOPIC,
        GeoCylinderArrayStamped,
        queue_size=10,
    )

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        waypoints_pub.publish(mission_data[WAYPOINTS_TOPIC])
        flyzones_pub.publish(mission_data[FLYZONES_TOPIC])
        search_grid_pub.publish(mission_data[SEARCH_GRID_TOPIC])
        air_drop_boundary_pub.publish(mission_data[AIR_DROP_BOUNDARY_TOPIC])
        off_axis_odlc_pub.publish(mission_data[OFF_AXIS_ODLC_TOPIC])
        emergent_last_known_pub.publish(mission_data[EMERGENT_LAST_KNOWN_TOPIC])
        ugv_drive_pub.publish(mission_data[UGV_DRIVE_TOPIC])
        lost_comms_pos_pub.publish(mission_data[LOST_COMM_POS_TOPIC])
        air_drop_pos_pub.publish(mission_data[AIR_DROP_POS_TOPIC])
        map_center_pos_pub.publish(mission_data[MAP_CENTER_POS_TOPIC])
        stationary_obstacles_pub.publish(mission_data[STATIONARY_OBSTACLES_TOPIC])

        rate.sleep()


def odlc_callback(data):
    client.upload_odlc(data)


def maps_callback(data):
    client.upload_map(1, data)


def telemetry_callback(data):
    client.upload_telemetry(data)


def listener():
    rospy.Subscriber(ODLC_TOPIC, Odlc, odlc_callback)
    rospy.Subscriber(MAPS_TOPIC, Map, maps_callback)
    rospy.Subscriber(TELEMETRY_TOPIC, Telemetry, telemetry_callback)
    rospy.spin()


def main():
    rospy.init_node("interop_client", anonymous=True)

    global client
    client = InteropClient(SERVER_URL, USERNAME, PASSWORD)

    listenerThread = threading.Thread(target=listener)
    listenerThread.setDaemon(True)
    listenerThread.start()

    talker()


if __name__ == "__main__":
    main()
