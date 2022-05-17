#ROS node that reads when the drop location has been reached and then drops the UGV
#ADD MISSION FILE PARSING
import pyfirmata
import rospy
from mavros_msgs.msg import WaypointReached
from sensor_msgs.msg import NavSatFix


def waypoint_sub():
    rospy.Subscriber("/mavros/mission/reached", WaypointReached, wp_motor())
    rospy.spin()

#def wap_motor(drop_wp):
#    if drop_wp.wp_eq == 2:


def gps_sub():
    rospy.Subscriber("/mavros/global_position/global", NavSatFix, gps_motor())
    rospy.spin()

#def gps_motor(drop_coord):


if __name__ == '__main__':
    rospy.init_node('ugv-drop', anonymous=True)
    waypoint_sub()
    #gps_sub()
