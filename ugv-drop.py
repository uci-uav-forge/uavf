# ROS node that reads when the drop location has been reached and then drops the UGV
# ADD SECOND SUBSCRIBER

from pyfirmata import Arduino, util
import rospy
from mavros_msgs.msg import WaypointReached
from std_msgs.msg import Int16
import time

# wait for startup tune before running
PORT = '/dev/ttyACM2'
PIN_NUM = 'd:9:s'
SPEED = 256  # doesn't run under 62 speed
RUNTIME = 10
WP_INDEX = 2

board = Arduino(PORT)
motor = board.get_pin(PIN_NUM)


def drop_sub():
    #rospy.Subscriber('drop-waypoint', Int16, return_index())
    WP_INDEX = rospy.wait_for_message('drop-waypoint', Int16)


# def return_index(index):
    # WP_INDEX = index
    # return WP_INDEX


def wp_sub():
    rospy.Subscriber("/mavros/mission/reached", WaypointReached, wp_motor())
    rospy.spin()


def wp_motor(drop_wp):
    if drop_wp.wp_seq == WP_INDEX:
        current = time.time()
        start = time.time()
        while (current - start < RUNTIME):
            motor.write(SPEED)
            time.sleep(0.1)
            current = time.time()


# def gps_sub():
# rospy.Subscriber("/mavros/global_position/global", NavSatFix, gps_motor())
# rospy.spin()

# def gps_motor(drop_coord):

def main():
    try:
        rospy.init_node('ugv-drop', anonymous=TRUE)
        drop_sub()
        wp_sub()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
