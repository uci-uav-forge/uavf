# testing file download and parsing without including ROS
# NEW PLAN: create a topic, parse mission file and publish
# index to it, ugv-drop subscribes and receives index
import rospy
from std_msgs.msg import Int16


def wp_index():
    mission = open('TestArcMission.waypoints', 'rt')
    mission_list = mission.readlines()
    print(mission_list)

    for line in mission_list:
        if line.find('\t19') != -1:
            return line[0]
    print('drop waypoint not found')


def drop_pub():
    pub = rospy.Publisher('drop-waypoint', Int16, queue_size=1)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        drop_index = wp_index()
        rospy.loginfo(drop_index)
        pub.publish(drop_index)
        rate.sleep()


def main():
    try:
        wp_index()
        rospy.init_node('drop-publisher', anonymous=TRUE)
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
