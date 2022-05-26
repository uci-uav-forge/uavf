# testing file download and parsing without including ROS
# NEW PLAN: create a topic, parse mission file and publish
# index to it, ugv-drop subscribes and receives index
import rospy
from std_msgs.msg import Int16


def wp_index():
    mission = open('TestArcMission.waypoints', 'rt')
    mission_list = mission.readlines()
    # print(mission_list)
    for line in mission_list:
        if line.find('\t20\t') != -1:
            return line[0]
    print('drop waypoint not found')


def drop_pub():
    pub = rospy.Publisher('drop-waypoint', Int16, queue_size=1)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        connections = pub.get_num_connections()
        if connections > 0:
            drop_index = wp_index()
            pub.publish(int(drop_index))
            rospy.loginfo('Published')
            break
        rate.sleep()


def main():
    try:
        rospy.init_node('drop-publisher', anonymous=True)
        drop_pub()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
