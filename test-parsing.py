# testing file download and parsing without including ROS
#NEW PLAN: create a topic, parse mission file and publish
# index to it, ugv-drop subscribes and receives index
import rospy

def wp_index():

    mission = open('TestArcMission.waypoints', 'rt')

    for line in mission:
        print(line)

def main():
    wp_index()

if __name__ == '__main__':
    main()