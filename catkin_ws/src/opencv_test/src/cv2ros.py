#!/usr/bin/env python3

import rospy
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

class cvBridgeTest:
    def __init__(self):
        self.pub = rospy.Publisher('image_topic_2', Image, queue_size=None) # publish Image Type Messages in the ROS Format
        self.bridge = CvBridge()

if __name__ == '__main__':
    test = cvBridgeTest()
    rospy.init_node('image_converter', anonymous=True)
    rate = rospy.Rate(10)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
    while True:
        try:
            ret, img = cam.read()
            img = test.bridge.cv2_to_imgmsg(img, 'bgr8')
            test.pub.publish( img )
            rate.sleep()
            print("Sent Image\n")
        except KeyboardInterrupt:
            print('Shutting Down')
            break
    cv2.destroyAllWindows()

