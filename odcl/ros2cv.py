#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

class cvBridgeRec:
    def __init__(self):
        self.sub = rospy.Subscriber('image_topic_2', Image, self.callback)
        self.bridge = CvBridge()
    def callback(self, data):
        cv_image = None
        try:
            #cv_image = self.bridge.imgmsg_to_cv2(data,'bgr8', desired_encoding='passthrough')
            cv_image = data
            cv_image = self.bridge.imgmsg_to_cv2( cv_image, "bgr8")
        except CvBridgeError as err:
            print(err)
        #print(cv_image)
        cv2.imshow('Test Window', cv_image)
        cv2.waitKey(3)
if __name__ == '__main__':
    test = cvBridgeRec()
    rospy.init_node('image_receiver', anonymous=True)
    rospy.spin()
