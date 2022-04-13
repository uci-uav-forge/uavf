#!/usr/bin/env python3

from Color import Color
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

import sklearn as sk
from sklearn import neighbors

color_dict = {0:'white', 1:'black', 2:'gray', 3:'red', 4:'blue', 5:'green', 6:'yellow', 7:'purple', 8:'brown', 9:'orange'}
class odcl_pipeline:
    def __init__(self):

        # run geolocation
        # run color
        # run letter
        self.color = Color()
        self.sub_crop = rospy.Subscriber('/target_bbox', Image, self.crop_callback, queue_size=20)
        self.sub_meta = rospy.Subscriber('/target_boox_meta', String, self.meta_callback, queue_size=20)
        self.bridge = CvBridge()
        self.crop_id = 0
        self.meta_id = 0

        colorData = np.load('labDataset.npy')
        Xtr = colorData[:,:3]
        Ytr = colorData[:,3]
        self.colorClassifier = neighbors.KNeighborsClassifier(n_neighbors=1)
        self.colorClassifier.fit(Xtr,Ytr)

    def crop_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data,'bgr8')
        except CvBridgeError as err:
            print(err)
        letter_mask, shapeColor, letterColor = self.color.target_segmentation(cv_image)
        cv2.imshow('mask', letter_mask)
        cv2.waitKey(3)

        letterColor = cv2.cvtColor(letterColor.reshape(1,1,3), cv2.COLOR_BGR2LAB)

        color_id = self.colorClassifier.predict( letterColor.reshape(1,-1) )[0]

        print(color_id)

        colorName = color_dict[ color_id ]
        print(colorName)
        self.crop_id+=1
        pass
    def meta_callback(self, data):
        self.meta_id += 1
        pass

if __name__ == '__main__':
    node = odcl_pipeline()
    rospy.init_node('odcl_node',anonymous=True)
    rospy.spin()
