#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from tile_image import Tiler
import math

from interpreter import TargetInterpreter, Target
from drawer import TargetDrawer
class Target_Localizer:
    def __init__(self):
        self.sub = rospy.Subscriber('/image_topic_2', Image, self.callback, queue_size = None) # limit to 1 at a time
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/target_bbox', Image, queue_size= 10)
        self.pub_meta = rospy.Publisher('/target_bbox_meta', String, queue_size = 10)
        
        model_name = '14_efficientdetd1_edgetpu.tflite'
        label_file = 'labels_formatted.txt'
        self.inter_TPU = TargetInterpreter(model_name, label_file, 'tpu', 0.33, order_key='efficientdetd2')
        self.tiler = Tiler(448,100)
        self.draw = TargetDrawer(self.inter_TPU.labels)

    def callback(self, data):
        cv_image = None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as err:
            print(err)
        targets = []
        print('Starting Tiling Process')
        for (hl,hu), (wl,wu), (i,j) in self.tiler.get_tiles(cv_image.shape):
            tile = cv_image[hl:hu , wl:wu, :]

            self.inter_TPU.interpret(tile)
            for t in self.inter_TPU.targets:
                bbox = self.tiler.tile2board(t.bbox, wl, hl)
                targets.append( Target(t.id, t.score, bbox) )
            targets = self.tiler.merge_overlapping(targets)
            #debating whether i need this or not
        print('Done with Tiling Process')

        '''
        display_img = np.zeros_like(cv_image)
        display_img = self.draw.draw_all(cv_image, targets)
        display_img = cv2.resize(display_img, (1000,1000), interpolation=cv2.INTER_AREA)

        cv2.imshow('Stream', display_img)
        cv2.waitKey(3)
        '''

        w,h = cv_image.shape[1], cv_image.shape[0]

        
        for t in targets:
            bbox = t.bbox
            shape = self.inter_TPU.labels[t.id]
            
            xmin, xmax = math.ceil(bbox.xmin*w), math.floor(bbox.xmax * w)
            ymin, ymax = math.ceil(bbox.ymin*h), math.floor(bbox.ymax * h)
            
            target_crop = cv_image[ymin:ymax, xmin:xmax]
            message_str = f'{shape}'
            self.pub.publish( self.bridge.cv2_to_imgmsg(target_crop, 'bgr8') )
            self.pub_meta.publish( message_str )
        

if __name__ == '__main__':
    target_localizer = Target_Localizer()
    rospy.init_node('Target_Localier', anonymous=True)
    rospy.spin()

