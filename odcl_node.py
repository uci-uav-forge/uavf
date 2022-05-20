#!/usr/bin/python

import rospy
import cv2
import time
import os
import numpy as np

#uavfpy module code
from uavfpy1.odcl.inference import TargetInterpreter, Tiler
from uavfpy1.odcl.color import Color
from uavfpy1.odcl.pipeline import Pipeline
from uavfpy1.odcl.location import Geolocation
from uavfpy1.odcl.utils.drawer import TargetDrawer

#mavros package imports
from mavros_msgs.msg import State
from sensor_msgs.msg import Imu, NavSatFix

BASE_DIR = "."


class OdclNode:
    def __init__(self, model_path, labels_path):
        interpreter = TargetInterpreter(
            model_path, 
            labels_path, 
            "tpu", 
            thresh=0.4, 
            order_key = "efficientdetd2"
        )
        tiler = Tiler(384, 50)
        drawer = TargetDrawer(interpreter.labels)
        color = Color()
        geolocator = Geolocation()
        

        self.pipeline = Pipeline(interpreter, tiler, color, geolocator, drawer)

        self.GPS_sub = rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.GPS_callback)
        self.IMU_sub = rospy.Subscriber('/mavros/imu/data', Imu, self.IMU_callback)

        self.longitude = 0
        self.latitude = 0
        self.altitude = 1 

        self.quat = (0,0,0,1)
        self.gps = (0,0)
        self.altitude = 0

    def GPS_callback(self, msg):
        self.gps = (msg.latitude, msg.longitude)
        self.altitude = msg.altitude


    def IMU_callback(self, msg):
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w
        self.quat = (x,y,z,w)


if __name__=='__main__':
    rospy.init_node('odcl_node', anonymous=True)

    max_num = 0
    for filename in os.listdir(BASE_DIR):
        if filename.endswith("odcl_data.txt"):
            curr_num = int(filename[0: filename.index("odcl_data.txt")])
            if curr_num > max_num:
                max_num = curr_num
    f_name = str(max_num + 1) + "odcl_data.txt" 
            

    model_path = "efdet.tflite"
    labels_path = "labels.txt"
    
    vid_capture_src = 0
    capture = cv2.VideoCapture(vid_capture_src)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)

    odcl_node = OdclNode(model_path, labels_path)

    print('before whiel')
    while(True):
        if capture.isOpened():
            status, image_raw = capture.read()
            target_data_list = odcl_node.pipeline.run(image_raw, odcl_node.gps, odcl_node.altitude, quat=odcl_node.quat)
            index = 1
            for data in target_data_list:
                line_string = "{}: Target longitude: {}, Target latitude{}, Shape color: {}, Letter color: {}, Shape: {}\n".format(
                    index, 
                    data[0],
                    data[1], 
                    data[2], 
                    data[3], 
                    data[4]
                )
                print('writing')
                f = open(f_name, "a")
                f.write(line_string)
                f.close()
                print('done writing')
                index += 1
            print('done forloop')

