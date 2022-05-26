#!/usr/bin/python

import rospy
import cv2
import time, datetime
import os, re
import numpy as np
from pathlib import Path

# uavfpy module code
from uavfpy1.odcl.inference import TargetInterpreter, Tiler
from uavfpy1.odcl.color import Color
from uavfpy1.odcl.pipeline import Pipeline
from uavfpy1.odcl.location import Geolocation
from uavfpy1.odcl.utils.drawer import TargetDrawer

# mavros package imports
from mavros_msgs.msg import State
from sensor_msgs.msg import Imu, NavSatFix

BASE_DIR = "."


class OdclNode:
    def __init__(self, model_path, labels_path):
        interpreter = TargetInterpreter(
            model_path, labels_path, "tpu", thresh=0.4, order_key="efficientdetd2"
        )
        tiler = Tiler(384, 50)
        drawer = TargetDrawer(interpreter.labels)
        color = Color()
        geolocator = Geolocation()

        self.pipeline = Pipeline(interpreter, tiler, color, geolocator, drawer)

        self.GPS_sub = rospy.Subscriber(
            "/mavros/global_position/global", NavSatFix, self.GPS_callback
        )
        self.IMU_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.IMU_callback)

        self.longitude = 0
        self.latitude = 0
        self.altitude = 1

        self.quat = (0, 0, 0, 1)
        self.gps = (0, 0)
        self.altitude = 0

    def GPS_callback(self, msg):
        self.gps = (msg.latitude, msg.longitude)
        self.altitude = msg.altitude

    def IMU_callback(self, msg):
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w
        self.quat = (x, y, z, w)


def get_next_in_dir(datadir: Path, suffix: str, k=5):
    """Get next image in directory `datadir` with optional suffix `suffix`

    Parameters
    ----------
    datadir : Path
        _description_
    suffix : str, by default ""
        optional suffix for files
    k : int, optional
        enumeration prefix length, by default 5
        for example, with k=5, images are labeled
        00000
        00001
        00002
        ...

    Returns
    -------
    str
        string with enumeration length k plus suffix
    """
    # find length k collection of digits
    digit_re = k * [r"\d"]
    # find suffix and only suffix
    suffix_re = rf"({suffix})"
    # a regex with the two together
    fname_re = re.compile(rf"/{digit_re}{suffix_re}/")

    # get max num using re
    max_num = 0
    for name in os.listdir(datadir):
        search = re.search(fname_re, name)
        if search is not None:
            current = int(search.group(0)[:k])
            if max_num < current:
                max_num = current

    # return padded string
    return str(max_num + 1).rjust(k, "0") + suffix


if __name__ == "__main__":
    rospy.init_node("odcl_node", anonymous=True)
    # odcl data dir is set to a rospy param
    data_dir = rospy.get_param("/odcl/odcldir", "~/odcldata")
    # make a folder with today's date
    data_dir = Path(data_dir).resolve() / datetime.date.today().strftime("%Y%m%d")
    os.makedirs(data_dir, exist_ok=True)
    # make a directory to save raw images
    raw_image_dir = data_dir / "rawimg"
    # make a directory to save cropped images
    crp_image_dir = data_dir / "crpimg"
    for dir in (raw_image_dir, crp_image_dir):
        os.makedirs(dir, exist_ok=True)

    # save odcl data txtfile
    odcl_data_num = get_next_in_dir(data_dir, "-data.csv")
    odcl_data_file = data_dir / odcl_data_num + "-data.csv"
    # write headers to file
    headers = "frame_no,in-frame-no,img_path,score,class,shapecolor,lettercolor,latitude,longitude\n"
    with open(odcl_data_file, "a") as f:
        f.write(headers)

    model_path = "efdet.tflite"
    labels_path = "labels.txt"

    vid_capture_src = 0
    capture = cv2.VideoCapture(vid_capture_src)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
    odcl_node = OdclNode(model_path, labels_path)

    frameno = 0
    while True:
        if capture.isOpened():
            rospy.loginfo(f"New frame {frameno}")

            # read raw image from camera
            status, image_raw = capture.read()

            # save raw image and log
            raw_img_path = str(
                raw_image_dir / get_next_in_dir(raw_image_dir, suffix="-raw.jpg")
            )
            cv2.imwrite(raw_img_path)
            rospy.loginfo(f"Raw image saved to {raw_img_path}")

            # run the pipeline
            targets = odcl_node.pipeline.run(
                image_raw,
                odcl_node.gps,
                odcl_node.altitude,
                quat=odcl_node.quat,
            )

            rospy.loginfo("Performed inference on raw image.")
            rospy.loginfo(f"Found {len(targets)} targets.")
            frameno += 1

            for i, target in enumerate(targets):
                # unpack target information
                lat = target["lat"]
                lon = target["lon"]
                scolor = target["scolor_str"]
                lcolor = target["lcolor_str"]
                shape = target["class"]
                img = target["croppedimg"]
                # score -> percentage
                score = round(target["score"], 3)

                # log to rospy
                targetinfo = f"Target {i}/{len(targets)}: shape={shape} ({score*100}%), scolor={scolor}, lcolor={lcolor}, lat={lat}, lon={lon}"
                rospy.loginfo(targetinfo)

                # write to csv
                targetcsvline = f"{frameno},{i},{raw_img_path},{score},{shape},{scolor},{lcolor},{lat},{lon}\n"
                with open(odcl_data_file, "a") as f:
                    f.write(targetcsvline)

                # save cropped image with unique identifier
                frameno_str = str(frameno).rjust(5)
                i_str = str(i).rjust(2)
                identifier = odcl_data_num[:5] + frameno_str + i_str
                crop_fname = crp_image_dir / str(identifier + "jpg")
                cv2.imwrite(crop_fname, img)
