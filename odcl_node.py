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
    tuple(int, str)
        integer, and len(k) string representation
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
    return max_num, str(max_num + 1).rjust(k, "0")


if __name__ == "__main__":
    rospy.init_node("odcl_node", anonymous=True)
    rospy.loginfo("Started ODCL node...")

    # odcl data dir is set to a rospy param
    data_dir = rospy.get_param("/odcl/odcldir", "~/odcldata")
    # make a folder with today's date
    data_dir = Path(data_dir).resolve() / datetime.date.today().strftime("%Y%m%d")
    # make a directory to save raw images
    raw_image_dir = data_dir / "raw"
    # make a directory to save cropped images
    crp_image_dir = data_dir / "crop"
    for dir in (data_dir, raw_image_dir, crp_image_dir):
        os.makedirs(dir, exist_ok=True)

    # save odcl data txtfile
    run_id, run_id_str = get_next_in_dir(data_dir, "-data.csv")
    odcl_data_file = data_dir / str(run_id_str + "-data.csv")

    # write CSV headers to file
    headers = "run_id,frame_no,target_no,target_id,img_path,crop_path,score,class,shape_color,letter_color,latitude,longitude,inference_time\n"
    with open(odcl_data_file, "a") as f:
        f.write(headers)

    model_path = "efdet.tflite"
    labels_path = "labels.txt"

    vid_capture_src = 0
    capture = cv2.VideoCapture(vid_capture_src)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
    odcl_node = OdclNode(model_path, labels_path)

    frameno = 1
    while True:
        if capture.isOpened():
            # read raw image from camera
            status, image_raw = capture.read()

            # get frame number
            frameno_str = str(frameno).rjust(5)
            frame_id = run_id_str + "_" + frameno_str
            rospy.loginfo(f"Captured Frame {frameno} with id {frame_id}")

            # save raw image and log
            raw_img_path = raw_image_dir / str(frame_id + ".jpg")
            cv2.imwrite(raw_img_path, image_raw)
            rospy.loginfo(f"\tRaw image saved: {raw_img_path}")

            # run the pipeline
            t0 = time.time()
            targets = odcl_node.pipeline.run(
                image_raw,
                odcl_node.gps,
                odcl_node.altitude,
                quat=odcl_node.quat,
            )
            # inference time in ms
            inftime = (time.time() - t0) * 1000

            rospy.loginfo(f"\tPerformed inference on raw image, took {inftime}ms")
            rospy.loginfo(f"\tFound {len(targets)} targets.")

            for i, target in enumerate(targets):
                targetno = str(i).rjust(2)
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
                targetinfo = f"\tTarget {targetno}/{len(targets)}: shape={shape} ({score*100}%), scolor={scolor}, lcolor={lcolor}, lat={lat}, lon={lon}"
                rospy.loginfo(targetinfo)

                # save cropped image with unique identifier
                targetid = frame_id + "_" + targetno
                crop_fname = crp_image_dir / str(targetid + "jpg")
                cv2.imwrite(crop_fname, img)
                rospy.loginfo("\tSaved cropped image to {crop_fname}")

                # write to csv
                targetcsvline = f"{run_id},{frameno},{targetno},{targetid},"
                targetcsvline += f"{raw_img_path},{crop_fname},"
                targetcsvline += f"{score},{shape},{scolor},{lcolor},{lat},{lon},"
                targetcsvline += f"{inftime}\n"
                with open(odcl_data_file, "a") as f:
                    f.write(targetcsvline)

            rospy.loginfo("\tRecorded data to {odcl_data_file}")

            frameno += 1
