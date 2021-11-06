#mike's modules for drawing, inference, tiling, and camera respectively
from pipeline import TargetDrawer
from interpreter import TargetInterpreter, BBox, Target
from tile_image import Tiler
from vsutils import VideoStreamCV


#multithreading module for inference
from threadingInterpreter import threadingInterpreter

#from threading library
from threading import Lock

#argparse to get command inputs
import argparse

from termcolor import colored

import cv2
'''
this module is a test for the multi-threading module.
it is currently not designed to support inputs.


'''

if __name__ == '__main__':
    #initialising all necessary objects
    cam = VideoStreamCV(src=1)
    interTPU = TargetInterpreter('model_edgetpu.tflite', 'labels.txt', "tpu", 0.33, order_key='efficientdetd2') 
    interCPU = TargetInterpreter('model.tflite', 'labels.txt', "cpu", 0.33, order_key='efficientdetd2')
    draw = TargetDrawer(interTPU.labels)
    tiler = Tiler(448,100)


    #initialising multi-thread safety locks
    gen_lock = Lock()
    target_lock = Lock()


    #initialising threads
    worker0 = threadingInterpreter(interTPU, tiler, gen_lock, target_lock,0)
    worker1 = threadingInterpreter(interCPU, tiler, gen_lock, target_lock,1)

    worker0.start()
    worker1.start()
    
    while True:
        img = cam.get_img()
        targets = []

        gen = tiler.get_tiles(img.shape)

        worker0.start_work(img, gen, targets)
        worker1.start_work(img, gen, targets)
        while worker0.working or worker1.working:
            pass
        #finished inference

        merged_targets = tiler.merge_overlapping(targets)
        img = draw.draw_all(img, merged_targets)
        cv2.imshow("image", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
