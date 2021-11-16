from pipeline import TargetDrawer
from interpreter import TargetInterpreter, BBox, Target
from tile_image import Tiler
from vsutils import VideoStreamCV


import geoLocation
from Target import Target, TargetList


import argparse
from termcolor import colored
import cv2
import numpy as np


#GLOBAL DRAW PARAMETERS
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
color = (0,255,255)
thickness = 1



#GLOBAL TEST PARAMETERS
DEPTH = 50 #arbitrary

def drawCoord(img, coord):
    x,y = coord
    img = cv2.putText(img, f'({x},{y})', (x,y+18), font, fontScale, color, thickness, cv2.LINE_AA)
    img = cv2.circle(img, coord, radius=0, color=color, thickness=5)
    return img
if __name__ == '__main__':
    cam = VideoStreamCV(src=0)
    params = np.load('calibration/webCamParams.npy', allow_pickle=True)
    cam.give_params(params)


    interTPU = TargetInterpreter('model_edgetpu.tflite', 'labels.txt', 'tpu', 0.33,order_key='efficientdetd2')
    draw = TargetDrawer(interTPU.labels)
    tiler = Tiler(448,100)


    while True:
        targets = []
        img = cam.get_img()

        gen = tiler.get_tiles(img.shape)

        for (hl,hu), (wl,wu), (i,j) in tiler.get_tiles(img.shape):
            tile_img = img[hl:hu, wl:wu, :]
            interTPU.interpret(tile_img)
            for t in interTPU.targets:
                targets.append(tiler.parse_localTarget(t,wl,hl))
        

        targets = tiler.merge_overlapping(targets)
        for t in targets:
            #test geoLocation relative distance calc
            x_center = int((t.bbox.xmin + (t.bbox.xmax - t.bbox.xmin)/2) * img.shape[1])
            y_center = int((t.bbox.ymin + (t.bbox.ymax - t.bbox.ymin)/2) * img.shape[0])
            img = drawCoord(img, (x_center,y_center))
        img = draw.draw_all(img, targets)
        cv2.imshow('image',img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
