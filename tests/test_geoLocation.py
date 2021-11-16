from pipeline import TargetDrawer
from interpreter import TargetInterpreter, BBox, Target
from tile_image import Tiler
from vsutils import VideoStreamCV


'''
    Requirements to run:
        -a .npy file that has all of the camera calibration outputs
        -tpu tflite model
        -labels.txt file
'''


'''
    Warning: this does not mean that geolocation works fully. the only way is to test this out
    in an actual flight test.

    Please confirm when we do.
'''
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

def drawCoord(img, coord, text_coord):
    '''
        Inputs:
        --------
            img: a reference to the image that will be written on. (m,n,3) array
            coord: absolute x,y coordinates of the target
            text_coord: coordinates of where text should be placed
        Outputs:
        --------
            an image with a (x,y) and a . drawn onto image (in center of bbox)
    '''
    x,y = coord
    tx,ty = text_coord
    img = cv2.putText(img, f'({x},{y})', (tx,ty+15), font, fontScale, color, thickness, cv2.LINE_AA)
    img = cv2.circle(img, (tx,ty), radius=0, color=color, thickness=5)
    return img
if __name__ == '__main__':
    #setup camera + parameters
    cam = VideoStreamCV(src=0)
    params = np.load('calibration/webCamParams.npy', allow_pickle=True)
    cam.give_params(params)

    #setup model
    interTPU = TargetInterpreter('model_edgetpu.tflite', 'labels.txt', 'tpu', 0.33,order_key='efficientdetd2')
    draw = TargetDrawer(interTPU.labels)
    tiler = Tiler(448,100)

    img = cam.get_img()
    
    #video stuff
    fourcc = "MJPG"
    frame_size = img.shape[1], img.shape[0]
    video_filename = 'test.avi'
    video_writer = cv2.VideoWriter_fourcc(*fourcc)
    video_out = cv2.VideoWriter(video_filename, video_writer, 6, frame_size)


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
            #get center of bbox and its position in image
            x_center = int((t.bbox.xmin + (t.bbox.xmax - t.bbox.xmin)/2) * img.shape[1])
            y_center = int((t.bbox.ymin + (t.bbox.ymax - t.bbox.ymin)/2) * img.shape[0])
             
            #run geoLocation to convert centers to GPS
            #most of the constants like heading, DEPTH, and GPS are set to a constant
            #run it w/ input stream once Nav gets stuff in
            x_rel, y_rel = geoLocation.get_relDist( (x_center,y_center), img.shape, (cam.focalx,cam.focaly), DEPTH)
            x_abs, y_abs = geoLocation.get_absCoord((x_rel,y_rel), 0) #asssume heading is 0. not yet integrated w/ drone
            x_gps, y_gps = geoLocation.meters_to_gps(x_abs,y_abs, (0,0))

            #draw to visualize calculation of coordinates
            img = drawCoord(img, (x_gps,y_gps), (x_center,y_center))
        img = draw.draw_all(img, targets)
        cv2.imshow('image',img)


        video_out.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_out.release()
            break
