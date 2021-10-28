from interpreter import TargetInterpreter, BBox, Target
from vsutils import VideoStreamCV
import argparse
import numpy as np
import cv2
import colorsys, random, math, platform
from pipeline import TargetDrawer

def imageSegment(img, segments):
    M,N = img.shape[0]//segments, img.shape[1]//segments
    return np.array([[img[x*M:x*M+M,y*N:y*N+N] for y in range(0,segments)] for x in range(0,segments)])

def offset_targets(targets, offset, img_shape, segment):
    x_offset, y_offset = offset
    for i in range(len(targets)):
        targets[i] = Target(targets[i].id,targets[i].score,BBox(
        targets[i].bbox.xmin / segment + float(x_offset) / img_shape[1],
        targets[i].bbox.ymin / segment + float(y_offset) / img_shape[0],
        targets[i].bbox.xmax / segment + float(x_offset) / img_shape[1],
        targets[i].bbox.ymax / segment + float(y_offset) / img_shape[0]
        ))

if __name__ == '__main__':

    tpu_model = 'mobilenet_edge.tflite'
    tpu_labels = 'coco_labels.txt'

    target_interTPU = TargetInterpreter(tpu_model, tpu_labels, "tpu", 0.33, "mn")
    vs = VideoStreamCV(src=0)
    drawTPU = TargetDrawer(target_interTPU.labels)

    segment = 3
    while True:
        img = vs.get_img()
        M,N = img.shape[0]//segment, img.shape[1]//segment
        array_imgs = imageSegment(img,segment)
        targets = []
        for i in range(0,segment**2,1):
            target_interTPU.interpret(array_imgs[i//segment,i%segment])
            offset_targets(target_interTPU.targets, ((i%segment)*N, (i//segment)*M), img.shape, segment)
            targets = targets + target_interTPU.targets
        
        img = drawTPU.draw_all(img,targets)
        cv2.imshow("image",img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

