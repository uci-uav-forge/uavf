import math, cv2, os
import numpy as np
from argparse import ArgumentParser
from time import time
from pathlib import Path

from interpreter import BBox, Target


class Tiler(object):
    def __init__(self, size, offset):
        # size of square image, e.g. 500 -> (500,500) tile
        self.size = size
        # tile offset in px
        self.offset = offset

    def get_tiles(self, raw_shape):
        h, w = raw_shape[0], raw_shape[1]
        # no of htiles
        self.no_ht = h // self.size
        # no of wtiles
        self.no_wt = w // self.size

        # remaining tiles get halved and added like, e.g.
        # ┌───────────────┐
        # │               │
        # │ ┌─────┬─────┐ │
        # │ │     │     │ │
        # │ ├─────┼─────┤ │
        # │ │     │     │ │
        # │ └─────┴─────┘ │
        # │               │
        # └───────────────┘
        # so that the tiling is "centered" in the image.
        # we are throwing away data a little bit, but not
        # much (448 px for an efficientdet v2), and doing
        # inference on a partial area is probably no good

        # pixels from top, left that the tiling actually starts
        self.hremain2 = math.floor(h % self.size / 2)
        self.wremain2 = math.floor(w % self.size / 2)
        # and their relative values on interval [0,0]
        self.relhremain2 = self.hremain2 / (self.no_ht * self.size)
        self.relwremain2 = self.wremain2 / (self.no_wt * self.size)

        # total width of tiled part of image
        self.tiledw = self.no_ht * self.size
        # total height of tiled part of image
        self.tiledh = self.no_wt * self.size

        for i in range(self.no_wt):
            for j in range(self.no_ht):
                hlower = self.size * j + self.hremain2
                hupper = self.size * (j + 1) + self.hremain2
                wlower = self.size * i + self.wremain2
                wupper = self.size * (i + 1) + self.wremain2
                yield (hlower, hupper), (wlower, wupper), (i, j)

    def tile2board(self, tileidx, bbox):
        """convert bbox "tile coordinates" to "board coordinates"

        bbox is a bbox
        tileidx is a tuple (i, j) indicating which tile that bbox belongs to

        integer indexes i, j also work as offset values, because bboxes
        inside of tiles are given as floating points in the interval [0,1]

        returns `bbox` object with new coordinates
        """
        i, j = tileidx
        x1, y1, x2, y2 = bbox

        return BBox(
            xmin=(float(i) + x1) / self.no_wt + self.relwremain2,
            ymin=(float(j) + y1) / self.no_ht + self.relhremain2,
            xmax=(float(i) + x2) / self.no_wt + self.relwremain2,
            ymax=(float(j) + y2) / self.no_ht + self.relhremain2,
        )


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--i", type=str, required=True, help="input image")
    ap.add_argument("--model", type=str, required=True, help="model path")
    ap.add_argument("--labels", type=str, required=True, help="labels path")
    opts = ap.parse_args()
    from pipeline import TargetDrawer
    from interpreter import TargetInterpreter

    # just use mobilenet + coco_labels for testing
    # $ wget https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
    # $ mv mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite mobilenet.tflite
    # $ wget https://dl.google.com/coral/canned_models/coco_labels.txt
    inter_TPU = TargetInterpreter(opts.model, opts.labels, "tpu", 0.33)
    inp_path = Path(opts.i).resolve()
    print("Reading: ", inp_path)

    raw_img = cv2.imread(str(inp_path))
    inp_stem = str(inp_path.parent / inp_path.stem)

    draw = TargetDrawer(inter_TPU.labels)
    # for mobilenet, use `300`; for efficientdet, use `448`
    # offset does nothing for now
    tiler = Tiler(300, 25)

    drawn = np.zeros_like(raw_img)

    all_targets = []
    for (hl, hu), (wl, wu), (i, j) in tiler.get_tiles(raw_img.shape):
        tile_input = raw_img[hl:hu, wl:wu, :]
        inter_TPU.interpret(tile_input)
        if len(inter_TPU.targets) > 0:
            drawn[hl:hu, wl:wu, :] = draw.draw_tile_frame(tile_input)

        for t in inter_TPU.targets:
            all_targets.append(Target(t.id, t.score, tiler.tile2board((i, j), t.bbox)))

    drawn = draw.draw_all(drawn, all_targets, color=(56, 80, 255))

    outputfname = inp_stem + "_targets.jpg"
    print(outputfname)
    cv2.imwrite(outputfname, drawn)
