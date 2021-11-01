import math, cv2, os
import numpy as np
from argparse import ArgumentParser
from time import time
from pathlib import Path
from interpreter import BBox, Target
import pyinstrument


class Tiler(object):
    def __init__(self, size: int, offset: int):
        # size of square image, e.g. 500 -> (500,500) tile
        if offset >= size:
            raise ValueError(
                "Can't have offset greater than size! offset={}, size={}".format(
                    offset, size
                )
            )
        self.size = size
        # tile offset in px
        self.offset = offset

    def get_tiles(self, raw_shape):
        self.h, self.w = raw_shape[0], raw_shape[1]

        a1 = self.size - self.offset

        # no of htiles
        self.no_ht = self.h // a1
        # no of wtiles
        self.no_wt = self.w // a1

        print(
            "Split {} x {} image into {} x {} tiles".format(
                self.w, self.h, self.no_wt, self.no_ht
            )
        )

        wlower, hlower, i, j = 0, 0, 0, 0
        wupper, hupper = self.size, self.size

        tiles = []
        # I'm sure there is a cleverer way to do this...
        while wupper < self.w:
            while hupper < self.h:
                tiles.append(((hlower, hupper), (wlower, wupper), (i, j)))
                hlower += a1
                hupper += a1
                j += 1
            wlower += a1
            wupper += a1
            hlower = 0
            hupper = self.size
            i += 1

        for t in tiles:
            yield t

    def tile2board(self, tbbox: BBox, wl, hl):
        return BBox(
            xmin=(tbbox.xmin * self.size + wl) / self.w,
            ymin=(tbbox.ymin * self.size + hl) / self.h,
            xmax=(tbbox.xmax * self.size + wl) / self.w,
            ymax=(tbbox.ymax * self.size + hl) / self.h,
        )

    def merge_overlapping(self):
        pass


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

    tiler = Tiler(300, -100)
    drawn = np.zeros_like(raw_img)
    all_targs = []

    profiler = pyinstrument.Profiler()

    profiler.start()
    for (hl, hu), (wl, wu), (i, j) in tiler.get_tiles(raw_img.shape):
        inp = raw_img[hl:hu, wl:wu, :]
        img = draw.draw_tile_frame(inp)
        inter_TPU.interpret(inp)
        for t in inter_TPU.targets:
            bbox = tiler.tile2board(t.bbox, wl, hl)
            all_targs.append(Target(t.id, t.score, bbox))

        drawn[hl:hu, wl:wu, :] = img
    profiler.stop()
    drawn = draw.draw_all(drawn, all_targs)

    outputfname = inp_stem + "_targets.jpg"
    print(outputfname)
    cv2.imwrite(outputfname, drawn)

    profiler.print()
