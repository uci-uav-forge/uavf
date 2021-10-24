import math, cv2, os
import numpy as np
from argparse import ArgumentParser
from time import time
from pathlib import Path


class Tiler(object):
    def __init__(self, size):
        # size of square image, e.g. 500 -> (500,500) tile
        self.size = size

    def tile_image(self, cv2imgarr):
        h, w = cv2imgarr.shape[0], cv2imgarr.shape[1]
        htiles = h // self.size
        wtiles = w // self.size

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

        hremain2 = math.floor(h % self.size / 2)
        wremain2 = math.floor(w % self.size / 2)

        tilemap = {}

        for i in range(wtiles):
            tilemap[i] = {}
            for j in range(htiles):
                hlower = self.size * j + hremain2
                hupper = self.size * (j + 1) + hremain2
                wlower = self.size * i + wremain2
                wupper = self.size * (i + 1) + wremain2
                yield (hlower, hupper), (wlower, wupper)


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
    target_interpreter = TargetInterpreter(opts.model, opts.labels, "tpu", 0.33)
    img_path = Path(opts.i).resolve()
    print("Reading: ", img_path)

    raw_img = cv2.imread(str(img_path))

    drawn_tgts = np.zeros_like(raw_img)

    draw = TargetDrawer(target_interpreter.labels)
    tiler = Tiler(300)

    t1, ntiles = time(), 0
    for i, ((hl, hu), (wl, wu)) in enumerate(tiler.tile_image(raw_img)):
        print("Tile {}".format(i + 1))
        ntiles = i + 1
        img = raw_img[hl:hu, wl:wu, :]
        target_interpreter.interpret(img)
        for t in target_interpreter.targets:
            obj_id_str = target_interpreter.labels[t.id]
            xmin = round(t.bbox[0], 2)
            ymin = round(t.bbox[1], 2)
            xmax = round(t.bbox[2], 2)
            ymax = round(t.bbox[3], 2)
        tile_seg = draw.draw_all(img, target_interpreter.targets)
        drawn_tgts[hl:hu, wl:wu, :] = tile_seg
    t2 = time()
    print(
        "Took {} ms for {} tiles; {} ms per tile.".format(
            round((t2 - t1) * 1000), ntiles, round(((t2 - t1) / ntiles) * 1000, 4)
        )
    )
    fname = str(img_path.stem) + "_tiled" + ".jpg"
    fname = str(img_path.parent / fname)
    cv2.imwrite(fname, drawn_tgts)
