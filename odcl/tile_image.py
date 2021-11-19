import math, cv2, os
import numpy as np
from argparse import ArgumentParser
from time import time
from pathlib import Path
from interpreter import BBox, Target
import pyinstrument
import networkx as nx
import matplotlib.pyplot as plt


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
            j = 0

        for t in tiles:
            yield t

    def tile2board(self, tbbox: BBox, wl, hl):
        return BBox(
            xmin=(tbbox.xmin * self.size + wl) / self.w,
            ymin=(tbbox.ymin * self.size + hl) / self.h,
            xmax=(tbbox.xmax * self.size + wl) / self.w,
            ymax=(tbbox.ymax * self.size + hl) / self.h,
        )

    def merge_overlapping(self, targets: list):
        """probably can optimize this"""
        # build an undirected connectivity graph for targets
        T = nx.Graph()
        # assign unique integer to each target
        for v, t in enumerate(targets):
            T.add_node(v, target=t)

        # connect targets that overlap
        for i, t1 in enumerate(targets):
            for j, t2 in enumerate(targets[i + 1 :]):
                j += i + 1
                if t1.bbox.overlap(t2.bbox):
                    T.add_edge(i, j)

        merged_targets = []

        # find connected components. Connected components are tiles
        # that overlap one another.
        for c in nx.connected_components(T):
            # get list of targets
            connected_targets = [T.nodes[v]["target"] for v in c]
            # smallest xmin, ymin and largest xmax, ymax in connected component
            # so that the box covers all targets in connected components
            txmin = min(connected_targets, key=lambda t: t.bbox.xmin).bbox.xmin
            tymin = min(connected_targets, key=lambda t: t.bbox.ymin).bbox.ymin
            txmax = max(connected_targets, key=lambda t: t.bbox.xmax).bbox.xmax
            tymax = max(connected_targets, key=lambda t: t.bbox.ymax).bbox.ymax
            # target with largest confidence score
            t_scoremax = max(connected_targets, key=lambda t: t.score)
            # we use that target's score and id in the new merged target
            tscore = t_scoremax.score
            tcls = t_scoremax.id
            # create new target. If there is only a single target in the list of
            # connected components, this is just that target. But if there are
            # multiples, this will be the box that holds all of them.
            new_target = Target(
                id=tcls, score=tscore, bbox=BBox(txmin, tymin, txmax, tymax)
            )
            merged_targets.append(new_target)
        return merged_targets

    def parse_localTarget(self, target, wl: int, hl: int):
        '''
        Inputs:
        --------
            target -> Target (from interpreter.py)
            wl -> location of min x value of box
            hl -> location of min y value of box

        Outputs:
        --------
            target -> Target (from interpreter.py)

        Description:
        ------------
            reformats tiled target information (local position) to the overall image's location (global).
        '''
        bbox = self.tile2board(target.bbox, wl, hl)
        return Target(target.id, target.score, bbox)


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
    inter_TPU = TargetInterpreter(opts.model, opts.labels, "tpu", 0.33, order_key="efficientdetd2")

    draw = TargetDrawer(inter_TPU.labels)
    # for mobilenet, use `300`; for efficientdet, use `448`
    # offset does nothing for now

    inp_path = Path(opts.i).resolve()
    raw_img = cv2.imread(str(inp_path))
    tiler = Tiler(448, 100)
    drawn = np.zeros_like(raw_img)
    all_targs = []

    times = []
    for (hl, hu), (wl, wu), (i, j) in tiler.get_tiles(raw_img.shape):
        t1 = time()
        inp = raw_img[hl:hu, wl:wu, :]
            # img = draw.draw_tile_frame(inp)
        inter_TPU.interpret(inp)
        for t in inter_TPU.targets:
            bbox = tiler.tile2board(t.bbox, wl, hl)
            all_targs.append(Target(t.id, t.score, bbox))
        drawn[hl:hu, wl:wu, :] = inp
        t2 = time()
        times.append(t2-t1)
    all_targs = tiler.merge_overlapping(all_targs)
    
    times = np.array(times)
    print("Inference on {}x{} tiles".format(tiler.no_ht , tiler.no_wt))
    print("Done. per tile mean={}ms std={}ms".format(round(times.mean()*1000), round(times.std()*1000)))
    print("Took {}ms".format(round(times.sum()*1000)))

    drawn = draw.draw_all(drawn, all_targs)
    outputfname = "./" + str(inp_path.stem) + "_targets.jpg"
    print(outputfname)
    cv2.imwrite(outputfname, drawn)