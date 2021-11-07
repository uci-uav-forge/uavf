import math, cv2, os
import numpy as np
from argparse import ArgumentParser
from time import time
from pathlib import Path

from interpreter import BBox, Target


class Tiler(object):
    def __init__(self, size):
        # size of square image, e.g. 500 -> (500,500) tile
        self.size = size

    def tile_image(self, cv2imgarr):
        h, w = cv2imgarr.shape[0], cv2imgarr.shape[1]
        # no of htiles
        self.ht = h // self.size
        # no of wtiles
        self.wt = w // self.size

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
        # total width of tiled part of image
        self.tiledw = self.ht * self.size
        # total height of tiled part of image
        self.tiledh = self.wt * self.size

        for i in range(self.wt):
            tilemap[i] = {}
            for j in range(self.ht):
                hlower = self.size * j + hremain2
                hupper = self.size * (j + 1) + hremain2
                wlower = self.size * i + wremain2
                wupper = self.size * (i + 1) + wremain2
                yield (hlower, hupper), (wlower, wupper), (i, j)

    def run_interp_tiles(self, cv2imgarr, interpreter):
        """Tile an image and run interpreter over tiles"""
        image_map = {}
        for (hl, hu), (wl, wu), (i, j) in self.tile_image(cv2imgarr):
            interpreter.interpret(cv2imgarr[hl:hu, wl:wu, :])
            image_map[(i, j)] = {"targets": [t for t in interpreter.targets]}
        return image_map

    def get_ineighs(self, i):
        neighs = []
        if i > 0:
            neighs.append(i - 1)
        if i < self.wt - 1:
            neighs.append(i + 1)
        return neighs

    def get_jneighs(self, j):
        neighs = []
        if j > 0:
            neighs.append(j - 1)
        if j < self.ht - 1:
            neighs.append(j + 1)
        return neighs

    def touching_wall(self, bb, epsilon=0.1):
        """check if a bbox is touching a wall. return None if it is not,
        else return the wall.

        walls are given by a list in order left top right bottom

        true if touching false if not touching

        touching is true if the edge of the bbox is within epsilon percent of the wall
        i.e. epsilon=0.05 on a 100 pixel image means if the wall is within 5 pixels of
        the wall it will return true.
        """
        x1, y1, x2, y2 = bb
        walls = np.array([False, False, False, False], dtype=bool)
        if x1 <= epsilon:
            walls[0] = True
        if y1 <= epsilon:
            walls[1] = True
        if x2 >= 1 - epsilon:
            walls[2] = True
        if y2 >= 1 - epsilon:
            walls[3] = True
        return walls

    def wall_match(self, bbA, bbB):
        """check if there is a 'wall match' between two target bboxes
        given by bbA and bbB.

        that is, if they touch each other through a wall, and if so, which
        walls.
        """
        wA = self.touching_wall(bbA)
        wallB = self.touching_wall(bbB)
        # to compare, we need to re-order so that the left wall of A faces
        # the right wall of B and the top wall of A faces the bottom wall of
        # B.
        wB = np.empty_like(wallB)
        # left
        wB[0] = wallB[2]
        # top
        wB[1] = wallB[3]
        # right
        wB[2] = wallB[0]
        # bottom
        wB[3] = wallB[1]
        return np.logical_and(wA, wB)

    def glob_annos(self, image_map):
        """glob annotations together."""
        globbed = []
        for (i, j), t1s in image_map.items():
            print("Tile {}".format((i, j)))
            for t1 in t1s["targets"]:
                id = t1.id
                # left neighbor
                if i >= 1:
                    for t2 in image_map[(i - 1, j)]["targets"]:
                        if self.wall_match(t1.bbox, t2.bbox)[0] and t1.id == t2.id:
                            print("\tShared Target: {}<->{}".format((i, j), (i - 1, j)))
                            t1board = self.tile2board((i, j), t1.bbox)
                            t2board = self.tile2board((i - 1, j), t2.bbox)
                            new_bbox = self.get_bbox_for2(t1board, t2board)
                            globbed_tgt = Target(id=id, score=1.0, bbox=new_bbox)
                            globbed.append(globbed_tgt)
                # top neighbor
                if j >= 1:
                    for t2 in image_map[(i, j - 1)]["targets"]:
                        if self.wall_match(t1.bbox, t2.bbox)[1] and t1.id == t2.id:
                            print("\tShared Target: {}<->{}".format((i, j), (i, j - 1)))
                            t1board = self.tile2board((i, j), t1.bbox)
                            t2board = self.tile2board((i, j - 1), t2.bbox)
                            new_bbox = self.get_bbox_for2(t1board, t2board)
                            globbed_tgt = Target(id=id, score=1.0, bbox=new_bbox)
                            globbed.append(globbed_tgt)

                # right neighbor
                if i < self.wt - 1:
                    for t2 in image_map[(i + 1, j)]["targets"]:
                        if self.wall_match(t1.bbox, t2.bbox)[2] and t1.id == t2.id:
                            print("\tShared Target: {}<->{}".format((i, j), (i + 1, j)))
                            t1board = self.tile2board((i, j), t1.bbox)
                            t2board = self.tile2board((i + 1, j), t2.bbox)
                            new_bbox = self.get_bbox_for2(t1board, t2board)
                            globbed_tgt = Target(id=id, score=1.0, bbox=new_bbox)
                            globbed.append(globbed_tgt)

                # bottom neighbor
                if j < self.ht - 1:
                    for t2 in image_map[(i, j + 1)]["targets"]:
                        if self.wall_match(t1.bbox, t2.bbox)[3] and t1.id == t2.id:
                            print("\tShared Target: {}<->{}".format((i, j), (i, j + 1)))
                            t1board = self.tile2board((i, j), t1.bbox)
                            t2board = self.tile2board((i, j + 1), t2.bbox)
                            new_bbox = self.get_bbox_for2(t1board, t2board)
                            globbed_tgt = Target(id=id, score=1.0, bbox=new_bbox)
                            globbed.append(globbed_tgt)
        return globbed

    def tile2board(self, location, bbox):
        """convert "tile coordinates" to "board coordinates"

        bbox is a bbox
        location is a tuple (i, j) indicating which tile
        that bbox belongs to

        returns `bbox` object with new coordinates
        """
        i, j = location
        x1, y1, x2, y2 = bbox
        # first x
        x1board = (i + x1) / self.wt
        x2board = (i + x2) / self.wt

        y1board = (j + y1) / self.ht
        y2board = (j + y2) / self.ht

        return BBox(
            xmin=x1board,
            ymin=y1board,
            xmax=x2board,
            ymax=y2board,
        )

    def get_bbox_for2(self, bbA, bbB):
        """from two bboxes in board coordinates, get a boox
        in board coordinates that contains them both."""
        new_xmin = min(bbA.xmin, bbB.xmin)
        new_ymin = min(bbA.ymin, bbB.ymin)
        new_xmax = max(bbA.xmax, bbB.xmax)
        new_ymax = max(bbA.ymax, bbB.ymax)
        return BBox(new_xmin, new_ymin, new_xmax, new_ymax)

    def parse_localTarget(self, target: 'Target object from Interpreter', wl: int, hl: int):
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
    target_interpreter = TargetInterpreter(opts.model, opts.labels, "tpu", 0.33)
    img_path = Path(opts.i).resolve()
    print("Reading: ", img_path)

    raw_img = cv2.imread(str(img_path))

    drawn_targets = np.zeros_like(raw_img)

    draw = TargetDrawer(target_interpreter.labels)
    tiler = Tiler(300)

    image_map = tiler.run_interp_tiles(raw_img, target_interpreter)
    print("Globbing...")
    globbed = tiler.glob_annos(image_map)

    print("Drawing individual targets on image")
    for i, ((hl, hu), (wl, wu), (_, _)) in enumerate(tiler.tile_image(raw_img)):
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
        drawn_targets[hl:hu, wl:wu, :] = tile_seg

    drawn_targets = draw.draw_all(drawn_targets, globbed, color=(255, 255, 0))

    cv2.imwrite("./tgts.jpg", drawn_targets)
