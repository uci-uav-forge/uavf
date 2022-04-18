from inference import TargetInterpreter, Target, Tiler
from utils.drawer import TargetDrawer
import logging
import cv2, time
import numpy as np
from color import Color
from uuid import uuid4


class FoundTarget(object):
    def __init__(self):
        self.uuid = uuid4()  # unique target id
        self.image = None  # image of the target
        self.shape = None  # shape
        self.character = None  # letter/number
        self.shapecolor = None  # shape color
        self.charcolor = None  # character color
        self.coord_gps = None  # GPS coordinate
        self.coord_local = None  # local "image" coordinate
        self.time = None  # time detected
        self.raw_frame = None  # raw image from which this targ was extracted
        self.telemetry = None  # vehicle telemetry upon capture
        self.target = None  # target object corresponding to this object


class Pipeline(object):
    def __init__(self, interpreter, tiler, color, drawer=None):
        self.interpreter = interpreter
        self.tiler = tiler
        if drawer is not None:
            self.drawer = drawer  # drawer object
            self.drawn = None  # drawn image
        self.color = color

    def inference_over_tiles(self, raw, resize=False):
        logging.info(f"Performing inference on w={raw.shape[1]}, h={raw.shape[0]} ...")
        draw = self.drawer is not None
        t0, tiles, times = time.time(), 0, []
        all_targs = []
        if draw:
            drawn = np.zeros_like(raw)
        for (hl, hu), (wl, wu), (i, j) in self.tiler.get_tiles(raw.shape):
            tiles += 1
            inp = raw[hl:hu, wl:wu, :]
            if draw:
                img = self.drawer.draw_tile_frame(inp)
            it0 = time.time()
            self.interpreter.interpret(inp, resize=resize)
            times.append(time.time() - it0)
            for t in self.interpreter.targets:
                bbox = self.tiler.tile2board(t.bbox, wl, hl)
                all_targs.append(Target(t.id, t.score, bbox))
            if draw:
                drawn[hl:hu, wl:wu, :] = inp
            all_targs = self.tiler.merge_overlapping(all_targs)
        logging.info(
            f"Inference took {round(time.time() - t0, 3)} seconds for {tiles} tiles."
        )
        times = np.array(times)

        logging.info(f"Inference took {round(np.mean(times), 3)} seconds per tile.")

        if draw:
            logging.info(f"Draw {len(all_targs)} onto image.")
            self.drawer.draw_all(drawn, all_targs)
            self.drawn = drawn

        return all_targs

    def parse_targets(self, raw, targets, interpreter):
        for target in targets:
            xmin, ymin, xmax, ymax = (
                int(raw.shape[1] * target.bbox.xmin),
                int(raw.shape[0] * target.bbox.ymin),
                int(raw.shape[1] * target.bbox.xmax),
                int(raw.shape[0] * target.bbox.ymax),
            )
            yield {
                "image": raw[ymin:ymax, xmin:xmax, :],
                "id": target.id,
                "class": interpreter.labels[target.id],
                "score": target.score,
            }

    def process_color(self, cropped_img):
        lmask, shape_color, letter_color = self.color.target_segmentation(cropped_img)
        return lmask, shape_color, letter_color

    def _mask_compare(self, img, mask, sz=(400, 400)):
        """utility to compare the extracted letter mask to the image side-by-side"""
        rsz_img = cv2.resize(img, sz)
        rsz_mask = cv2.resize(mask, sz)
        rsz_mask = cv2.cvtColor(rsz_mask, cv2.COLOR_GRAY2RGB)
        return np.concatenate((rsz_img, rsz_mask), axis=1)

    def run(self, raw, gps):
        """Primary run method for the imaging pipeline"""
        targets = self.inference_over_tiles(raw)
        for targ in self.parse_targets(raw, targets, self.interpreter):
            logging.info(f"Target: class={targ['class']},\t score={targ['score']}")
            lmask, shape_color, letter_color = self.process_color(targ["image"])
            scolor_str = self.color.get_readable_color(shape_color)
            lcolor_str = self.color.get_readable_color(letter_color)
            logging.info(
                f"\tTarget shapecolor: {scolor_str} lettercollor: {lcolor_str}"
            )

            cv2.imshow("Target", self._mask_compare(targ["image"], lmask))
            cv2.waitKey(2500)


if __name__ == "__main__":
    MODEL_PATH = "./models/efficientdet_lite0_320_ptq_edgetpu.tflite"

    logging.basicConfig(
        format="%(levelname)s:%(processName)s@%(module)s\t%(message)s",
        level=logging.INFO,
    )

    FILE_PATH = "./example_images/plaza_sm.jpg"

    interpreter = TargetInterpreter(
        MODEL_PATH,
        "./models/coco_labels.txt",
        "tpu",
        thresh=0.4,
        order_key="efficientdetd3",
    )
    tiler = Tiler(320, 50)
    drawer = TargetDrawer(interpreter.labels)
    color = Color()
    pipeline = Pipeline(interpreter, tiler, color, drawer)
    image_raw = cv2.imread(FILE_PATH)
    pipeline.run(image_raw, None)

    cv2.imshow("image", cv2.resize(pipeline.drawn, (1600, 1200)))
    cv2.waitKey(0)
