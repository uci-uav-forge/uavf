import logging
import cv2, time
import numpy as np
from uuid import uuid4

# relative imports
from .inference import TargetInterpreter, Target, Tiler
from .utils.drawer import TargetDrawer
from .color import Color
from .location import Geolocation 

class FoundTarget(object):
    def __init__(self, **kwargs):
        self.uuid = uuid4()


class Pipeline(object):
    def __init__(self, interpreter, tiler, color, geolocator, drawer=None):
        self.interpreter = interpreter
        self.tiler = tiler
        self.img = None  # raw image
        if drawer is not None:
            self.drawer = drawer  # drawer object
            self.drawn = None  # drawn image
        else:
            self.drawer = None
            self.drawn = None
        self.color = color
        self.foundtargets = {}
        self.geolocator = geolocator

    def inference_over_tiles(self, raw, resize=False):
        logging.info(f"Performing inference on w={raw.shape[1]}, h={raw.shape[0]} ...")
        draw = self.drawer is not None
        t0, tiles, times = time.time(), 0, []
        all_targs = []
        if draw:
            # get empty draw
            drawn = np.zeros_like(raw)

        for (hl, hu), (wl, wu), (i, j) in self.tiler.get_tiles(raw.shape):
            tiles += 1
            interpreter_input = raw[hl:hu, wl:wu, :]
            if draw:
                # draw the frame
                draw_frame = self.drawer.draw_tile_frame(interpreter_input)
            it0 = time.time()
            self.interpreter.interpret(interpreter_input, resize=resize)
            times.append(time.time() - it0)
            for t in self.interpreter.targets:
                bbox = self.tiler.tile2board(t.bbox, wl, hl)
                all_targs.append(Target(t.id, t.score, bbox))
            if draw:
                drawn[hl:hu, wl:wu, :] = draw_frame
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
                "bbox_center": ((xmin+xmax)/2, (ymin+ymax)/2),
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

    def run(self, raw, gps, altitude, quat=None):
        """Primary run method for the imaging pipeline"""

        '''
            Parameters:
            -----------
            raw: MxN image
            gps: (latitude,longitude) both scalars that represent current GPS coordinate of UAV.
            altitude: altitude of the UAV in METERS
            pitch: scalar that represents current UAV pitch rotation.
            roll: scalar that represents current UAV roll rotation.
            yaw: scalar that represents current UAV yaw rotation

            Returns:
            --------
            List of target data as a tuple of:
                target_lat: Latitude of the located target on the ground,
                target_lon: Longitude of the located target on the ground,
                scolor_str: Name of the color of the shape on the target,
                lcolor_str: Name of the color of the letter on the target
            in the event that multiple targets are found in the image.
        '''
        # this is the expensive method
        targets = self.inference_over_tiles(raw)

        target_data_list = list()

        for targ in self.parse_targets(raw, targets, self.interpreter):
            logging.info(f"Target: class={targ['class']},\t score={targ['score']}")

            target_lat, target_lon = self.geolocator.compute(altitude,quat,gps, targ["bbox_center"])
            lmask, shape_color, letter_color = self.process_color(targ["image"])
            scolor_str = self.color.get_readable_color(shape_color)
            lcolor_str = self.color.get_readable_color(letter_color)
            logging.info(f"\tTarget shapecolor: {scolor_str} lettercolor: {lcolor_str}")
            # cv2.imshow("Target", self._mask_compare(targ["image"], lmask))
            # cv2.waitKey(250)

            target_data_list.append((target_lat, target_lon, scolor_str, lcolor_str, targ['class']))

        return target_data_list


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
    geolocator = Geolocation()
    pipeline = Pipeline(interpreter, tiler, color, geolocator, drawer)
    image_raw = cv2.imread(FILE_PATH)
    pipeline.run(image_raw, None)

    cv2.imshow("image", cv2.resize(pipeline.drawn, (1600, 1200)))
    cv2.waitKey(0)
