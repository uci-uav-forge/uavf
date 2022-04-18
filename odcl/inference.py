import platform, cv2
import numpy as np
from tflite_runtime import interpreter
from collections import namedtuple
import logging
import networkx as nx
import time

_EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]

# this is a dict that re-orders the outputs of the model.
# for some reason, some models have different orderings;
# we want the order to be (boxes, category id, score, n)
# but e.g. d2 gives it to us like
# 0: score
# 1: boxes
# 2: n
# 3: category
#
# so this dict is to re-order outputs, given different models.

TENSOR_ORDERS = {
    "efficientdetd2": (1, 3, 0, 2),
    "mobilenet": (0, 1, 2, 3),
    "efficientdetd3": (0, 1, 2, 3),
}

# added as a data type
class BBox(object):
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.area = (self.xmax - self.xmin) + (self.ymax - self.ymin)

    def overlap(self, other):
        xc = (self.xmin <= other.xmax) and (other.xmin <= self.xmax)
        yc = (self.ymin <= other.ymax) and (other.ymin <= self.ymax)
        return xc and yc


class Target(object):
    def __init__(self, id, score, bbox):
        self.id = id
        self.score = score
        self.bbox = bbox


class TargetInterpreter(object):
    def __init__(self, model_path, label_path, cpu, thresh, order_key="mobilenet"):
        self.cpu = cpu
        logging.info(f"Finding labels from {label_path}")
        self.labels = self.get_labels(label_path)
        logging.info(f"Making interpreter from {model_path}")
        self.interpreter = self.make_interpreter(model_path)
        logging.info("Allocating tensors")
        self.interpreter.allocate_tensors()
        self.targets = []
        self.thresh = thresh
        # the key for assessing tensor orders
        # there are 4 tensors returned by tflite object detectors:
        # the bboxes, classes, class scores, and number of objects detected.
        # the ordering of these tensors changes from model to model, some give
        # bbox first, then classes, etc. while others give classes first, then
        # bboxes, etc. These keys belong to `TENSOR_ORDERS` above which map the
        # correct item to its ordering.
        logging.info(f"Setting order key: {order_key}")
        self.tensor_orders_key = order_key

        t0, t1, t2, t3 = TENSOR_ORDERS[self.tensor_orders_key]
        self.category_ids = self.output_tensor(t1)
        logging.info(f"Created interpreter: {self}")

    def get_labels(self, label_path):
        labels = {}
        with open(label_path, "r") as f:
            for i, ln in enumerate(f.readlines()):
                cat_label = ln.strip()
                labels[i] = cat_label
        return labels

    def make_interpreter(self, model_path_or_content, device=None, delegate=None):
        """Make new TPU interpreter instance given a model path

        Parameters
        ----------
        model_path_or_content : str
            filepath to model. recommended to use absolute path in ROS scripts
        device : str, optional
            None -> use any TPU
            ":<n>" -> use nth TPU
            "usb" -> use USB TPU
            "usb:<n> -> use nth USB TPU
            "pci" -> use PCI TPU
            "pci:<n> -> use nth PCI TPU
        delegate : loaded TPU Delegate object, optional
            supercedes "device" flag

        Returns
        -------
        tflite.Interpreter
            the interpreter
        """
        if self.cpu == "tpu":
            if delegate:
                delegates = [delegate]
            else:
                delegates = [
                    self.load_edgetpu_delegate({"device": device} if device else {})
                ]

            if isinstance(model_path_or_content, bytes):
                return interpreter.Interpreter(
                    model_content=model_path_or_content,
                    experimental_delegates=delegates,
                )
            else:
                return interpreter.Interpreter(
                    model_path=model_path_or_content, experimental_delegates=delegates
                )
        elif self.cpu == "cpu":
            return interpreter.Interpreter(model_path=model_path_or_content)

    def load_edgetpu_delegate(self, options=None):
        """load edgetpu delegate from _EDGETPU_SHARED_LIB with options

        Parameters
        ----------
        options : dict, optional
            TPU options, by default None

        Returns
        -------
        loaded Delegate object
            the TPU
        """
        return interpreter.load_delegate(_EDGETPU_SHARED_LIB, options or {})

    def input_tensor(self):
        """get input tensor

        Returns
        -------
        tensor
            the input tensor
        """
        tensor_index = self.interpreter.get_input_details()[0]["index"]
        return self.interpreter.tensor(tensor_index)()[0]

    def set_input_tensor(self, image, resize=False):
        """set the input tensor from (cv2) image array of size (h, w c)

        Parameters
        ----------
        image : np.array
            h, w, c
        """
        if resize:
            h, w, c = self.input_image_size()
            image = cv2.resize(image, (h, w))
        self.input_tensor()[:, :, :] = image

    def output_tensor(self, i):
        """Return output tensor regardless of quantization parameters

        Parameters
        ----------
        i : int
            which output tensor to grab

        Returns
        -------
        tensor
            output tensor
        """
        output_details = self.interpreter.get_output_details()[i]
        output_data = np.squeeze(self.interpreter.tensor(output_details["index"])())
        if "quantization" not in output_details:
            return output_data
        scale, zero_point = output_details["quantization"]
        if scale == 0:
            return output_data - zero_point
        return scale * (output_data - zero_point)

    def input_image_size(self):
        """Get interpreter input size

        Returns
        -------
        tuple of int
            (height, width, colors)
        """
        _, h, w, c = self.interpreter.get_input_details()[0]["shape"]
        return h, w, c

    def interpret(self, img, resize=True):
        self.set_input_tensor(img, resize=resize)
        self.interpreter.invoke()
        self.targets = self.get_output(self.thresh)

    def get_output(self, score_threshold):
        """Return list of detected objects

        Parameters
        ----------
        score_threshold : float
            number from 0-1 indicating thresh percentage

        Returns
        -------
        list of Target
            list of namedtuples containing target info
        """
        t0, t1, t2, t3 = TENSOR_ORDERS[self.tensor_orders_key]
        boxes = self.output_tensor(t0)
        category_ids = self.output_tensor(t1)
        scores = self.output_tensor(t2)
        n = self.output_tensor(t3)

        # print(boxes.shape, category_ids.shape, scores.shape, n.shape)

        def make(i):
            ymin, xmin, ymax, xmax = boxes[i]
            return Target(
                id=int(category_ids[i]),
                score=scores[i],
                bbox=BBox(
                    xmin=np.maximum(0.0, xmin),
                    ymin=np.maximum(0.0, ymin),
                    xmax=np.minimum(1.0, xmax),
                    ymax=np.minimum(1.0, ymax),
                ),
            )

        return [make(i) for i in range(int(n)) if scores[i] >= score_threshold]


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
        logging.info(f"Created tiler with size {size} and offset {offset}")

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
        cc = list(nx.connected_components(T))

        for c in cc:
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
        """
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
        """
        bbox = self.tile2board(target.bbox, wl, hl)
        return Target(target.id, target.score, bbox)
