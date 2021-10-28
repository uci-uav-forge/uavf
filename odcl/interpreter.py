import colorsys, platform, cv2, random, math
import numpy as np
from tflite_runtime import interpreter
from collections import namedtuple

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

TENSOR_ORDERS = {"efficientdetd2": (1, 3, 0, 2), "mobilenet": (0, 1, 2, 3)}


Target = namedtuple("Target", ["id", "score", "bbox"])
BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])


class TargetInterpreter(object):
    def __init__(self, model_path, label_path, cpu, thresh, order_key="mobilenet"):
        self.cpu = cpu
        self.labels = self.get_labels(label_path)
        self.interpreter = self.make_interpreter(model_path)
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
        self.tensor_orders_key = order_key

    def get_labels(self, label_path):
        labels = {}
        with open(label_path, "r") as f:
            for ln in f.readlines():
                cat_n, cat_label = ln.strip().split(":")
                cat_n = int(cat_n)
                labels[cat_n] = cat_label
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

    def interpret(self, img):
        self.set_input_tensor(img)
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
