from numpy.lib.shape_base import dsplit
import numpy as np
from collections import namedtuple
from tflite_runtime import interpreter
import cv2, platform, math, random, argparse
import colorsys
from termcolor import colored

_EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]

Target = namedtuple("Target", ["id", "score", "bbox"])
BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])


class TargetInterpreter(object):
    def __init__(self, model_path, label_path, cpu):
        self.cpu = cpu
        self.interpreter = self.make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.labels = self.get_labels(label_path)
        self.targets = []

    def get_labels(self, path):
        """Get object labels from filepath

        object labels are given as  `##:object string` where ## is the integer
        object ID, one per line

        e.g. two_wheel_labels.txt
        01:bicycle
        02:motorcycle

        Parameters
        ----------
        path : str
            file path, recommended in ROS systems to use absolute path.

        Returns
        -------
        dict
            key is integer ID, val is dict of "name" and "color" values.
        """
        labels = {}
        with open(path, "r") as f:
            for line in f.readlines():
                cls, clsstr = line.strip().split(":")
                cls = int(cls) - 1
                labels[cls] = clsstr

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
        else:
            raise ValueError("Must pass `cpu` or `tpu` to constructor")

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

    def set_input_tensor(self, image):
        """set the input tensor from (cv2) image array of size (h, w c)

        Parameters
        ----------
        image : np.array
            h, w, c
        """
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

    def interpret(self, img, thresh, top_k):
        self.set_input_tensor(img)
        self.interpreter.invoke()
        self.targets = self.get_output(thresh, top_k)

    def get_output(self, score_threshold, top_k):
        """Return list of detected objects

        Parameters
        ----------
        score_threshold : float
            number from 0-1 indicating thresh percentage
        top_k : int
            no. of top objects to return

        Returns
        -------
        list of Target
            list of namedtuples containing target info
        """

        BOXES = 1
        CATID = 3
        SCORE = 0
        N = 2

        boxes = self.output_tensor(BOXES)
        category_ids = self.output_tensor(CATID)
        scores = self.output_tensor(SCORE)
        n = self.output_tensor(N)

        # for some reason, some models re-order the output tensors...
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


class TargetDrawer(object):
    def __init__(self, labels):
        # get labels
        self.labels = labels
        # get colors
        self.colors = {}
        for l in labels.keys():
            self.colors[l] = self.get_rand_color()

    @staticmethod
    def get_rand_color():
        # get a random color between 1 and 0
        return colorsys.hsv_to_rgb(
            random.randint(0, 255),
            random.randint(200, 255),
            random.randint(180, 210),
        )

    def draw_target_bbox(self, img, target):
        """Draw a bbox, class label, and confidence score around a target onto image

        Parameters
        ----------
        img : cv2 image
            (h, w, 3) 8-bit array
        target : Target
            target for drawing

        Returns
        -------
        img
            updated image with target drawn onto it
        """
        w, h = img.shape[1], img.shape[0]
        xmin, xmax = math.ceil(target.bbox.xmin * w), math.floor(target.bbox.xmax * w)
        ymin, ymax = math.ceil(target.bbox.ymin * h), math.floor(target.bbox.ymax * h)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        # draw rectangle
        img = cv2.rectangle(img, pt1, pt2, self.colors[target.id], 2)
        # draw text
        textpt = (pt1[0], pt1[1] + 25)
        text = self.labels[target.id] + " : " + str(round(target.score * 100))
        img = cv2.putText(
            img,
            text,
            textpt,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            self.colors[target.id],
            2,
        )
        return img

    def draw_all(self, img, targets):
        """Draw all current targets onto img

        Parameters
        ----------
        img : cv2 Image
            (H, W, 3) 8-bit
        targets: list of Target
            targets to draw

        Returns
        -------
        (H, W, 3) 8-bit image
            Image with targets drawn
        """
        for target in targets:
            img = self.draw_target_bbox(img, target)
        return img

    def make_target_bbox_img_opencv(self, img, targets):
        return self.draw_all(img, targets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run inference on video stream")
    ap.add_argument("--model", type=str, required=True, help="path to tflite model")
    ap.add_argument("--labels", type=str, required=True, help="path to labels file")
    ap.add_argument(
        "--cpu",
        type=str,
        required=False,
        default="cpu",
        help=colored("cpu", "blue")
        + " if using CPU, "
        + colored("tpu", "blue")
        + " if using TPU.",
    )
    opts = ap.parse_args()

    from vsutils import VideoStreamCV

    target_interpreter = TargetInterpreter(opts.model, opts.labels, opts.cpu)

    vs = VideoStreamCV()
    draw = TargetDrawer(target_interpreter.labels)
    while True:
        img = vs.get_img()
        target_interpreter.interpret(img, 0.25, 5)
        for t in target_interpreter.targets:
            obj_id_str = target_interpreter.labels[t.id]
            xmin = round(t.bbox[0], 2)
            ymin = round(t.bbox[1], 2)
            xmax = round(t.bbox[2], 2)
            ymax = round(t.bbox[3], 2)
            # obj_bbox_str = "({}, {}) to ({}, {})".format(xmin, ymin, xmax, ymax)
            # print("\tfound: id={}, bbox=[{}]".format(obj_id_str, obj_bbox_str))
        img = draw.draw_all(img, target_interpreter.targets)
        cv2.imshow("image", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
