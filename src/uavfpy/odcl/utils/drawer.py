import numpy as np
import cv2, math, random
import colorsys


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
        # get a random color
        rgb = colorsys.hsv_to_rgb(
            random.uniform(0, 1),
            random.uniform(0.6, 1),
            random.uniform(0.8, 1),
        )
        rgb = [min(255, int(c * 255)) for c in rgb]
        return rgb

    def draw_tile_frame(self, img, alpha=0.9):
        """draw a frame around the input. Useful for visualizing tiles."""
        pt1 = (1, 1)
        pt2 = (img.shape[1] - 1, img.shape[0] - 1)
        A = img.copy()
        B = img.copy()
        A = cv2.rectangle(A, pt1, pt2, color=(255, 255, 255), thickness=2)
        return cv2.addWeighted(B, alpha, A, (1 - alpha), 0)

    def draw_target_bbox(self, img, target, color=None):
        """Draw a bbox, class label, and confidence score around a target onto image

        updated image with target drawn onto it
        """
        w, h = img.shape[1], img.shape[0]
        xmin, xmax = math.ceil(target.bbox.xmin * w), math.floor(target.bbox.xmax * w)
        ymin, ymax = math.ceil(target.bbox.ymin * h), math.floor(target.bbox.ymax * h)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        if color is None:
            color = self.colors[target.id]

        # draw rectangle
        img = cv2.rectangle(img, pt1, pt2, color, 2)
        # draw text
        textpt = (pt1[0], pt1[1] + 25)
        text = self.labels[target.id] + " : " + str(round(target.score * 100))
        img = cv2.putText(
            img,
            text,
            textpt,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        return img

    def draw_all(self, img, targets, color=None):
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
            img = self.draw_target_bbox(img, target, color=color)
        return img

    def make_target_bbox_img_opencv(self, img, targets):
        return self.draw_all(img, targets)
