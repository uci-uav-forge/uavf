### Utilities for streaming video
from threading import Thread
import cv2, atexit, time

# Camera object. Grabs frames from camera in a new thread
# to avoid blocking main thread

"""
    Also contains camera parameters for any camera geometry needed.
"""


class VideoStreamCV(object):
    def __init__(self, src: int = 0, ready_timeout: int = 10):
        self.ready_timeout = ready_timeout
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        # camera parameters
        self.proj_mtx = None
        self.focalx = None
        self.focaly = None
        self.distortion = None

        while True:
            img = self.get_img()
            if img is not None:
                self.h, self.w = img.shape[0], img.shape[1]
                break
        atexit.register(self.exit)

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.img = self.capture.read()
            time.sleep(0.01)

    def get_img(self, undistort=False):
        try:
            if (
                undistort
            ):  # if undistort is true, undistort image before sending to stream
                h, w = self.img.shape[:2]
                optimal_mtx, _ = cv2.getOptimalNewCameraMatrix(
                    self.proj_mtx, self.distortion, (w, h), 1, (w, h)
                )
                self.img = cv2.undistort(
                    self.img, self.proj_mtx, self.distortion, None, optimal_mtx
                )
            return self.img
        except AttributeError:
            return None

    def give_params(self, params):
        ret, mtx, dist, rvecs, tvecs = params

        self.proj_mtx = mtx
        self.distortion = dist

        self.focalx = mtx[0, 0]  # following diagonals to get focal length
        self.focaly = mtx[1, 1]

    def exit(self):
        print("releasing video stream")
        self.capture.release()
