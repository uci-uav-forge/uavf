### Utilities for streaming video
from threading import Thread
import cv2, atexit, time

# Camera object. Grabs frames from camera in a new thread
# to avoid blocking main thread
class VideoStreamCV(object):
    def __init__(self, src=0, ready_timeout=10):
        self.ready_timeout = ready_timeout
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
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

    def get_img(self):
        try:
            return self.img
        except AttributeError:
            return None

    def exit(self):
        print("releasing video stream")
        self.capture.release()
