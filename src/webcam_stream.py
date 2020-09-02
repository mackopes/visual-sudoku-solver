import cv2
import threading
import time

from datetime import datetime


class WebcamStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        _, self.frame = self.cap.read()

        self.stopped = False
        self.start_time = None
        self.end_time = None
        self.frames_n = 0

    def read(self):
        self.frames_n += 1
        return self.frame

    def update(self):
        while True:
            if self.stopped:
                return

            _, self.frame = self.cap.read()
            time.sleep(1 / 30)

    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        self.start_time = datetime.now()
        self.end_time = None

    def stop(self):
        self.stopped = True
        self.end_time = datetime.now()
        self.thread.join()

    def fps(self):
        if self.start_time is None:
            raise AttributeError("You need to start and stop the stream first to calculate the fps")

        if self.end_time is None:
            raise AttributeError("You need to stop the stream first to calculate the fps")

        total_seconds = (self.end_time - self.start_time).total_seconds()
        return self.frames_n / total_seconds
