import logging
import threading
import subprocess

import numpy as np
import cv2

def fourcc(a, b, c, d):
    return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)
def pixelformat(string):
    if len(string) != 3 and len(string) != 4:
        msg = "{} is not a pixel format".format(string)
        raise ValueError(msg)
    if len(string) == 3:
        return fourcc(string[0], string[1], string[2], ' ')
    else:
        return fourcc(string[0], string[1], string[2], string[3])

def get_calibration() -> tuple:
    fs = cv2.FileStorage(
        "rectify_ov9281_neg_alpha.yaml", cv2.FILE_STORAGE_READ
    )

    map_l = (fs.getNode("map_l_1").mat(), fs.getNode("map_l_2").mat())

    map_r = (fs.getNode("map_r_1").mat(), fs.getNode("map_r_2").mat())

    fs.release()

    return map_l, map_r
def resize(frame, dst_width):
    width = frame.shape[1]
    height = frame.shape[0]
    scale = dst_width * 1.0 / width
    return cv2.resize(frame, (int(scale * width), int(scale * height)))

def open_ov9281(width, height, sensor_id):
    """Open the Jetson onboard camera."""
    cap = cv2.VideoCapture(sensor_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, pixelformat("GREY"))
    cap.set(cv2.CAP_PROP_CONVERT_RGB,0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, frame = cam.cap.read()
        if frame is None:
            #logging.warning('Camera: cap.read() returns None...')
            break
        frame = resize(frame, 1280.0)
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        img_l = frame[:, :640]
        img_r = frame[:, 640:]
        if cam.recti:
            img_l = cv2.remap(img_l, *cam.map_l, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, *cam.map_r, cv2.INTER_LINEAR)
        cam.img_handle = img_l, img_r


        
    cam.thread_running = False


class Camera():
    def __init__(self, sensor_id, rectified=True, width=2560, height=800):
        self.is_opened = False
        self.thread_running = False
        self.img_handle = None
        self.recti = rectified
        self.img_width = width
        self.img_height = height
        self.cap = None
        self.thread = None
        self.sensor_id = sensor_id
        self.map_l, self.map_r = get_calibration()
        self._open()  # try to open the camera

    def _open(self):
        """Open camera based on command line arguments."""
        if self.cap is not None:
            raise RuntimeError('camera is already opened!')

        logging.info('Camera: using onboard camera')
        self.cap = open_ov9281(self.img_width, self.img_height, self.sensor_id)
        self._start()

    def isOpened(self):
        return self.is_opened

    def _start(self):
        if not self.cap.isOpened():
            logging.warning('Camera: starting while cap is not opened!')
            return

        # Try to grab the 1st image and determine width and height
        _, self.img_handle = self.cap.read()
        if self.img_handle is None:
            logging.warning('Camera: cap.read() returns no image!')
            self.is_opened = False
            return

        self.is_opened = True

        self.img_height, self.img_width = self.img_handle.shape
        # start the child thread if not using a video file source
        # i.e. rtsp, usb or onboard
        assert not self.thread_running
        self.thread_running = True
        self.thread = threading.Thread(target=grab_img, args=(self,))
        self.thread.setDaemon(True)
        self.thread.start()

    def _stop(self):
        if self.thread_running:
            self.thread_running = False
            #self.thread.join()

    def read(self):
        """Read a frame from the camera object.

        Returns None if the camera runs out of image or error.
        """
        if not self.is_opened:
            return None

        return self.img_handle

    def release(self):
        self._stop()
        try:
            self.cap.release()
        except:
            pass
        self.is_opened = False

    def __del__(self):
        self.release()
