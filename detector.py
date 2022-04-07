"""
This file is the implementation of object detector
"""

import cv2
import numpy as np


class MyDetector:
    """
    Inputs:
        cameras:    a list, which contains the ids of cameras
                        example:    two third-party cameras, then cameras = [1, 2]
        roi:        a list, which contains the pixel coordinates for each camera
                        example:    we want [340:720, 500:800] for camera 1, and
                                    we want [300:700, 550:850] for camera 2, then
                                    roi = [[340, 720, 500, 800], [300, 700, 550, 850]].
        waitTime:  a number in the unit of millisecond

    Class Variables:
        __cameras:      a list, which stores VideoCapture for each camera
        __roi_info:     a list, which stores roi coordinates for each camera
        __mask:         a list, which stores processed rois
        __background:   a list, which stores the background scene of each camera
        __candidates:   a list, which stores candidate frames for updating background scenes
        __bg_counter:   an integer counter for updating background scenes
        __wT:           wait time
        __area_max:     the largest area of a detected contour
        __area_min:     the smallest area of a detected contour

        num_cameras:    the number of cameras
        roi:            a list, which stores roi of each frame from each camera
    """
    def __init__(self, cameras=None, roi=None, wt=30, area_min=700, area_max=15000):
        if cameras is None:
            self.__cameras = [cv2.VideoCapture(0)]        # the default setting is the internal camera only
        else:
            self.__cameras = []
            for camera in cameras:
                self.__cameras.append(cv2.VideoCapture(camera))
        self.num_cameras = len(self.__cameras)
        if roi is None:
            self.__roi_info = []
            for i in range(self.num_cameras):
                self.__roi_info.append(None)
        else:
            self.__roi_info = roi
        self.__mask = [None] * self.num_cameras
        self.__background = [None] * self.num_cameras
        self.__candidates = []
        for i in range(self.num_cameras):
            self.__candidates.append([])
        self.__bg_counter = 0
        self.__wT = wt
        self.__area_max = area_max
        self.__area_min = area_min
        self.roi = [None] * self.num_cameras

    def __get_roi(self, frame, camera_idx):
        """
        returns the region of interest of video with index camera_idx
        """
        if self.__roi_info[camera_idx] is None:
            return frame
        left_top = self.__roi_info[camera_idx][0]
        right_top = self.__roi_info[camera_idx][1]
        left_bottom = self.__roi_info[camera_idx][2]
        right_bottom = self.__roi_info[camera_idx][3]
        return frame[left_top:right_top, left_bottom:right_bottom]

    def __init_background(self):
        """
        use the initial frames as the backgrounds
        """
        init_frames = [] * self.num_cameras
        for i in range(self.num_cameras):
            init_frames.append([])
        for num in range(30):       # use the first 30 frames as background
            for camera in range(self.num_cameras):
                _, temp_frame = self.__cameras[camera].read()
                init_frames[camera].append(self.__get_roi(temp_frame, camera))
        for camera in range(self.num_cameras):
            self.__background[camera] = np.median(init_frames[camera], axis=0).astype(dtype=np.uint8)
            self.__background[camera] = cv2.cvtColor(self.__background[camera], cv2.COLOR_BGR2GRAY)

    def __update_background(self):
        """
        add current frames to the candidate list, if background counter has reached its
        upper limit, i.e., the number of candidates is enough, updates the background
        """
        for camera in range(self.num_cameras):
            self.__candidates[camera].append(self.roi[camera])      # add the current roi to the candidate list
        self.__bg_counter += 1
        if self.__bg_counter == 30:                                 # reached the upper limit
            for camera in range(self.num_cameras):
                self.__background[camera] = np.median(self.__candidates[camera], axis=0).astype(dtype=np.uint8)
                self.__background[camera] = cv2.cvtColor(self.__background[camera], cv2.COLOR_BGR2GRAY)
                self.__candidates[camera] = []
            self.__bg_counter = 0

    def __read(self):
        """
        read rois from current frames from cameras, store
        them into class variables frames and roi, then
        apply background subtraction
        """
        # read frames
        for camera in range(self.num_cameras):
            _, temp_frame = self.__cameras[camera].read()
            self.roi[camera] = self.__get_roi(temp_frame, camera)
        # remove background and draw bounding box
        self.__remove_background()

    def __remove_background(self):
        """
        process rois, remove backgrounds, and draw bounding boxes
        """
        for camera in range(self.num_cameras):
            g_frame = cv2.cvtColor(self.roi[camera], cv2.COLOR_BGR2GRAY)    # convert current frame to grayscale
            d_frame = cv2.absdiff(g_frame, self.__background[camera])       # remove background
            # blur the image, remove noises, parameters: kernel size, standard deviation
            b_frame = cv2.GaussianBlur(d_frame, (11, 11), 0)
            # thresholding
            _, t_frame = cv2.threshold(b_frame, 30, 255, cv2.THRESH_BINARY)
            self.__mask[camera] = t_frame
            # find contours
            contours, _ = cv2.findContours(t_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            """
            for debugging: this line of code draws contours   
            cv2.drawContours(self.roi[camera], contours, cv2.FILLED, (0, 0, 255))
            """
            # Calculate area and remove small elements
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.__area_min < area < self.__area_max:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(self.roi[camera], (x, y), (x + w, y + h), (0, 255, 0), 3)

    def show(self):
        """
        displays real-time videos with detected objects
        """
        for camera in range(self.num_cameras):
            cv2.imshow("camera " + str(camera + 1), self.roi[camera])
            cv2.imshow("mask " + str(camera + 1), self.__mask[camera])


    def detect(self):
        """
        an interface for users -- a warp function
        """
        self.__init_background()
        while True:
            self.__read()
            #TODO: update the background periodically
            #self.__update_background()
            self.show()
            key = cv2.waitKey(self.__wT)
            if key == 27:
                cv2.destroyAllWindows()
                break



#myins = MyDetector(cameras=[0], roi=[[100, 200, 100, 200]])
myins = MyDetector(cameras=["test.mp4"])
myins.detect()

