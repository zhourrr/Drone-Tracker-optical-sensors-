"""
This file is the implementation of object detector
"""

import cv2
import numpy as np
from tracker import MyTracker
from posi import *


class MyDetector:
    """
    Inputs:
        captures:   a list, which contains the ids of camera captures
                        example:    two third-party cameras, then cameras = [1, 2]
        roi:        a list, which contains the pixel coordinates for each camera
                        example:    we want [340:720, 500:800] for camera 1, and
                                    we want [300:700, 550:850] for camera 2, then
                                    roi = [[340, 720, 500, 800], [300, 700, 550, 850]].
        waitTime:   a number in the unit of millisecond
        area_min:   the smallest area of a detected contour
        area_max:   the largest area of a detected contour
        thr_d:      the threshold for background subtraction
        thr_t:      the threshold for IOU tracker
        thr_s:      the threshold for similarity comparison
        cameras:    a list, which contains camera objects

    Class Variables:
        __captures:     a list, which stores VideoCapture for each camera
        __cameras:      a list, which stores Camera objects
        __roi_info:     a list, which stores roi coordinates for each camera
        __mask:         a list, which stores processed rois
        __background:   a list, which stores the background scene of each camera
        __candidates:   a list, which stores candidate frames for updating background scenes
        __bg_counter:   an integer counter for updating background scenes
        __wT:           wait time
        __area_max:     the largest area of a detected contour
        __area_min:     the smallest area of a detected contour
        __threshold:    the threshold for background subtraction
        __trackers:     a list, which contains tracking units for each camera
        __cur_objects:  a list, which stores currently detected objects
        __similarity:   a constant threshold for similarity comparison

        num_cameras:    the number of cameras
        roi:            a list, which stores roi of each frame from each camera
        trajectory:     a dictionary, which stores trajectories of detected objects
    """

    def __init__(self, captures=None, roi=None, wt=15, area_min=300, area_max=50000,
                 thr_d=35, thr_t=0.4, thr_s=0.45, cameras=None):
        # captures
        if captures is None:
            self.__captures = [cv2.VideoCapture(0)]  # the default setting is the internal camera only
        else:
            self.__captures = []
            for capture in captures:
                self.__captures.append(cv2.VideoCapture(capture))
        self.num_cameras = len(self.__captures)
        # cameras
        self.__cameras = []
        if cameras is None:
            for i in range(self.num_cameras):
                self.__cameras.append(Camera())
        else:
            for i in range(self.num_cameras):
                self.__cameras.append(cameras[i])
        # roi
        if roi is None:
            self.__roi_info = []
            for i in range(self.num_cameras):
                self.__roi_info.append(None)
        else:
            self.__roi_info = roi
        self.__mask = [None] * self.num_cameras
        self.__background = []
        self.__candidates = []
        for i in range(self.num_cameras):
            self.__candidates.append([])
        self.__bg_counter = 0
        self.__wT = wt
        self.__area_max = area_max
        self.__area_min = area_min
        self.__threshold = thr_d
        self.__trackers = []
        for i in range(self.num_cameras):
            self.__trackers.append(MyTracker(thr_t))
        self.__cur_objects = []
        for i in range(self.num_cameras):
            self.__cur_objects.append([])
        self.__similarity = thr_s
        self.roi = [None] * self.num_cameras
        self.trajectory = []

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
        for num in range(30):  # use the first 30 frames as background
            for capture in range(self.num_cameras):
                ret, temp_frame = self.__captures[capture].read()
                init_frames[capture].append(self.__get_roi(temp_frame, capture))
        for capture in range(self.num_cameras):
            self.__background.append(np.median(init_frames[capture], axis=0).astype(dtype=np.uint8))

    def __update_background(self):
        """
        add current frames to the candidate list, if background counter has reached its
        upper limit, i.e., the number of candidates is enough, updates the background
        """
        for capture in range(self.num_cameras):
            self.__candidates[capture].append(self.roi[capture])  # add the current roi to the candidate list
        self.__bg_counter += 1
        if self.__bg_counter == 30:  # reached the upper limit
            for capture in range(self.num_cameras):
                self.__background[capture] = np.median(self.__candidates[capture], axis=0).astype(dtype=np.uint8)
                self.__candidates[capture] = []
            self.__bg_counter = 0

    def __read(self):
        """
        read rois from current frames from cameras, store
        them into class variables frames and roi, then
        apply background subtraction
        """
        # read frames
        for capture in range(self.num_cameras):
            ret, temp_frame = self.__captures[capture].read()
            if not ret:
                return ret
            self.roi[capture] = self.__get_roi(temp_frame, capture)
        # remove background and get the masks
        self.__remove_background()
        # find contours
        self.__get_contours()
        # consult the tracking unit
        self.__track()
        return True

    def __remove_background(self):
        """
        process rois and remove backgrounds
        """
        for capture in range(self.num_cameras):
            g_background = cv2.cvtColor(self.__background[capture], cv2.COLOR_BGR2GRAY)
            g_frame = cv2.cvtColor(self.roi[capture], cv2.COLOR_BGR2GRAY)  # convert current frame to grayscale
            d_frame = cv2.absdiff(g_frame, g_background)  # remove background
            # blur the image, remove noises, parameters: kernel size, standard deviation
            b_frame = cv2.GaussianBlur(d_frame, (5, 5), 0)
            # thresholding
            _, t_frame = cv2.threshold(b_frame, self.__threshold, 255, cv2.THRESH_BINARY)
            # morphological processing
            kernel = np.ones((17, 17), np.uint8)
            t_frame = cv2.morphologyEx(t_frame, cv2.MORPH_CLOSE, kernel)
            self.__mask[capture] = t_frame

    def __get_contours(self):
        """
        find contours on the mask images
        """
        for capture in range(self.num_cameras):
            self.__cur_objects[capture] = []
            contours, _ = cv2.findContours(self.__mask[capture], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # for debugging: this line of code draws contours
            # self.roi[capture] = np.zeros_like(self.__mask[capture], np.uint8)
            # cv2.drawContours(self.roi[capture], contours, cv2.FILLED, (255, 255, 255), 5)
            for contour in contours:
                area = cv2.contourArea(contour)
                # remove small or large objects
                if self.__area_min < area < self.__area_max:
                    x, y, w, h = cv2.boundingRect(contour)
                    obj = self.roi[capture][y:y + h, x:x + w]
                    bg = self.__background[capture][y:y + h, x:x + w]
                    # check if this object is similar to the background, reduce the impact of shaking
                    if self.__similarity_compare(obj, bg):
                        continue
                    self.__cur_objects[capture].append((x, y, w, h))
                    cv2.rectangle(self.roi[capture], (x, y), (x + w, y + h), (0, 255, 0), 3)

    def __track(self):
        """
        track detected objects and store their trajectories
        """
        # for now, track single object only
        for capture in range(self.num_cameras):
            ids = self.__trackers[capture].track(self.__cur_objects[capture])
            for id_t, x_t, y_t, w_t, h_t in ids:
                cv2.putText(self.roi[capture], str(id_t), (x_t, y_t + h_t + 15), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 0), 2)

        if len(self.__cur_objects[0]) != 1 or len(self.__cur_objects[1]) != 1:
            pass
        else:
            x0 = self.__cur_objects[0][0][0] + self.__cur_objects[0][0][2] / 2
            y0 = self.__cur_objects[0][0][1] + self.__cur_objects[0][0][3] / 2
            x1 = self.__cur_objects[1][0][0] + self.__cur_objects[1][0][2] / 2
            y1 = self.__cur_objects[1][0][1] + self.__cur_objects[1][0][3] / 2
            m = Model([self.__cameras[0], self.__cameras[1]])
            pos_3d = m.get_coord_basic([(x0, y0), (x1, y1)])
            if pos_3d[2] >= 0:
                print("position error")
            else:
                self.trajectory.append(pos_3d)

    def __similarity_compare(self, img1, img2):
        """
        return True if the two images are similar, return False otherwise.
        """
        # calculate histograms
        hist1b = cv2.calcHist([img1], [0], None, [256], [0, 255])
        hist1g = cv2.calcHist([img1], [1], None, [256], [0, 255])
        hist1r = cv2.calcHist([img1], [2], None, [256], [0, 255])
        hist2b = cv2.calcHist([img2], [0], None, [256], [0, 255])
        hist2g = cv2.calcHist([img2], [1], None, [256], [0, 255])
        hist2r = cv2.calcHist([img2], [2], None, [256], [0, 255])
        # histogram matching
        match_b = cv2.compareHist(hist1r, hist2r, cv2.HISTCMP_BHATTACHARYYA)
        match_g = cv2.compareHist(hist1g, hist2g, cv2.HISTCMP_BHATTACHARYYA)
        match_r = cv2.compareHist(hist1b, hist2b, cv2.HISTCMP_BHATTACHARYYA)
        res = (match_b + match_g + match_r) / 3.0
        # histogram comparison
        if res <= self.__similarity:
            return True
        else:
            return False

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
        flag_s = False
        while True:
            if not self.__read():
                cv2.destroyAllWindows()
                break
            # TODO: update the background periodically
            # self.__update_background()
            self.show()
            if flag_s:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(self.__wT)
            if key == 27:  # press Esc to exit
                cv2.destroyAllWindows()
                break
            elif key == 112:        # press p to pause
                cv2.waitKey(0)      # press any key to continue
            elif key == 115:
                flag_s = not flag_s
