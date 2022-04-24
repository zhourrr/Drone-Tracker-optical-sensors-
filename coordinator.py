"""
This file is the implementation of ID-assignment coordinator.

in detector initialization, include a coordinator.
in the initialization of each tracker, include a coordinator (pointer), points to the detector's coordinator
"""

import cv2


class Coordinator:
    """
    Inputs:
        detector:           a detector object

    Class variables:
        __detector:         a detector object
        __num_trackers:     the number of trackers
        __counter:          an id counter, increment when a new object is detected
        __similarity:       a threshold for similarity comparison
        __y_margin:         a threshold for y-coordinate comparison
    """
    def __init__(self, detector):
        self.__detector = detector
        self.__num_trackers = self.__detector.num_cameras
        self.__counter = 0
        self.__similarity = 0.6
        self.__y_margin = 35

    def assign_id(self, capture, x, y, w, h):
        """
        this function takes as input a detected object (its coordinates and its tracker number)
        returns its id
        """
        img = self.__detector.roi[capture][y:y+h, x:x+w]
        for i in range(self.__num_trackers):
            if i == capture:
                continue
            candidates = self.__detector.trackers[i].objects
            for id_t, val in candidates.items():
                x_t, y_t, w_t, h_t = val[0], val[1], val[2], val[3]
                if self.__y_compare(y, h, y_t, h_t):
                    img_temp = val[4]
                    if self.__similarity_compare(img, img_temp):
                        return id_t
        self.__counter += 1
        return self.__counter

    def __y_compare(self, y1, h1, y2, h2):
        """
        returns True if the two y-coordinates are equal within an acceptable margin of error
        """
        center_y1 = int(y1 + h1 / 2)
        center_y2 = int(y2 + h2 / 2)
        if abs(center_y1 - center_y2) <= self.__y_margin:
            return True
        else:
            return False

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
