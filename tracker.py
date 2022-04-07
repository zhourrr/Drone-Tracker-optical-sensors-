"""
This file is the implementation of IOU Tracker.
"""

import math


class TrackerIOU:
    """
    Class Variables:
        __id_count:     keep the count of the IDs
        __threshold:    a threshold value for iou matching

        objects:        a dictionary, which stores information of detected objects,
                        in the form of id: (left-top position x, y, width, height)
    """
    def __init__(self, threshold):
        self.objects = {}
        self.__threshold = threshold
        self.__id_count = 0

    def track(self, objs_rect):
        """
        Inputs:
            objs_rect:  a list which contains information of detected objects
                        in the form of [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]

        Outputs:
            objs_id:    a list which stores ids and other information of input objects
        """
        current_objs = {}
        objs_id = []
        for x, y, w, h in objs_rect:
            # find out if that object was detected already
            id_t = self.__iou_match(x, y, w, h)
            if id_t == -1:                      # new object, assign a new id to it
                self.__id_count += 1
                id_t = self.__id_count
            current_objs[id_t] = (x, y, w, h)
            objs_id.append((id_t, x, y, w, h))
        self.objects = current_objs.copy()      # update the detected objects
        return objs_id

    def __iou_match(self, x, y, w, h):
        """
        this function calculates IOU scores of input object and each already detected object,
        it returns object id if there is a match, returns -1 if there is no match
        """
        max_iou = 0.0
        temp_id = -1
        for id_t, pos_t in self.objects.items():
            iou = self.__iou_calculator(x, y, w, h, pos_t[0], pos_t[1], pos_t[2], pos_t[3])
            if iou > max_iou:
                temp_id = id_t
                max_iou = iou
        if max_iou > self.__threshold:  # this object has already been detected
            return temp_id
        else:
            return -1

    @staticmethod
    def __iou_calculator(x1, y1, w1, h1, x2, y2, w2, h2):
        """
        returns iou fraction.
        """
        area1 = w1 * h1
        area2 = w2 * h2
        # intersection coordinates
        xi_lt = max(x1, x2)
        yi_lt = max(y1, y2)
        xi_rb = min(x1 + w1, x2 + w2)
        yi_rb = min(y1 + h1, y2 + h2)
        wi = xi_rb - xi_lt
        hi = yi_rb - yi_lt
        # if width and height are both positive, then the intersection area is positve
        if wi > 0 and hi > 0:
            overlap = wi * hi
        else:
            overlap = 0.0
        # union area
        union = area1 + area2 - overlap
        return overlap / union





