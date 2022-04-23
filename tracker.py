"""
This file is the implementation of IOU Tracker.
"""

import cv2
import numpy as np

from kalman_filter import MyKalmanFilter


class MyTracker:
    """
    Class Variables:
        __capture:      the tracker id
        __thr_iou:      a threshold value for iou matching
        __thr_kf:       a threshold value for Kalman filter matching
        __kfs:          a dictionary, which stores Kalman filters for each detected object in the form of
                        (Kalman filter, lost_counter, total_counter)
        __coordinator:  an id assignment coordinator
        __predictions:  a dictionary, which stores Kalman filter predictions
        __max_frames:   a constant which specifies the maximum frame number during which the
                        Kalman filter keeps tracking

        objects:        a dictionary, which stores information of previously detected objects,
                        in the form of id: (left-top position x, y, width, height, img)
    """
    def __init__(self, capture, coordinator, threshold, max_frames=60):
        self.__capture = capture
        self.__thr_iou = threshold
        self.__thr_kf = threshold * 0.4
        self.__kfs = {}
        self.__coordinator = coordinator
        self.__predictions = {}
        self.__max_frames = max_frames
        self.objects = {}

    def track(self, objs_rect, img):
        """
        Inputs:
            objs_rect:  a list which contains information of detected objects
                        in the form of [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]

        Outputs:
            objs_id:    a list which stores ids and other information of input objects
        """
        objs_id = []
        current_objs = {}
        for x, y, w, h in objs_rect:
            # IOU tracking
            id_t = self.__iou_match(x, y, w, h)
            if id_t == -1:                      # IOU match fails, consult the Kalman filter
                id_t = self.__kf_match(x, y, w, h)
                if id_t == -1:                  # not previously detected, get an id for it!
                    id_t = self.__coordinator.assign_id(self.__capture, x, y, w, h)
                    self.__kf_set(id_t, x, y, w, h)
            # multiple objects with same id! tracker fails, consult the id coordinator!
            if id_t in current_objs.keys():
                area_cur = w * h
                area_other = current_objs[id_t][2] * current_objs[id_t][3]
                # the one with bigger area wins!
                if area_cur > area_other:
                    x_o, y_o = current_objs[id_t][0], current_objs[id_t][1]
                    w_o, h_o = current_objs[id_t][2], current_objs[id_t][3]
                    img_o = current_objs[id_t][4]
                    id_o = self.__coordinator.assign_id(self.__capture, x_o, y_o, w_o, h_o)
                    self.__kf_set(id_o, x, y, w, h)
                    current_objs[id_o] = (x_o, y_o, w_o, h_o, img_o)
                else:
                    id_t = self.__coordinator.assign_id(self.__capture, x, y, w, h)
                    self.__kf_set(id_t, x, y, w, h)
            current_objs[id_t] = (x, y, w, h, np.array(img[y:y + h, x:x + w]))
        self.objects = current_objs.copy()      # update the detected objects
        self.__kf_update_not_detected()         # update the kalman filter by its own prediction if not detected
        for id_t, val in current_objs.items():
            objs_id.append((id_t, val[0], val[1], val[2], val[3]))
        self.__kf_predict()
        return tuple(objs_id)

    def __iou_match(self, x, y, w, h):
        """
        this function calculates IOU scores of each pair of input object and previously detected object,
        it returns object id if there is a match, returns -1 if there is no match
        """
        max_iou = 0.0
        temp_id = -1
        for id_t, pos_t in self.objects.items():
            iou = self.__iou_calculator(x, y, w, h, pos_t[0], pos_t[1], pos_t[2], pos_t[3])
            if iou > max_iou:
                temp_id = id_t
                max_iou = iou
        if max_iou > self.__thr_iou:    # this object has already been detected
            # update the Kalman filter
            self.__kf_update_detected(temp_id, x, y, w, h)
            return temp_id
        else:
            return -1

    def __kf_set(self, id_t, x, y, w, h):
        """
        this function initializes or updates Kalman filter whose corresponding object is detected just now
        """
        if id_t not in self.__kfs.keys():
            # form: [Kalman filter, lost_counter, total_counter]
            self.__kfs[id_t] = [MyKalmanFilter(x, y, w, h), 0, 0]
        else:
            self.__kf_update_detected(id_t, x, y, w, h)

    def __kf_predict(self):
        """
        this functions predicts the trajectories of currently detected objects and store them
        """
        self.__predictions = {}
        for id_t, val in self.__kfs.items():
            kf = val[0]
            self.__predictions[id_t] = kf.predict()

    def __kf_match(self, x, y, w, h):
        """
        this function employs Kalman filter and check whether the current detected object matches any
        prediction from previously detected object
        it returns object id if there is a match, returns -1 if there is no match
        """
        max_iou = 0.0
        temp_id = -1
        for id_t, prediction in self.__predictions.items():
            x_t, y_t, w_t, h_t = prediction[0], prediction[1], prediction[2], prediction[3]
            iou = self.__iou_calculator(x, y, w, h, x_t, y_t, w_t, h_t)
            if iou > max_iou:
                temp_id = id_t
                max_iou = iou
        if max_iou > self.__thr_kf:     # this object matches a prediction from a previously detected object
            # update the Kalman filter
            self.__kf_update_detected(temp_id, x, y, w, h)
            return temp_id
        else:
            return -1

    def __kf_update_detected(self, id_t, x, y, w, h):
        """
        this function updates the Kalman filter with id id_t
        """
        if id_t in self.objects.keys():
            dx = x - self.objects[id_t][0]
            dy = y - self.objects[id_t][1]
            dw = w - self.objects[id_t][2]
            dh = h - self.objects[id_t][3]
        else:
            dx = x - self.__kfs[id_t][0].xp
            dy = y - self.__kfs[id_t][0].yp
            dw = w - self.__kfs[id_t][0].wp
            dh = h - self.__kfs[id_t][0].hp
        if id_t in self.__kfs.keys():
            self.__kfs[id_t][0].update(x, y, w, h, dx, dy, dw, dh)
            self.__kfs[id_t][1] = 0
            self.__kfs[id_t][2] += 1

    def __kf_update_not_detected(self):
        """
        this function only updates Kalman filters whose corresponding object is not currently detected
        """
        delete_list = []
        for id_t in self.__kfs.keys():
            if id_t not in self.objects.keys():     # not detected
                self.__kfs[id_t][1] += 1
                self.__kfs[id_t][2] += 1
                # remove random glitches
                if self.__kfs[id_t][2] < 7:
                    delete_list.append(id_t)
                    continue
                # reached the maximum tracking frames
                if self.__kfs[id_t][1] == self.__max_frames:
                    delete_list.append(id_t)
                    continue
                x, y, w, h, dx, dy, dw, dh = self.__predictions[id_t]
                self.__kfs[id_t][0].update(x, y, w, h, dx, dy, dw, dh)
        # delete unexpected Kalman filters
        for id_t in delete_list:
            self.__kfs.pop(id_t)

    def show_kf(self, img):
        """
        this functions shows the Kalman filter prediction on img
        """
        for id_t, pair in self.__kfs.items():
            kf = pair[0]
            x = int(kf.xp)
            y = int(kf.yp)
            w = int(kf.wp)
            h = int(kf.hp)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, str(id_t), (x + w + 3, int(y + h / 2 + 1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    def show_old(self, img):
        """
        this function shows the previously detected objects
        """
        for id_t, val in self.objects.items():
            x = val[0]
            y = val[1]
            w = val[2]
            h = val[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(img, str(id_t), (x - 25, int(y + h / 2 + 1)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

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
        # if width and height are both positive, then the intersection area is positive
        if wi > 0 and hi > 0:
            overlap = wi * hi
        else:
            overlap = 0.0
        # union area
        union = area1 + area2 - overlap
        if union == 0:
            return 0
        return overlap / union
