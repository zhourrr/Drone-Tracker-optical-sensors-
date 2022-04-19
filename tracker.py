"""
This file is the implementation of IOU Tracker.
"""
import cv2

from kalman_filter import MyKalmanFilter


class MyTracker:
    """
    Class Variables:
        __id_count:     keep the count of the IDs
        __threshold:    a threshold value for iou matching
        __kfs:          a dictionary, which stores kalman filters for each detected object in the form of
                        (Kalman filter, counter)
        __max_frames:   a constant which specifies the maximum frame number during which the
                        Kalman filter keeps tracking

        objects:        a dictionary, which stores information of previously detected objects,
                        in the form of id: (left-top position x, y, width, height)
    """
    def __init__(self, threshold, max_frames=12):
        self.objects = {}
        self.__kfs = {}
        self.__threshold = threshold
        self.__id_count = 0
        self.__max_frames = max_frames

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
            if id_t == -1:                      # IOU match fails, consult the Kalman filter
                id_t = self.__kf_match(x, y, w, h)
                if id_t == -1:                  # new object! assign a new id.
                    self.__id_count += 1
                    id_t = self.__id_count
                    self.__kfs[id_t] = [MyKalmanFilter(x, y, w, h), 0]
            current_objs[id_t] = (x, y, w, h)
            objs_id.append((id_t, x, y, w, h))
        self.objects = current_objs.copy()      # update the detected objects
        self.__kf_update()                      # update the kalman filter by its own prediction if not detected
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
            # update the Kalman filter
            dx = x - self.objects[temp_id][0]
            dy = y - self.objects[temp_id][1]
            dw = w - self.objects[temp_id][2]
            dh = h - self.objects[temp_id][3]
            self.__kfs[temp_id][0].update(x, y, w, h, dx, dy, dw, dh)
            self.__kfs[temp_id][1] = 0
            return temp_id
        else:
            return -1

    def __kf_match(self, x, y, w, h):
        """
        this function employs Kalman filter and check whether the current detected object matches any
        prediction from previously detected object
        it returns object id if there is a match, returns -1 if there is no match
        """
        max_iou = 0.0
        temp_id = -1
        for id_t, pair in self.__kfs.items():
            kf = pair[0]
            prediction = kf.predict()
            iou = self.__iou_calculator(x, y, w, h, prediction[0], prediction[1], prediction[2], prediction[3])
            if iou > max_iou:
                temp_id = id_t
                max_iou = iou
        if max_iou > self.__threshold:  # this object matches a prediction from a previously detected object
            # update the Kalman filter
            dx = x - self.__kfs[temp_id][0].x
            dy = y - self.__kfs[temp_id][0].y
            dw = w - self.__kfs[temp_id][0].w
            dh = h - self.__kfs[temp_id][0].h
            self.__kfs[temp_id][0].update(x, y, w, h, dx, dy, dw, dh)
            self.__kfs[temp_id][1] = 0
            return temp_id
        else:
            return -1

    def __kf_update(self):
        """
        this function only updates Kalman filters whose ids are not currently detected
        """
        delete_list = []
        for id_t in self.__kfs.keys():
            if id_t not in self.objects.keys():
                self.__kfs[id_t][1] += 1
                if self.__kfs[id_t][1] == self.__max_frames:
                    delete_list.append(id_t)
                    continue
                x, y, w, h, dx, dy, dw, dh = self.__kfs[id_t][0].predict()
                self.__kfs[id_t][0].update(x, y, w, h, dx, dy, dw, dh)
        for id_t in delete_list:
            self.__kfs.pop(id_t)

    def show_kf(self, img):
        """
        this functions shows the Kalman filter prediction on img
        """
        for id_t, pair in self.__kfs.items():
            kf = pair[0]
            x, y, w, h, _, _, _, _ = kf.predict()
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

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
        return overlap / union
