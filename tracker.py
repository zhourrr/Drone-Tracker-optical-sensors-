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
                        (Kalman filter, lost_counter, total_counter)
        __max_frames:   a constant which specifies the maximum frame number during which the
                        Kalman filter keeps tracking

        objects:        a dictionary, which stores information of previously detected objects,
                        in the form of id: (left-top position x, y, width, height)
    """
    def __init__(self, threshold, max_frames=60):
        self.objects = {}
        self.__kfs = {}
        self.__threshold = threshold
        self.__thr_kf = threshold * 0.7
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
        objs_id = []
        current_objs = {}
        predictions = {}
        contained_list = []
        # let Kalman filters predict the trajectories
        for id_t, pair in self.__kfs.items():
            kf = pair[0]
            predictions[id_t] = kf.predict()
        for x, y, w, h in objs_rect:
            # find out if the current object has been detected already
            id_t = self.__iou_match(x, y, w, h, contained_list)
            if id_t == -1:                      # IOU match fails, consult the Kalman filter
                id_t = self.__kf_match(x, y, w, h, predictions, contained_list)
                if id_t == -1:                  # new object! assign a new id.
                    self.__id_count += 1
                    id_t = self.__id_count
                    self.__kfs[id_t] = [MyKalmanFilter(x, y, w, h), 0, 0]
            if id_t in current_objs.keys():     # multiple objects with same id!
                # merge objects!
                x, y, w, h = self.__merge(current_objs, id_t, x, y, w, h)
                contained_list.append(id_t)
            current_objs[id_t] = (x, y, w, h)
        contained_list = set(contained_list)
        self.__kf_update_contained(contained_list, current_objs)   # update the Kalman filter of contained objects
        self.objects = current_objs.copy()              # update the detected objects
        self.__kf_update_not_detected(predictions)      # update the kalman filter by its own prediction if not detected
        for id_t, val in current_objs.items():
            objs_id.append((id_t, val[0], val[1], val[2], val[3]))
        return tuple(objs_id)

    def __iou_match(self, x, y, w, h, contained_lis):
        """
        this function calculates IOU scores of input object and each already detected object,
        it returns object id if there is a match, returns -1 if there is no match
        """
        max_iou = 0.0
        temp_id = -1
        for id_t, pos_t in self.objects.items():
            if self.__contain(x, y, w, h, pos_t[0], pos_t[1], pos_t[2], pos_t[3], contained_lis, id_t):
                return id_t
            iou = self.__iou_calculator(x, y, w, h, pos_t[0], pos_t[1], pos_t[2], pos_t[3])
            if iou > max_iou:
                temp_id = id_t
                max_iou = iou
        if max_iou > self.__threshold:  # this object has already been detected
            # update the Kalman filter
            self.__kf_update_detected(temp_id, x, y, w, h)
            return temp_id
        else:
            return -1

    def __kf_match(self, x, y, w, h, predictions, contained_lis):
        """
        this function employs Kalman filter and check whether the current detected object matches any
        prediction from previously detected object
        it returns object id if there is a match, returns -1 if there is no match
        """
        max_iou = 0.0
        temp_id = -1
        for id_t, prediction in predictions.items():
            x_t, y_t, w_t, h_t = prediction[0], prediction[1], prediction[2], prediction[3]
            if self.__contain(x, y, w, h, x_t, y_t, w_t, h_t, contained_lis, id_t):
                return id_t
            iou = self.__iou_calculator(x, y, w, h, x_t, y_t, w_t, h_t)
            if iou > max_iou:
                temp_id = id_t
                max_iou = iou
        if max_iou > self.__thr_kf:           # this object matches a prediction from a previously detected object
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
        self.__kfs[id_t][0].update(x, y, w, h, dx, dy, dw, dh)
        self.__kfs[id_t][1] = 0
        self.__kfs[id_t][2] += 1

    def __kf_update_not_detected(self, predictions):
        """
        this function only updates Kalman filters whose ids are not currently detected
        """
        delete_list = []
        for id_t in self.__kfs.keys():
            if id_t not in self.objects.keys():                 # not detected
                self.__kfs[id_t][1] += 1
                self.__kfs[id_t][2] += 1
                # random glitch
                if self.__kfs[id_t][2] < 3:
                    delete_list.append(id_t)
                    continue
                # reached the maximum tracking frames
                if self.__kfs[id_t][1] == self.__max_frames:
                    delete_list.append(id_t)
                    continue
                x, y, w, h, dx, dy, dw, dh = predictions[id_t]
                self.__kfs[id_t][0].update(x, y, w, h, dx, dy, dw, dh)
        for id_t in delete_list:
            self.__kfs.pop(id_t)

    def __kf_update_contained(self, contained_lis, dic):
        for id_t in contained_lis:
            x, y, w, h = dic[id_t]
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
            self.__kfs[id_t][0].update(x, y, w, h, dx, dy, dw, dh)
            self.__kfs[id_t][1] = 0
            self.__kfs[id_t][2] += 1

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
            cv2.rectangle(img, (x + 5, y + 5), (x + w - 5, y + h - 5), (0, 0, 255), 3)
            cv2.putText(img, str(id_t), (x + 15, y + h + 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

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
            cv2.putText(img, str(id_t), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    @staticmethod
    def __merge(dic, id_t, x, y, w, h):
        """
        this function merges two objects with same id
        """
        x1, y1, w1, h1 = dic[id_t]
        x_t = min(x1, x)
        y_t = min(y1, y)
        rb_x = max(x1 + w1, x + w)
        rb_y = max(y1 + h1, y + h)
        w_t = rb_x - x_t
        h_t = rb_y - y_t
        return x_t, y_t, w_t, h_t

    @staticmethod
    def __contain(x1, y1, w1, h1, x2, y2, w2, h2, contained_lis, id_t):
        """
        returns true if rectangle1 is contained in the rectangle2, or vice versa.
        returns false otherwise.
        """
        margin_of_error = 30
        if (x1 <= (x2 + margin_of_error) and (x1 + w1 + margin_of_error) >= (x2 + w2) and
                y1 <= (y2 + margin_of_error) and (y1 + h1 + margin_of_error) >= (y2 + h2)):
            contained_lis.append(id_t)
            return True
        elif (x2 <= (x1 + margin_of_error) and (x2 + w2 + margin_of_error) >= (x1 + w1) and
                y2 <= (y1 + margin_of_error) and (y2 + h2 + margin_of_error) >= (y1 + h1)):
            contained_lis.append(id_t)
            return True
        return False

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
