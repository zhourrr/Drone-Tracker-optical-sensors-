"""
This file is the implementation of Kalman filter, which is mainly used in the tracker.
"""

import numpy as np
import cv2


class MyKalmanFilter:
    """
    8 states:           (x, y, w, h, dx, dy, dw, dh)
                        x and y are the central coordinates
    self.x:
    self.y:             these are real measurements
    self.w:
    self.h:
    """
    def __init__(self, x, y, w, h):
        self.__kf = cv2.KalmanFilter(8, 8)
        matrix_t = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        matrix_m = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        self.__kf.transitionMatrix = matrix_t
        self.__kf.measurementMatrix = matrix_m
        self.__kf.predict()
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(w)], [np.float32(h)],
                             [np.float32(0)], [np.float32(0)], [np.float32(0)], [np.float32(0)]])
        self.__kf.correct(measured)
        res = self.__kf.predict()
        self.xp = int(res[0])
        self.yp = int(res[1])
        self.wp = int(res[2])
        self.hp = int(res[3])
        self.__kf.correct(measured)

    def update(self, x, y, w, h, dx, dy, dw, dh):
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(w)], [np.float32(h)],
                             [np.float32(dx)], [np.float32(dy)], [np.float32(dw)], [np.float32(dh)]])
        self.__kf.correct(measured)

    def predict(self):
        predicted = self.__kf.predict()
        x_t, y_t, w_t, h_t = int(predicted[0]), int(predicted[1]), int(predicted[2]), int(predicted[3])
        dx_t, dy_t, dw_t, dh_t = int(predicted[4]), int(predicted[5]), int(predicted[6]), int(predicted[7])
        self.xp = x_t
        self.yp = y_t
        self.wp = w_t
        self.hp = h_t
        return x_t, y_t, w_t, h_t, dx_t, dy_t, dw_t, dh_t
