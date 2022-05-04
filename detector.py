import cv2
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
        cameras:    a list, which contains camera objects
        width:      frame width
        height:     frame height
    Class Variables:
        __captures:     a list, which stores VideoCapture for each camera
        __cameras:      a list, which stores Camera objects
        __roi_info:     a list, which stores roi coordinates for each camera
        __mask:         a list, which stores processed rois
        __background:   a list, which stores the background scene of each camera
        __bs:           a list, which stores opencv background subtractor for each camera
        __cur_objects:  a list, which stores currently detected objects
        __wT:           wait time
        __area_max:     the largest area of a detected contour
        __area_min:     the smallest area of a detected contour
        __similarity:   a constant threshold for similarity comparison
        __flag_s:       a flag variable for video pause and continue
        num_cameras:    the number of cameras
        roi:            a list, which stores roi of each frame from each camera
        trackers:       a list, which contains tracking units for each camera
    """

    def __init__(self, captures=None, roi=None, wt=15, area_min=50, area_max=50000,
                 cameras=None, width=1920, height=1080):
        # video captures
        if captures is None:
            # the default setting is the internal camera only, however, the file descriptor for the internal
            # camera in different operating systems might differ
            self.__captures = [cv2.VideoCapture(0)]
        else:
            self.__captures = []
            for num in range(len(captures)):
                self.__captures.append(cv2.VideoCapture(captures[num]))
                self.__captures[num].set(3, width)
                self.__captures[num].set(4, height)
                # self.__captures[num].set(5, 30)
        self.num_cameras = len(self.__captures)     # record the total number of cameras
        self.__cameras = []
        self.__roi_info = []
        self.__background = []
        self.__bs = []
        self.__mask = [None] * self.num_cameras
        self.__cur_objects = []
        for i in range(self.num_cameras):
            # instantiate cameras
            if cameras is None:
                self.__cameras.append(Camera())
            else:
                self.__cameras.append(cameras[i])
            # roi
            if roi is None:
                self.__roi_info.append(None)
            else:
                self.__roi_info.append(roi[i])
            # opencv background subtractor
            self.__bs.append(cv2.createBackgroundSubtractorMOG2(history=3000, varThreshold=12, detectShadows=True))
            # one detected object list for each camera
            self.__cur_objects.append([])
        self.__wT = wt
        self.__area_max = area_max
        self.__area_min = area_min
        self.__similarity = 0.35
        self.__flag_s = False
        self.roi = [None] * self.num_cameras
        self.trackers = []

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

    def init_background(self):
        """
        use the initial frames as the backgrounds
        """
        for num in range(150):   # use the initial 150 frames as background scenes
            for capture in range(self.num_cameras):
                ret, temp_frame = self.__captures[capture].read()
                self.__bs[capture].apply(temp_frame)
        for capture in range(self.num_cameras):
            self.__background.append(self.__bs[capture].getBackgroundImage())

    def __update_background(self):
        """
        updates background images gradually and regularly
        """
        for capture in range(self.num_cameras):
            self.__background[capture] = self.__bs[capture].getBackgroundImage()
        return

    def __remove_background(self):
        """
        process rois and remove backgrounds
        """
        for capture in range(self.num_cameras):
            # background subtraction
            d_frame = self.__bs[capture].apply(self.roi[capture])
            # blur the image, remove noises, parameters: kernel size, standard deviation
            b_frame = cv2.GaussianBlur(d_frame, (9, 9), 0)
            # thresholding
            _, t_frame = cv2.threshold(b_frame, 200, 255, cv2.THRESH_BINARY)
            self.__mask[capture] = t_frame

    def __rgb_threshold(self, img, bg):
        """
        not used!!!
        this function thresholds an RGB image and returns the resultant image
        """
        diff_img = cv2.absdiff(img, bg)
        diff_img = cv2.GaussianBlur(diff_img, (9, 9), 0)
        _, t_image_1 = cv2.threshold(diff_img[:, :, 0], 30, 255, cv2.THRESH_BINARY)
        _, t_image_2 = cv2.threshold(diff_img[:, :, 1], 30, 255, cv2.THRESH_BINARY)
        _, t_image_3 = cv2.threshold(diff_img[:, :, 2], 30, 255, cv2.THRESH_BINARY)
        temp = cv2.bitwise_or(t_image_1, t_image_2)
        temp = cv2.bitwise_or(temp, t_image_3)
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        diff_img = cv2.absdiff(g_img, g_bg)
        diff_img = cv2.GaussianBlur(diff_img, (9, 9), 0)
        _, t_img = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)
        res = cv2.bitwise_or(temp, t_img)
        return res

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
                    # shape detection, remove irregular objects
                    if w > 5 * h or h > 5 * w:
                        continue
                    self.__cur_objects[capture].append((x, y, w, h))

    def __track(self):
        """
        track detected objects and store their trajectories
        """
        objects = []
        for capture in range(self.num_cameras):
            # display previously detected objects
            # self.trackers[capture].show_old(self.roi[capture])
            objects_id = self.trackers[capture].track(self.__cur_objects[capture], self.roi[capture])
            for id_t, x_t, y_t, w_t, h_t in objects_id:
                cv2.rectangle(self.roi[capture], (x_t, y_t), (x_t + w_t, y_t + h_t), (0, 255, 0), 3)
                cv2.putText(self.roi[capture], str(id_t), (int(x_t + w_t / 2 - 3), y_t - 15), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)
            objects.append(objects_id)
            # display Kalman filter predictions
            # self.trackers[capture].show_kf(self.roi[capture])
        return tuple(objects)

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
        objects = self.__track()
        # update background scenes
        self.__update_background()
        return objects

    def __show(self):
        """
        displays real-time videos with detected objects
        """
        for camera in range(self.num_cameras):
            cv2.imshow("camera " + str(camera + 1), self.roi[camera])
            cv2.imshow("mask " + str(camera + 1), self.__mask[camera])

    def tracker_init(self, coordinator):
        """
        this function instantiates and initializes trackers for each camera
        """
        for i in range(self.num_cameras):
            self.trackers.append(MyTracker(i, coordinator, 0.45))

    def detect(self):
        """
        an interface for users -- a wrapper function
        """
        res = self.__read()
        if not res:
            cv2.destroyAllWindows()
            for capture in range(self.num_cameras):
                self.__captures[capture].release()
            return False
        self.__show()
        # keyboard interrupt
        if self.__flag_s:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(self.__wT)
        if key == 27:           # press Esc to exit
            cv2.destroyAllWindows()
            for capture in range(self.num_cameras):
                self.__captures[capture].release()
            return False
        elif key == 112:        # press p to pause
            cv2.waitKey(0)      # press any key to continue
        elif key == 115:
            self.__flag_s = not self.__flag_s
        return res
