import cv2
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture(0)
cap_2 = cv2.VideoCapture(1)
# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(varThreshold=80)
object_detector_2 = cv2.createBackgroundSubtractorMOG2(varThreshold=80)

while True:
    ret, frame = cap.read()
    ret_2, frame_2 = cap_2.read()

    height, width, _ = frame.shape
    height_2, width_2, _ = frame_2.shape

    # Extract Region of interest
    #roi = frame[340: 720,500: 800]
    roi = frame
    roi_2 = frame_2

    # 1. Object Detection
    mask = object_detector.apply(roi)
    mask_2 = object_detector_2.apply(roi_2)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    _, mask_2 = cv2.threshold(mask_2, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_, _ = cv2.findContours(mask_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    detections_2 = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 300 and area < 15000:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])
    for cnt_2 in contours_2:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt_2)
        if area > 300 and area < 15000:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt_2)

            detections_2.append([x, y, w, h])
    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    boxes_ids_2 = tracker.update(detections_2)
    for box_id_2 in boxes_ids_2:
        x, y, w, h, id = box_id_2
        cv2.putText(roi_2, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi_2, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi_2", roi_2)
    cv2.imshow("Frame_2", frame_2)
    cv2.imshow("Mask_2", mask_2)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()