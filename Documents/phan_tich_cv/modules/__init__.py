import os
import cv2
import shutil
import torch
import numpy as np
from ultralytics import YOLO

CACHE_DIR = '.cache'
'''
map_label = {0: '-',
             1: 'avatar',
             2: 'block',
             3: 'info',
             4: 'job_title',
             5: 'name'}
'''

map_label = {0: 'avatar',
             1: 'block',
             2: 'infor',
             3: 'job_title',
             4: 'name'}


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def sort_box(boxes):
    sorted_boxes = []
    for box in boxes:
        sorted_boxes.append(order_points_clockwise(box))
    mid_points = []
    for box in sorted_boxes:
        try:
            mid = line_intersection((box[0], box[2]), (box[1], box[3]))
            mid_points.append(mid)
        except:
            continue
    sorted_indices = np.argsort(mid_points, axis=0)
    sorted_boxes = sorted(sorted_boxes, key=lambda sorted_indices: [sorted_indices[0][1], sorted_indices[0][0]])
    return sorted_boxes


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def crop_box(img, boxes, out_folder, sort=True):
    h, w, c = img.shape

    if sort:
        boxes = sort_box(boxes)

    for i, box in enumerate(boxes):
        box_name = os.path.join(out_folder, f"{i}.png")

        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
        x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
        x1 = max(0, x1)
        x2 = max(0, x2)
        x3 = max(0, x3)
        x4 = max(0, x4)
        y1 = max(0, y1)
        y2 = max(0, y2)
        y3 = max(0, y3)
        y4 = max(0, y4)
        min_x = max(0, min(x1, x2, x3, x4))
        min_y = max(0, min(y1, y2, y3, y4))
        max_x = min(w, max(x1, x2, x3, x4))
        max_y = min(h, max(y1, y2, y3, y4))

        tw = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        th = int(np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2))
        pt1 = np.float32([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
        pt2 = np.float32([[0, 0],
                          [tw - 1, 0],
                          [tw - 1, th - 1],
                          [0, th - 1]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        cropped = cv2.warpPerspective(img, matrix, (tw, th))

        try:
            cv2.imwrite(box_name, cropped)
        except:
            print(box_name, " is missing")

    return boxes


class YOLO_Det:
    def __init__(self, weight_path=None, model_name=None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load model pretrain
        self.model = YOLO(weight_path)
        self.model.to(self.device)

    def __call__(
            self,
            image,
            return_result=False,
            output_path=None):
        """
        Input: path to image
        Output: boxes (coordinates of 4 points)
        """

        # Detect and OCR for final result
        result = self.model(image, conf=0.25, verbose=False, nms=True, iou=0.5)
        boxes_list = result[0].boxes.data[:, :4]
        labels_idx = result[0].boxes.data[:, 5]
        labels = [map_label[int(i)] for i in labels_idx]
        box = result[0].boxes
        conf = box.conf

        # if return_result:
        detect_image = result[0].plot()
        # cv2.imwrite(os.path.join(output_path, 'detect_image.jpg'), detect_image)

        return boxes_list, labels, detect_image, conf