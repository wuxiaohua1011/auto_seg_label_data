import cv2
from pathlib import Path
from typing import List
import os
import numpy as np
import numpy.ma as ma
from pydantic import BaseModel
import csv
import pandas as pd

CAR_COLOR = [142, 0, 0]


def non_max_suppression_fast(boxes, overlapThresh):
    """
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    :param boxes:
    :param overlapThresh:
    :return:
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def find_img_paths(dir_path: Path) -> List[Path]:
    paths: List[Path] = sorted(dir_path.iterdir(), key=os.path.getmtime)
    return paths


def find_bbox_coord(ss_img, min_bb_size=150):
    result = []
    mask = (ss_img == CAR_COLOR).all(-1)
    ss_img[mask] = [255, 255, 255]
    ss_img[~mask] = [0, 0, 0]

    im_bw = cv2.cvtColor(ss_img, cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_bb_size:
            cv2.rectangle(ss_img, (int(x), int(y)), (int(x + w), int(y + h)), color=[255, 0, 0], thickness=2)
            result.append([x, y, w, h])
    result = np.array(result)
    result = non_max_suppression_fast(result, overlapThresh=0.1)
    return result


def draw_bbox(rgb_img, bboxes):
    for bbox in bboxes:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(rgb_img, (int(x), int(y)), (int(x + w), int(y + h)), color=[255, 0, 0], thickness=2)
    cv2.imshow("result", rgb_img)
    cv2.waitKey(1)


if __name__ == '__main__':
    rgb_img_paths = find_img_paths(Path("./data/front_rgb"))
    ss_img_paths = find_img_paths(Path("./data/ss"))
    indexes = range(0, len(rgb_img_paths))

    fields = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    log = []
    df = pd.DataFrame(columns=fields)
    for i in indexes:
        print(f"Frame {i + 1}")
        rgb_img = cv2.imread(rgb_img_paths[i].as_posix())
        ss_img = cv2.imread(ss_img_paths[i].as_posix())
        bboxes = find_bbox_coord(ss_img, min_bb_size=100)
        for box in bboxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            df = df.append(
                {
                    "filename": f"frame_{i + 1}",
                    "width": 800,
                    "height": 600,
                    "class": 1,
                    "xmin": x,
                    "ymin": y,
                    "xmax": x + w,
                    "ymax": y + h
                },
                ignore_index=True
            )
        draw_bbox(rgb_img, bboxes)
    df.to_csv(Path("output.csv"), index=False)

    # print(len(bboxes))
    # draw_bbox(rgb_img, bboxes)
    # print()
