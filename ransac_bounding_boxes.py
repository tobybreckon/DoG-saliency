##########################################################################

# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2
import argparse
import sys
import random
import time
import numpy as np

##########################################################################

from saliencyDoG import SaliencyDoG

##########################################################################


# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
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
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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
    return boxes[pick].astype("int"), x1

##########################################################################


def draw_bounding_boxes(bboxes, img, box_colour=(0, 0, 255),
                        box_line_thickness=1):
    
    # Draw bounding boxes onto an image
    
    # box_colour = BGR tuple for box outine colour
    # box_line_thickness = box outline thickness
    
    for box in bboxes:

        xa = box[0]
        ya = box[1]
        box_width = box[2]
        box_height = box[3]

        img = cv2.rectangle(img, (xa, ya), (xa + box_width, ya + box_height),
                            box_colour, box_line_thickness)

    return img


def ransac_bounding_boxes(img, min_box=0.25, max_box=0.5, threashold=2000000,
                          samples=100000, nms_threashold=0.1):

    # read in an image, generate it's saliency map and generate bounding
    # boxes in format: [left_x, left_y, box_width, box_height]

    # min_box = minimum boudning box size as percentage of frame dimensions
    # max_box = maximum boudning box size as percentage of frame dimensions
    # threashold = saliency density (saliency per pixel)
    # samples = number of random samples to take
    # nms_threahsold = acceptable overlap threashold for nms

    start = time.time()

    # initialize saliency_mapper
    saliency_mapper = SaliencyDoG(low_pass_filter=True, ch_3=True,
                                  multi_layer_map=True)

    # generate saliency map
    saliency_map = saliency_mapper.generate_saliency(img)

    # generate integral image
    integral_image = cv2.integral(saliency_map)

    bounding_boxes = []
    box_confidences = []

    # -1 as indexing starts at 0
    frame_width = integral_image.shape[1] - 1
    frame_height = integral_image.shape[0] - 1

    min_box_width = int(min_box * frame_width)
    min_box_height = int(min_box * frame_height)

    max_box_width = int(max_box * frame_width)
    max_box_height = int(max_box * frame_height)

    for box in range(samples):

        # define points a,b,c,d in integral image to calculate
        # saliency in box

        # pick random pixel - point d
        xd = random.randint(min_box_width, frame_width)
        yd = random.randint(min_box_height, frame_height)

        # pick point a to left and up of point d
        xa = random.randint(max(0, xd - max_box_width), xd - min_box_width)
        ya = random.randint(max(0, yd - max_box_height), yd - min_box_height)

        xb = xd
        yb = ya

        xc = xa
        yc = yd

        box_saliency = (integral_image[yd][xd] - integral_image[yb][xb] -
                        integral_image[yc][xc] + integral_image[ya][xa])

        if box_saliency > threashold:

            box_width = xd - xa
            box_height = yd - ya

            bounding_boxes.append([xa, ya, box_width, box_height])
            box_confidences.append(float(box_saliency))

    indices = cv2.dnn.NMSBoxes(bounding_boxes, box_confidences, threashold, nms_threashold)

    nms_bounding_boxes = []

    for idx in indices:
        
        # unpack idx
        idx = idx[0]

        box = bounding_boxes[idx]
        xa = box[0]
        ya = box[1]
        box_width = box[2]
        box_height = box[3]

        nms_bounding_boxes.append([xa, ya, box_width, box_height])

    print(time.time()-start)

    return nms_bounding_boxes

##########################################################################


if __name__ == "__main__":

    # parse command line arguments for camera ID or video file

    parser = argparse.ArgumentParser(
        description='Perform ' +
        sys.argv[0] +
        ' example operation on incoming camera/video image')
    parser.add_argument(
        'image_file',
        metavar='image_file',
        type=str,
        nargs='?',
        help='specify image file')
    args = parser.parse_args()

    # read in image
    img = cv2.imread(args.image_file)

    # generate bounding boxes
    bboxes = ransac_bounding_boxes(img)

    output = draw_bounding_boxes(bboxes, img)

    cv2.imwrite("BoundingBoxes.png", output)
    cv2.waitKey(0)

##########################################################################
