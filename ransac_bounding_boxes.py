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


def ransac_bounding_boxes(img, min_box=0.05, max_box=0.25, threashold=45,
                          samples=100000, box_colour=(0, 0, 255),
                          box_line_thickness=1):

    # read in an image, generate it's saliency map and place bounding
    # boxes

    # min_box = minimum boudning box size as percentage of frame dimensions
    # max_box = maximum boudning box size as percentage of frame dimensions
    # threashold = saliency density (saliency per pixel)
    # samples = number of random samples to take
    # box_colour = BGR tuple for box outine colour
    # box_line_thickness = box outline thickness

    start = time.time()

    # initialize saliency_mapper
    saliency_mapper = SaliencyDoG()

    # generate saliency map
    saliency_map = saliency_mapper.generate_saliency(img)

    # generate integral image
    integral_image = cv2.integral(saliency_map)

    # draw bounding boxes on original image
    output = img

#    boxes = []
#    confidences = []

    # -1 as indexing starts at 0
    frame_width = integral_image.shape[1] - 1
    frame_height = integral_image.shape[0] - 1

    min_box_width = round(min_box * frame_width)
    min_box_height = round(min_box * frame_height)

    max_box_width = round(max_box * frame_width)
    max_box_height = round(max_box * frame_height)

    for box in range(samples):

        # pick random pixel - point 1
        x1 = random.randint(min_box_width, frame_width)
        y1 = random.randint(min_box_height, frame_height)

        # pick pixel to left and up of point 1
        x2 = random.randint(max(0, x1 - max_box_width), x1 - min_box_width)
        y2 = random.randint(max(0, y1 - max_box_height), y1 - min_box_height)

        # define points a,b,c,d in integral image to calculate
        # saliency in box

        xa = min(x1, x2)
        ya = min(y1, y2)

        xb = max(x1, x2)
        yb = min(y1, y2)

        xc = min(x1, x2)
        yc = max(y1, y2)

        xd = max(x1, x2)
        yd = max(y1, y2)

        box_saliency = (integral_image[yd][xd] - integral_image[yb][xb] -
                        integral_image[yc][xc] + integral_image[ya][xa])

        box_saliency_density = box_saliency / (abs(x1-x2) * abs(y1-y2))

        if box_saliency_density > threashold:
#                boxes.append([x1, y1, x2, y2])
#                confidences.append(box_saliency_density)


            output = cv2.rectangle(output, (x1, y1), (x2, y2),
                                   box_colour, box_line_thickness)


    print(time.time()-start)

    return output

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

    ##########################################################################

    # read in image
    img = cv2.imread(args.image_file)

    # generate bounding boxes
    output = ransac_bounding_boxes(img)

    cv2.imwrite("BoundingBoxes.png", output)
    cv2.waitKey(0)

##########################################################################
