##########################################################################

# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2
import argparse
import sys
import random
import time

##########################################################################

from saliencyDoG import SaliencyDoG

##########################################################################


def draw_bounding_boxes(bboxes, img, box_colour=(0, 0, 255),
                        box_line_thickness=1):

    # Draw bounding boxes onto an image
    # boxes in format: [left_x, left_y, box_width, box_height]

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

##########################################################################


def ransac_bounding_boxes(img, min_box=0.25, max_box=0.4,
                          threashold=2082689, samples=99752,
                          nms_threashold=0.09, n=1, log=True, lpf=True,
                          channels_3=False, mlm=False):

    # read in an image, generate it's saliency map and generate bounding
    # boxes in format: [left_x, left_y, box_width, box_height]

    # min_box = minimum boudning box size as percentage of frame dimensions
    # max_box = maximum boudning box size as percentage of frame dimensions
    # threashold = saliency density (saliency per pixel)
    # samples = number of random samples to take
    # nms_threahsold = acceptable overlap threashold for nms
    # n = maximum number of bounding boxes
    # log = toggle log output to console for time elapsed
    # lpf = toggle low pass filer for saliency mapper
    # channels_3 = toggle processing on 3 channels
    # mlm = toggle saliency map generation on every pyramid layer

    start = time.time()

    # initialize saliency_mapper
    saliency_mapper = SaliencyDoG(low_pass_filter=lpf, ch_3=channels_3,
                                  multi_layer_map=mlm)

    # generate saliency map
    saliency_map = saliency_mapper.generate_saliency(img)

    # generate integral image
    integral_image = cv2.integral(saliency_map)

    bounding_boxes = []
    box_confidences = []
    
    # load image back into CPU
    integral_image = cv2.UMat.get(integral_image)

    # -1 as indexing starts at 0
    frame_width = integral_image.shape[1] - 1
    frame_height = integral_image.shape[0] - 1

    min_box_width = int(min_box * frame_width)
    min_box_height = int(min_box * frame_height)

    max_box_width = int(max_box * frame_width)
    max_box_height = int(max_box * frame_height)

    for box in range(int(samples)):

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

    # perform Non Maximum Suppression
    indices = cv2.dnn.NMSBoxes(bounding_boxes, box_confidences, threashold,
                               nms_threashold)

    # select n best boxes
    n_best_boxes = []

    for idx in indices:

        # upnack index and sort list
        n_best_boxes.append(idx[0])
        n_best_boxes = sorted(n_best_boxes, reverse=True)

        if len(n_best_boxes) > int(n):
            n_best_boxes = n_best_boxes[:int(n)]

    # output n best nms bounding boxes
    nms_bounding_boxes = []

    for idx in n_best_boxes:

        box = bounding_boxes[idx]
        xa = box[0]
        ya = box[1]
        box_width = box[2]
        box_height = box[3]

        nms_bounding_boxes.append([xa, ya, box_width, box_height])

    if log:
        print("Bounding Boxes generated in {}".format(time.time()-start))

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
    parser.add_argument(
        'output_file',
        metavar='output_file',
        type=str,
        nargs='?',
        help='specify output image file')
    args = parser.parse_args()

    # read in image
    img = cv2.imread(args.image_file)

    # generate bounding boxes
    bboxes = ransac_bounding_boxes(img)

    # draw bounding boxes
    output = draw_bounding_boxes(bboxes, img)

    cv2.imwrite(args.output_file, output)

##########################################################################
