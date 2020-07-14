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


def ransac_bounding_boxes(img, min_box=50, threashold=28, samples=10000,
                          box_colour=(0, 0, 255), box_line_thickness=1):

    # read in an image, generate it's saliency map and place bounding
    # boxes

    # min_box = minimum dimension of either side of box
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

    for box in range(samples):

        # -1 as indexing starts at 0

        x1 = random.randint(0, (integral_image.shape[1] - 1))
        y1 = random.randint(0, (integral_image.shape[0] - 1))

        x2 = random.randint(0, (integral_image.shape[1] - 1))
        y2 = random.randint(0, (integral_image.shape[0] - 1))

        if abs(x1-x2) < min_box or abs(y1-y2) < min_box:

            continue

        else:

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

                output = cv2.rectangle(output, (x1, y1), (x2, y2),
                                       box_colour, box_line_thickness)

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
