##########################################################################

# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2
import argparse
import sys
import math

##########################################################################

from saliencyDoG import SaliencyDoG

##########################################################################


if __name__ == "__main__":

    keep_processing = True
    toggle_saliency = True

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

    import random

    # initialize saliency_mapper
    saliency_mapper = SaliencyDoG()

    img = cv2.imread(args.image_file)
    saliency_map = saliency_mapper.generate_saliency(img)
    integral_image = cv2.integral(saliency_map)

    minimum_box = 5

    for box in range(1):

        x1 = random.randint(0, integral_image.shape[0])
        y1 = random.randint(0, integral_image.shape[1])

        x2 = random.randint(0, integral_image.shape[0])
        y2 = random.randint(0, integral_image.shape[1])

        if abs(x1-x2) < 5 or abs(y1-y2) < 5:

            continue

        else:

            x3 = x1
            y3 = y2

            x4 = x2
            y4 = y1

            print(x1, y1)
            print(x2, y2)
            print(x3, y3)
            print(x4, y4)

            xa = min(x1, x2)
            ya = min(y1, y2)

            xb = max(x1, x2)
            yb = min(y1, y2)

            xc = min(x1, x2)
            yc = max(y1, y2)

            xd = max(x1, x2)
            yd = max(y1, y2)

            print("################")


            print(xa, ya)
            print(xb, yb)
            print(xc, yc)
            print(xd, yd)














#    cv2.imshow("Bounding Boxes", output)
#    cv2.waitKey(0)

##########################################################################
