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
    threashold = 30000

    for scale in range(1, 9):

        saliency_map_scaled = cv2.resize(saliency_map, (saliency_map.shape[1]/2**scale, saliency_map.shape[0]/2**scale):

        for box in range(100000):

            x1 = random.randint(0, (integral_image.shape[1] - 1) / 8)
            y1 = random.randint(0, (integral_image.shape[0] - 1) / 8)

            x2 = random.randint(0, (integral_image.shape[1] - 1) / 8)
            y2 = random.randint(0, (integral_image.shape[0] - 1) / 8)

            if abs(x1-x2) < 5 or abs(y1-y2) < 5:

                continue

            else:


                xa = min(x1, x2)
                ya = min(y1, y2)

                xb = max(x1, x2)
                yb = min(y1, y2)

                xc = min(x1, x2)
                yc = max(y1, y2)

                xd = max(x1, x2)
                yd = max(y1, y2)

#            print(xa, ya)
#            print(xb, yb)
#            print(xc, yc)
#            print(xd, yd)

                box_saliency = integral_image[yd][xd] - integral_image[yb][xb] - integral_image[yc][xc] + integral_image[ya][xa]

                if box_saliency < threashold:

                    continue

                else:

                    print(box_saliency)
                    saliency_map = cv2.rectangle(saliency_map, (x1, y1), (x2, y2), (255, 0, 0), 2)



    cv2.imwrite("BoundingBoxes.png", saliency_map)

##########################################################################
