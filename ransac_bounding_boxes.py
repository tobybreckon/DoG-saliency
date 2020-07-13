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

    # read in image
    img = cv2.imread(args.image_file)

    # generate saliency map
    saliency_map = saliency_mapper.generate_saliency(img)

    minimum_box = 5
    maximum_box = 50
    threashold = 100000

    output = img

    for scale in range(0, 1):

        saliency_map_scaled = cv2.resize(saliency_map, (saliency_map.shape[1]//(2**scale), saliency_map.shape[0]//(2**scale)))

        integral_image = cv2.integral(saliency_map_scaled)

        for box in range(100000):

            x1 = random.randint(0, (integral_image.shape[1] - 1))
            y1 = random.randint(0, (integral_image.shape[0] - 1))

            x2 = random.randint(0, (integral_image.shape[1] - 1))
            y2 = random.randint(0, (integral_image.shape[0] - 1))

            if abs(x1-x2) < minimum_box or abs(y1-y2) < minimum_box or abs(x1-x2) > maximum_box or abs(y1-y2) > maximum_box:

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


                box_saliency = integral_image[yd][xd] - integral_image[yb][xb] - integral_image[yc][xc] + integral_image[ya][xa]

                if box_saliency > threashold*(2**scale):

                    print(box_saliency)
                    output = cv2.rectangle(output, (x1*(2**scale), y1*(2**scale)), (x2*(2**scale), y2*(2**scale)), (255, 0, 0), 2)


    cv2.imwrite("BoundingBoxes.png", output)
    cv2.waitKey(0)

##########################################################################
