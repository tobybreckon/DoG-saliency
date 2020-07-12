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


    # initialize saliency_mapper
    saliency_mapper = SaliencyDoG(ch_3=not(args.grayscale),
                                  low_pass_filter=args.low_pass_filter,
                                  multi_layer_map=args.multi_layer_map)



##########################################################################
