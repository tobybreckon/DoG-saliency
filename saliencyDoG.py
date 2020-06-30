##########################################################################

# DoG saliency [Katramados / Breckon 2011] - reference implementation -

# This implementation:
# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2
import numpy as np

##########################################################################


class SaliencyDoG:
    def __init__(self, pyramid_height=5, shift=5, 3_ch=False, multi_layer_map=False):

        self.pyramid_height = pyramid_height
        self.shift = shift
        self.3_ch = 3_ch
        self.multi_layer_map = multi_layer_map

def bottom_up_gaussian_pyramid(src, pyramid_height):

    # Produce Un - step 1 of algortithm defined in [Katramados / Breckon 2011]
    # Uses a 5 X 5 Gaussian filter

    un = src

    for _ in range(pyramid_height):
        height, width = un.shape
        un = cv2.pyrDown(un, (width/2, height/2))

    return un

##########################################################################


def top_down_gaussian_pyramid(src, pyramid_height):

    # Produce D1 - step 2 of algorithm defined in [Katramados / Breckon 2011]

    dn = src

    for _ in range(pyramid_height, 0, -1):
        height, width = dn.shape
        dn = cv2.pyrUp(dn, (width*2, height*2))

    return dn

##########################################################################


def saliency_map(u1, d1):

    # Produce S - step 3 of algorithm defined in [Katramados / Breckon 2011]

    # Calculate Minimum Ratio (MiR) Matrix
    matrix_ratio = cv2.divide(u1, d1)
    matrix_ratio_inv = cv2.divide(d1, u1)

    # Caluclate pixelwise min
    mir = cv2.min(matrix_ratio, matrix_ratio_inv)

    # Derive salience by subtracting from scalar 1
    s = cv2.subtract(1.0, mir)

    return s


##########################################################################


def divog_saliency(src, pyramid_height):

    # Complete implementation of all 3 parts of algortihm defined in
    # [Katramados / Breckon 2011]

    # Convert pixels to 32-bit floats
    src = src.astype(np.float32)

    # Shift image by 5^n to avoid division by zero or any number in range
    # 0.0 - 1.0
    src = cv2.add(src, 5**pyramid_height)

    # Base of Gaussian Pyramid (source frame)
    u1 = src

    un = bottom_up_gaussian_pyramid(src, pyramid_height)
    d1 = top_down_gaussian_pyramid(un, pyramid_height)
    s = saliency_map(u1, d1)

    # Normalize to 0 - 255 int range
    s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return s

##########################################################################
