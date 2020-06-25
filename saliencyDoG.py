##########################################################################

# DoG saliency [Katramados / Breckon 2011] - reference implementation -

# This implementation:
# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2

##########################################################################


def bottom_up_gaussian_pyramid(src, pyramid_height):

    # Produce Un - step 1 of algortithm defined in [Katramados / Breckon 2011]

    un = cv2.add(src, 2**pyramid_height)

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
