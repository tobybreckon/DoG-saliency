##########################################################################

# DoG saliency [Katramados / Breckon 2011] - reference implementation -

# This implementation:
# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2

##########################################################################

# import saliencyDOG

##########################################################################


def bottom_up_gaussian_pyramid(src, n):

    # Produce Un - step 1 of algortithm defined in [Katramados / Breckon 2011]

    # height, width, channels = src.shape

    # un = cv2.pyrDown(src, dstsize=(new_width, new_height))

    un = src

    for _ in range(n):
        un = cv2.pyrDown(un)

    return un

##########################################################################


def top_down_gaussian_pyramid(src, n):

    # Produce Dn - step 2 of algorithm defined in [Katramados / Breckon 2011]

    dn = src

    for _ in range(n, 0, -1):
        dn = cv2.pyrUp(dn)

    return dn

# if __name__ == '__main__':

##########################################################################

#    bottom_up_gaussian_pyramid(src)
