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

    height, width, channels = src.shape

    # un = cv2.pyrDown(src, dstsize=(new_width, new_height))

    for _ in range(n):
        src = cv2.pyrDown(src)

    un = src

    return un

##########################################################################

# if __name__ == '__main__':

##########################################################################

#    bottom_up_gaussian_pyramid(src)
