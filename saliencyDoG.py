##########################################################################

# DoG saliency [Katramados / Breckon 2011] - reference implementation -

# This implementation:
# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2

##########################################################################

# import saliencyDOG

##########################################################################


def bottom_up_gaussian_pyramid(src, pyramid_height):

    # Produce Un - step 1 of algortithm defined in [Katramados / Breckon 2011]

    un = src

    for _ in range(pyramid_height):
        height, width, channels = un.shape
        un = cv2.pyrDown(un, (width/2, height/2))

    return un

##########################################################################


def top_down_gaussian_pyramid(src, pyramid_height):

    # Produce D1 - step 2 of algorithm defined in [Katramados / Breckon 2011]

    dn = src

    for _ in range(pyramid_height, 0, -1):
        height, width, channels = dn.shape
        dn = cv2.pyrUp(dn, (width*2, height*2))

    return dn

##########################################################################


#def saliency_map(un, d1):

    # Produce S - step 3 of algorithm defined in [Katramados / Breckon 2011]




# if __name__ == '__main__':

##########################################################################

#    bottom_up_gaussian_pyramid(src)
