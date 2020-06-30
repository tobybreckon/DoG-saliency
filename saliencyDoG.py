##########################################################################

# DoG saliency [Katramados / Breckon 2011] - reference implementation -

# This implementation:
# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2
import numpy as np

##########################################################################


class SaliencyDoG:

    # Parameters:
    # pyramid_height - n as defined in [Katramados / Breckon 2011]
    # shift - k as defined in [Katramados / Breckon 2011]
    # ch_3 - process colour image on every channel
    # low_pass_filter - toggle low pass filter
    # multi_layer_map - the second version of the algortihm as defined in [Katramados / Breckon 2011]

    def __init__(self, pyramid_height=5, shift=5, ch_3=False, low_pass_filter=False):#, multi_layer_map=False):

        self.pyramid_height = pyramid_height
        self.shift = shift
        self.ch_3 = ch_3
        self.low_pass_filter = low_pass_filter
#        self.multi_layer_map = multi_layer_map


    def bottom_up_gaussian_pyramid(self, src):

        # Produce Un - step 1 of algortithm defined in [Katramados / Breckon 2011]
        # Uses a 5 X 5 Gaussian filter

        un = src

        for _ in range(self.pyramid_height):
            height, width = un.shape
            un = cv2.pyrDown(un, (width/2, height/2))

        return un


    def top_down_gaussian_pyramid(self, src):

        # Produce D1 - step 2 of algorithm defined in [Katramados / Breckon 2011]

        dn = src

        for _ in range(self.pyramid_height, 0, -1):
            height, width = dn.shape
            dn = cv2.pyrUp(dn, (width*2, height*2))

        return dn


    def saliency_map(self, u1, d1):

        # Produce S - step 3 of algorithm defined in [Katramados / Breckon 2011]

        # Calculate Minimum Ratio (MiR) Matrix
        matrix_ratio = cv2.divide(u1, d1)
        matrix_ratio_inv = cv2.divide(d1, u1)

        # Caluclate pixelwise min
        mir = cv2.min(matrix_ratio, matrix_ratio_inv)

        # Derive salience by subtracting from scalar 1
        s = cv2.subtract(1.0, mir)

        return s


    def divog_saliency(self, src):

        # Complete implementation of all 3 parts of algortihm defined in
        # [Katramados / Breckon 2011]

        # Convert pixels to 32-bit floats
        src = src.astype(np.float32)

        # Shift image by k^n to avoid division by zero or any number in range
        # 0.0 - 1.0
        src = cv2.add(src, self.shift**self.pyramid_height)

        # Base of Gaussian Pyramid (source frame)
        u1 = src

        un = self.bottom_up_gaussian_pyramid(src)
        d1 = self.top_down_gaussian_pyramid(un)
        s = self.saliency_map(u1, d1)

        # Normalize to 0 - 255 int range
        s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # low-pass filter as defined by original author
        if self.low_pass_filter:
            avg = cv2.mean(s)
            s = cv2.subtract(s, avg)

        return s


    def generate_saliency(self, src):

        if self.ch_3:

            # Split colour image into RBG channels
            channel_array = cv2.split(src)

            # Generate Saliency Map for each channel
            for channel in range(3):

                channel_array[channel] = self.divog_saliency(channel_array[channel])

            # Merge back into one grayscale image with floor division to keep
            # int pixel values
            return channel_array[0]//3 + channel_array[1]//3 + channel_array[2]//3
        else:

            # Convert to grayscale
            src_bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

            # Generate Saliency Map
            return self.divog_saliency(src_bw)

##########################################################################
