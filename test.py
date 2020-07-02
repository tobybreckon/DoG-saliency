##########################################################################

# Tests for DoG Saliency

##########################################################################

import cv2
import numpy as np
from saliencyDoG import SaliencyDoG

##########################################################################

ORIGINAL_DIR = 'test/samples/'
TRUTH_DIR = 'test/true_saliency_maps/'

##########################################################################


class TestClass:

    def test_one(self):
        test_img = cv2.imread(ORIGINAL_DIR + 'fig_2.png')
        test_saliency_mapper = SaliencyDoG()
        test_map = test_saliency_mapper.generate_saliency(test_img)
        test_map_truth = cv2.imread(TRUTH_DIR + 'fig_2_saliency.png',
                                    0)
        assert np.array_equal(test_map, test_map_truth)

    def test_two(self):
        test_img = cv2.imread(ORIGINAL_DIR + 'fig_2.png')
        test_saliency_mapper = SaliencyDoG(ch_3=True)
        test_map = test_saliency_mapper.generate_saliency(test_img)
        test_map_truth = cv2.imread(TRUTH_DIR + 'fig_2_saliency_3_ch.png',
                                    0)
        assert np.array_equal(test_map, test_map_truth)

    def test_three(self):
        test_img = cv2.imread(ORIGINAL_DIR + 'fig_2.png')
        test_saliency_mapper = SaliencyDoG(low_pass_filter=True)
        test_map = test_saliency_mapper.generate_saliency(test_img)
        test_map_truth = cv2.imread(TRUTH_DIR + 'fig_2_saliency_lp.png',
                                    0)
        assert np.array_equal(test_map, test_map_truth)

    def test_four(self):
        test_img = cv2.imread(ORIGINAL_DIR + 'fig_2.png')
        test_saliency_mapper = SaliencyDoG(multi_layer_map=True)
        test_map = test_saliency_mapper.generate_saliency(test_img)
        test_map_truth = cv2.imread(TRUTH_DIR + 'fig_2_saliency_mlm.png',
                                    0)
        assert np.array_equal(test_map, test_map_truth)

##########################################################################
