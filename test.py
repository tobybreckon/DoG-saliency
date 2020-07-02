import cv2
import numpy as np
from saliencyDoG import SaliencyDoG

class TestClass:
    def test_one(self):
        test_img = cv2.imread('test/samples/fig_2.png')
        test_saliency_mapper = SaliencyDoG()
        test_map = test_saliency_mapper.generate_saliency(test_img)
        test_map_truth = cv2.imread('test/true_saliency_maps/fig_2_saliency.png')
        test_map_truth = cv2.cvtColor(test_map_truth, cv2.COLOR_BGR2GRAY)
        assert np.array_equal(test_map, test_map_truth)

