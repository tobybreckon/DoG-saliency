##########################################################################

# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2
import argparse
import sys
import random
import time
import numpy as np

##########################################################################

from saliencyDoG import SaliencyDoG

##########################################################################
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def ransac_bounding_boxes(img, min_box=50, threashold=40, samples=10000,
                          box_colour=(0, 0, 255), box_line_thickness=1):

    # read in an image, generate it's saliency map and place bounding
    # boxes

    # min_box = minimum dimension of either side of box
    # threashold = saliency density (saliency per pixel)
    # samples = number of random samples to take
    # box_colour = BGR tuple for box outine colour
    # box_line_thickness = box outline thickness

    start = time.time()
    # initialize saliency_mapper
    saliency_mapper = SaliencyDoG()

    # generate saliency map
    saliency_map = saliency_mapper.generate_saliency(img)

    # generate integral image
    integral_image = cv2.integral(saliency_map)

    # draw bounding boxes on original image
    output = img

    boxes = []
    confidences = []

    for box in range(samples):

        # -1 as indexing starts at 0

        x1 = random.randint(0, (integral_image.shape[1] - 1))
        y1 = random.randint(0, (integral_image.shape[0] - 1))

        x2 = random.randint(0, (integral_image.shape[1] - 1))
        y2 = random.randint(0, (integral_image.shape[0] - 1))

        if abs(x1-x2) < min_box or abs(y1-y2) < min_box:

            continue

        else:

            # define points a,b,c,d in integral image to calculate
            # saliency in box

            xa = min(x1, x2)
            ya = min(y1, y2)

            xb = max(x1, x2)
            yb = min(y1, y2)

            xc = min(x1, x2)
            yc = max(y1, y2)

            xd = max(x1, x2)
            yd = max(y1, y2)

            box_saliency = (integral_image[yd][xd] - integral_image[yb][xb] -
                            integral_image[yc][xc] + integral_image[ya][xa])

            box_saliency_density = box_saliency / (abs(x1-x2) * abs(y1-y2))

            if box_saliency_density > threashold:
                boxes.append([x1, y1, x2, y2])


    indices = non_max_suppression_fast(np.asarray(boxes), 0.99)
    print(indices)
#            confidences.append(box_saliency_density)

#    indices = cv2.dnn.NMSBoxes(boxes, confidences, threashold, 0.01)

    for i in indices:
#        i = i[0]
#        box = boxes[i]
        x1 = i[0]
        y1 = i[1]
        x2 = i[2]
        y2 = i[3]

        output = cv2.rectangle(output, (x1, y1), (x2, y2),
                               box_colour, box_line_thickness)


#            if box_saliency_density > threashold:

#                output = cv2.rectangle(output, (x1, y1), (x2, y2),
#                                       box_colour, box_line_thickness)

    return output

##########################################################################


if __name__ == "__main__":

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

    # read in image
    img = cv2.imread(args.image_file)

    # generate bounding boxes
    output = ransac_bounding_boxes(img)

    cv2.imwrite("BoundingBoxes.png", output)
    cv2.waitKey(0)

##########################################################################
