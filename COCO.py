########################################################################## 

# Optimize parameters of ransac_bounding_boxes library with COCO dataset
# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

from pycocotools.coco import COCO
import numpy as np
from scipy.optimize import minimize
import cv2

##########################################################################

from ransac_bounding_boxes import ransac_bounding_boxes

##########################################################################


# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def calculateIOU(x):
    min_box=x[0]
    max_box=x[1]
    threashold=x[2]
    samples=x[3] 
    nms_threashold=x[4]
    n=x[5]
    
    ANNOTATIONS_FILE_PATH = "/media/sf_COCO/annotations_trainval2017/annotations/person_keypoints_val2017.json"
    IMAGES_FILE_PATH = "/media/sf_COCO/val2017/"

    coco = COCO(ANNOTATIONS_FILE_PATH)

    # get category id for 'person'
    catIds = coco.getCatIds(catNms=['person'])

    # get annotation id for these cats, and no crowd
    annIds = coco.getAnnIds(catIds=catIds, iscrowd=None)

    # load these annotations
    anns = coco.loadAnns(annIds)

    for annotation in anns:
        
#        load associated image
        img_id = annotation["image_id"]
        img_file_name = coco.loadImgs(img_id)[0]['file_name']

        img = cv2.imread(IMAGES_FILE_PATH + img_file_name)

        # perform bounding box by saliency
        predicted = ransac_bounding_boxes(img, min_box, max_box, threashold, samples, 
                                          nms_threashold, n)

        # no bounding box = 0 iou
        if len(predicted) == 0:
            return 0.0

        # just take best bounding box for now
        predicted = predicted[-1]

        # bounding boxes given as top left coords, width, height
        ground_truth = annotation['bbox']
        ground_truth = [int(x) for x in ground_truth]

        # bounding boxes as pair of coordinates
        ground_truth_coords = [ground_truth[0], ground_truth[1], ground_truth[0] + ground_truth[2], ground_truth[1] + ground_truth[3]]
        predicted_coords = [predicted[0], predicted[1], predicted[0] + predicted[2], predicted[1] + predicted[3]]

        iou = bb_intersection_over_union(ground_truth_coords, predicted_coords)

#        print(ground_truth)
#        print(predicted)

        print(iou)

        # just do one image for now
        break

    return iou

# define our optimization objective
def objective(x):
    # negative as maximization 
    return -calculateIOU(x)

def constraint1(x):
    # max box < 1
    return 1 - x[1]

def constraint2(x):
    # min box < 0.5
    return 0.5 - x[0]

def constraint3(x):
    # min box < max box
    return x[1] - x[0]

cons1 = ({'type': 'ineq', 'fun': constraint2})
cons2 = ({'type': 'ineq', 'fun': constraint2})
cons3 = ({'type': 'ineq', 'fun': constraint3})
cons = [cons1, cons2, cons3]

# define bounds on values for parameters
b1 = (0.0, 0.49)
b2 = (0.5, 1.0)
b3 = (0,10000000)
b4 = (0, 1000000)
b5 = (0, 1)
b6 = (1, 100)
bnds = (b1,b2,b3,b4,b5,b6)

# initial guesses for algorithm
min_box_guess=0.25
max_box_guess=0.9
threashold_guess=2000000
samples_guess=100000
nms_threashold_guess=0.1
n_guess=1

x0 = np.array([min_box_guess, max_box_guess, threashold_guess, samples_guess, nms_threashold_guess, n_guess])

sol = minimize(objective,x0,method="SLSQP",bounds=bnds,constraints=cons,options={"disp": True})

xOpt = sol.x
IOUOpt = -sol.fun

# display parameters and output
print(xOpt)
print(IOUOpt)
