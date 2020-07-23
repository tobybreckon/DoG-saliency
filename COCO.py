from pycocotools.coco import COCO
import numpy as np
import cv2
from ransac_bounding_boxes import ransac_bounding_boxes

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
    
    # load associated image
    img_id = annotation["image_id"]
    img_file_name = coco.loadImgs(img_id)[0]['file_name']
    img = cv2.imread(IMAGES_FILE_PATH + img_file_name)

    # perform bounding box by saliency
    predicted = ransac_bounding_boxes(img)
    
    # just take first bounding box for now
    predicted = predicted[0]

    # bounding boxes given as top left coords, width, height
    ground_truth = annotation['bbox']

    # convert to 2 sets of coords
    ground_truth[2] = ground_truth[0] + ground_truth[2]
    ground_truth[3] = ground_truth[1] + ground_truth[3]
    ground_truth = [int(x) for x in ground_truth]

    ground_truth_coords = [ground_truth[0], ground_truth[1], ground_truth[0] + ground_truth[2], ground_truth[1] + ground_truth[3]]
    predicted_coords = [predicted[0], predicted[1], predicted[0] + predicted[2], predicted[1] + predicted[3]]

    iou = bb_intersection_over_union(ground_truth_coords, predicted_coords)

    print(ground_truth)
    print(predicted)

    print(iou)

    break
