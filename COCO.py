from pycocotools.coco import COCO
import numpy as np
import cv2

ANNOTATIONS_FILE_PATH = "/media/sf_COCO/annotations_trainval2017/annotations/person_keypoints_val2017.json"
IMAGES_FILE_PATH = "/media/sf_COCO/val2017"

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
    img = cv2.imread(img_file_name)

    break
