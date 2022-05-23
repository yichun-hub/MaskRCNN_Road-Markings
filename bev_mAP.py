import os
import sys
import random
import math
import time
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import colorsys
import codecs, json


from mrcnn import visualize
from mrcnn.model import log


dir_path = os.path.abspath("../../")


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

from coco_master.PythonAPI.pycocotools.coco import COCO
from coco_master.PythonAPI.pycocotools.cocoeval import COCOeval
from coco_master.PythonAPI.pycocotools import mask as maskUtils

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/sign/"))  # To find local version
import bev_project

#MODEL_DIR = os.path.join(ROOT_DIR, "hole_logs/project_1") ##########最新權重檔的資料夾
MODEL_DIR = ("/media/sda1/chun/research/logs/homo_2data_2")

# Local path to trained weights file
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_hole_0100.h5")
print(MODEL_PATH)

# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "hole_logs")

# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
    #utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(bev_project.SignConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH, by_name=True)

class_names = ['BG', 'straight arrow', 'left arrow', 'right arrow', 'straight left arrow', 'straight right arrow', 'pedestrian crossing', 'special lane'] #######





############################################################
#  Dataset
############################################################
dataset_val = bev_project.SignDataset()
dataset_val.load_sign("/home/geo/kh/Mask_RCNN-master_sign/samples/sign_data", "val")#######
dataset_val.prepare()
print('Test: %d' % len(dataset_val.image_ids))

#########################################################################################

APs = []
results= []
anno = []

#np.random.shuffle(dataset_val.image_ids)
for image_id in dataset_val.image_ids: 
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config,
                               image_id, use_mini_mask=False)
    
    #print('gt_class_id:', gt_class_id)
    #print('gt_bbox:', gt_bbox)
    #print('gt_mask:', gt_mask)
    
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    info = dataset_val.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset_val.image_reference(image_id)))
    

    # Run object detection
    result = model.detect([image], verbose=0)
    r = result[0]
   
    #print('result=',r["rois"], r["class_ids"], r["scores"], r['masks'])
    
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=0.5)
    APs.append(AP)
    print(AP)
    #print(precisions)
    #print(recalls)


'''
    ### predict
    del r['masks']
    r['image_id'] = info["id"]
    value_list = r['rois'].tolist(), r['class_ids'].tolist(), r['scores'].tolist(), [r['image_id']]
    key_list = r.keys()
    result_list = dict(zip(key_list, value_list))
    results.append(result_list)



    ### groundtruth
    annotation_list = gt_bbox.tolist(), gt_class_id.tolist(), [info["id"]]
    annokey_list = ['gt_bbox', 'gt_class_id', 'image_id']
    anno_file = dict(zip(annokey_list, annotation_list))
    anno.append(anno_file)

file_name = 'results_allTW.json'
with open(file_name,'w') as outfile:
    json.dump(results, outfile,ensure_ascii=False)

file_name = 'gt_allTW.json'
with open(file_name,'w') as outfile:
    json.dump(anno, outfile,ensure_ascii=False)'''



  

print("APs: ", APs)
print("mAP: ", np.mean(APs) * 100, "%")









