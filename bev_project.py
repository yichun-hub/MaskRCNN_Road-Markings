"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

'''
import tensorflow as tf
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras import backend as K
K.set_session(sess)'''


import os
import sys
import time
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt


from shapely.geometry import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from coco_master.PythonAPI.pycocotools.coco import COCO
from coco_master.PythonAPI.pycocotools.cocoeval import COCOeval
from coco_master.PythonAPI.pycocotools import mask as maskUtils

import imgaug
from imgaug import augmenters

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = ("/media/sda1/chun/research/logs")




############################################################
#  Configurations
############################################################


class SignConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    NAME = "hole"
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  ##########################+背景+類別

    STEPS_PER_EPOCH = 7000 #100

    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class SignDataset(utils.Dataset):

    def load_sign(self, dataset_dir, subset):
 
        # Add classes. 


        self.add_class("hole", 1, "straight arrow")
        self.add_class("hole", 2, "left arrow")
        self.add_class("hole", 3, "right arrow")
        self.add_class("hole", 4, "straight left arrow")
        self.add_class("hole", 5, "straight right arrow")
        self.add_class("hole", 6, "pedestrian crossing")
        self.add_class("hole", 7, "special lane")
        #self.add_class("sign", 9, "新類別")##############


        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys
	

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        SumLable1=0
        SumLable11=0
        SumLable2=0


        # Add images
        for a in annotations:

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            rects = [r['shape_attributes'] for r in a['regions']]
            name = [s['region_attributes']for s in a['regions']]

######################################################################################

            sumn=0 
            delpolygon=[] 
            num_ids=[]

            for n in name:

            	try:

            		if n['name']=='straight arrow':
            			num_ids.append(1)

            		elif n['name']=='left arrow':
            			num_ids.append(2)

            		elif n['name']=='right arrow':
            			num_ids.append(3)

            		elif n['name']=='straight left arrow':
            			num_ids.append(4)

            		elif n['name']=='straight right arrow':
            			num_ids.append(5)

            		elif n['name']=='pedestrian crossing':
            			num_ids.append(6)

            		elif n['name']=='special lane':
            			num_ids.append(7)

            		else:
            			delpolygon.append(sumn)


            	except:
            		pass
            	sumn+=1
            polygons=np.delete(polygons,delpolygon)


            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            idSUM=0
            delr=[]
            print(num_ids)
            if (num_ids!=[]):
            	for g in polygons:           
            		try:
            			xr = max(g['all_points_x']) - min(g['all_points_x'])
            			yr = max(g['all_points_y']) - min(g['all_points_y'])
            			print('xr='+str(xr)+',yr='+str(yr))
            			if (xr < 20) and (yr < 20):
            				delr.append(idSUM)
            				print('too small')
            			else:
            				print('xr,yr > 20')

            		except:
            			pass
            		idSUM+=1
            polygons=np.delete(polygons,delr)
            num_ids=np.delete(num_ids,delr)
            print(str(num_ids)+'//'+str(polygons))

            self.add_image(
                	"hole",
                	image_id=a['filename'],  
                	path=image_path,
                	class_id=num_ids,
                	width=width, height=height,
                	polygons=polygons)





    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        #print(image_info)
        if image_info["source"] != "hole":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        #print(info)
        if info["source"] != "hole":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['class_id']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
          if p['name'] == 'ellipse':
            rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'])
            mask[rr, cc, i] = 1

          elif (p['name'] == 'polygon') or (p['name'] == 'polyline'):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

          else:
            raise Exception('Unknown annotation type. Supported annotation types: Polygon, Polyline, Ellipse.')

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask.astype(np.bool), num_ids


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "hole":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

#augmentation = imgaug.augmenters.Sometimes(5/6,imgaug.augmenters.OneOf([  
               #imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
               #imgaug.augmenters.ContrastNormalization((0.75, 1.5)),
               #imgaug.augmenters.AddToBrightness((-30, 30)),
               #imgaug.augmenters.Add((-20, 20))]))


def train(model):
    """Train thecd model."""
    # Training dataset.
    dataset_train = SignDataset()
    dataset_train.load_sign(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SignDataset()
    dataset_val.load_sign(args.dataset, "val")
    dataset_val.prepare()
    print('Train: %d' % len(dataset_train.image_ids))
    print('Test: %d' % len(dataset_val.image_ids))

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training heads network")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
		epochs=100,#100
		layers='all')
'''
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=40,
                layers='all',
                augmentation=augmentation)'''



def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

def loss_visualize(epoch, tra_loss, val_loss):
    plt.style.use("ggplot")
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("Epoch_Loss")
    plt.plot(epoch, tra_loss, label='train_loss', color='r', linestyle='-', marker='o')
    plt.plot(epoch, val_loss, label='val_loss', linestyle='-', color='b', marker='^')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(RESULT_DIR, 'loss.jpg'))
    plt.show()

#x_epoch, y_tra_loss, y_val_loss = modellib.call_back()
#loss_visualize(x_epoch, y_tra_loss, y_val_loss)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments #解析命令行參數 
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SignConfig()
    else:
        class InferenceConfig(SignConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = SignDataset()
        #val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_sign(args.dataset, subset="val")
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(60))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(60))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

#sess.close()



