import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import colorsys


dir_path = os.path.abspath("../../")

#image_path=os.path.abspath(sys.argv[1]) 
image_path="/home/geo/kh/Mask_RCNN-master_sign/samples/sign_data/all_TW_homo/"
#save_filename = os.path.basename(image_path)
#save_dir = os.path.abspath("../hole_result/project_4")
save_dir = ("/media/sda1/chun/research/result/all_TW_homo_front+homo model")

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

###################################
#############visualize#############
from skimage.measure import find_contours
from matplotlib.patches import Polygon
def random_colors(N, bright=True):

    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):#alpha=0.5
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, all_info=False, save="test.png"):
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax: 
        _, ax = plt.subplots(1, figsize=figsize)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        auto_show = False

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')
    #ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        #color=(1.0, 1.0, 1.0)
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            #x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 , caption,####這裡調整label字型顏色大小
                color='w', size=10)####這裡調整label字型顏色大小
        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    if all_info:
    	#ax.imshow(masked_image.astype(np.uint8))
    	saveax=ax.imshow(masked_image.astype(np.uint8))
    	saveax.figure.savefig(save)
    else:
    	img=(masked_image.astype(np.uint8))
    	skimage.io.imsave(save,img)
    if auto_show:
        plt.show()
#############visualize#############
###################################

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/sign/"))  
import bev_project #############

#MODEL_DIR = os.path.join(ROOT_DIR, "hole_logs/project_1") ##########最新權重檔的資料夾
MODEL_DIR = ("/media/sda1/chun/research/logs/front_homo")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_hole_0100.h5")
print(COCO_MODEL_PATH)
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


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
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'straight arrow', 'left arrow', 'right arrow', 'straight left arrow', 'straight right arrow',	'pedestrian crossing', 'special lane'] #######


for img in os.listdir(image_path):
	image = skimage.io.imread(os.path.join(image_path,img))

	print(img)
	# Run detection
	if image.shape[-1] is 4:
    		image = image[..., :3]
	results = model.detect([image], verbose=1)


	# Visualize results
	r = results[0]
	
	save_DIR=save_dir + '/'+ img
	height, width = image.shape[:2]
	img=display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, 	r['scores'], show_bbox=False, all_info=True, save=save_DIR, figsize=((width/100),(height/100)))
	


'''
image = skimage.io.imread(image_path)
# Run detection
if image.shape[-1] is 4:
    image = image[..., :3]
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
print(r['class_ids'])
save_DIR=save_dir + '/' +save_filename
height, width = image.shape[:2]
img=visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], show_bbox=False, all_info=True, save=save_DIR, figsize=((width/100),(height/100)))'''
#skimage.io.imsave(save_dir + '/' +save_filename,img)

