import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
from imageio import imread

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/new")

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/new/notes20180630T1719/mask_rcnn_notes_0030.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/new/notes20180702T1230/mask_rcnn_notes_0007.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

from notes import NotesDataset, NotesConfig, get_ax
    
config = NotesConfig()
config.display()

# Training dataset
dataset_train = NotesDataset()
dataset_train.load_notes('train')
dataset_train.prepare()

# Validation dataset
dataset_val = NotesDataset()
dataset_val.load_notes('val')
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(COCO_MODEL_PATH, by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1000, 
            layers='heads')