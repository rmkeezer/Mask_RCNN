"""
Mask R-CNN
Configurations and data loading code for the synthetic notes dataset.
This is a duplicate of the code in the noteobook train_notes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import os.path
import sys
import math
import random
import numpy as np
import cv2
from imageio import imread
import matplotlib
import matplotlib.pyplot as plt

from notes_config256 import NotesConfig

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

FULLSET = np.array([fn for fn in os.listdir(ROOT_DIR + '/datasets/midis/2secmask/') if fn.endswith(".wav.png")])
np.random.shuffle(FULLSET)
split = int(len(FULLSET)*0.9)
TRAINSET, VALSET = FULLSET[:split], FULLSET[split:]

class NotesDataset(utils.Dataset):
    """Generates the notes synthetic dataset. The dataset consists of simple
    notes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_notes(self, settype=None):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        notes = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',\
            'A0','A#0','B0',\
            'C1','C#1','D1','D#1','E1','F1','F#1','G1','G#1','A1','A#1','B1',\
            'C2','C#2','D2','D#2','E2','F2','F#2','G2','G#2','A2','A#2','B2',\
            'C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3',\
            'C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4',\
            'C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5',\
            'C6','C#6','D6','D#6','E6','F6','F#6','G6','G#6','A6','A#6','B6',\
            'C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7',\
            'C8',\
            '-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']
        for i in range(len(notes)):
            self.add_class("notes", i, notes[i])

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of notes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        if settype == "train":
            fnset = TRAINSET
        elif settype == "val":
            fnset = VALSET
        for fn in fnset:
            if os.path.isfile(ROOT_DIR + "/datasets/midis/2secmask/" + fn[:-8] + '_mask.npy'):
                self.add_image("notes", image_id=fn[:-8], path=ROOT_DIR + "/datasets/midis/2secmask/" + fn)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        image = imread(self.image_info[image_id]['path'])
        return image

    def image_reference(self, image_id):
        """Return the notes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "notes":
            notes = np.load(info['path'][:-8] + '_notes.npy')
            print(notes)
            if len(notes) == 0:
                notes = [(0,0)]
            return np.array([n[1] for n in notes])
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for notes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = np.load(info['path'][:-8] + '_mask.npy')
        notes = np.load(info['path'][:-8] + '_notes.npy')
        class_ids = np.array([n[1] for n in notes])
        if len(notes) == 0:
            class_ids = np.array([0])
        #mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        
        #class_ids = np.array([self.class_names.index(s[0]) for s in notes])
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax