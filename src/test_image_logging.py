# Train Neural network on dataset with fast.ai
from pathlib import Path
from fastai.vision import *
import os
import wandb
from fastai.callbacks.hooks import *
from fastai.callback import Callback
import json

from wandb.fastai import WandbCallback
from functools import partialmethod
import PIL
import torch

WB_PROJECT="test-deep-drive"
WB_ENTITY="stacey"

# for testing
#os.environ['WANDB_MODE'] = 'dryrun'

# Segmentation Classes extracted from dataset source code
# See https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/label.py
segmentation_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]
     
# Initialize W&B project
wandb.init(project=WB_PROJECT, entity=WB_ENTITY)

# Define hyper-parameters
config = wandb.config           # for shortening
config.framework = "fast.ai"    # AI framework used (for when we create other versions)
config.img_size = (360, 640)    # dimensions of resized image - can be 1 dim or tuple

config.batch_size = 6          # Batch size during training
config.epochs = 10             # Number of epochs for training

# data paths
path_lbl = Path() / 'mini_labels'
path_img = Path() / 'mini_images'

# Associate a label to an input
get_y_fn = lambda x: path_lbl / f'{x.stem}_train_id.png'

# Load data into train & validation sets
src = (SegmentationItemList.from_folder(path_img) #.use_partial_data(0.1)
       .split_none().label_from_func(get_y_fn, classes=segmentation_classes))

# Resize, augment, load in batch & normalize (so we can use pre-trained networks)
data = (src.transform(get_transforms(), size=config.img_size, tfm_y=True)
        .databunch(bs=config.batch_size)
        .normalize(imagenet_stats))

print("Number of images: ", len(data.train_ds))

raw = []
ground_truth = []
# process images
for image in data.train_ds:
  # log raw image
  x = image2np(image[0].data*255).astype(np.uint8)
  raw_source = PIL.Image.fromarray(x)
  raw.append(raw_source)
 
  # log ground truth prediction: convert to plotly color map
  # via image save (instead of fastai default)
  img_label = image[1]
  x_label = image2np(img_label.data).astype(np.uint8)
  plt.imsave("label_x.png", x_label, cmap="tab20")
  f = open_image("label_x.png")
  x = image2np(f.data*255).astype(np.uint8)
  raw_x_label = PIL.Image.fromarray(x)
  ground_truth.append(raw_x_label)  

# log arrays to W&B
wandb.log({"camera view" : [wandb.Image(e) for e in raw],
          "ground truth" : [wandb.Image(e) for e in ground_truth]})

