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

segmentation_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]

def labels():
  l = {}
  for i, label in enumerate(segmentation_classes):
    l[i] = label
  return l

def rand_image():
  return np.random.randint(255, size=(400, 400, 3))

# generate random masks -- Nick's test code
def gen_random_masks():
  mask_list = []
  n = 400
  m = 400
  for i in range(n):
    inner_list = []
    for j in range(m):
      v = 0
      if i < 200:
        v = 1
      if i > 200:
        v = 2
      if j < 200:
        v = v + 3
      if j > 200:
        v = v + 6
      inner_list.append(v)
    mask_list.append(inner_list)

  mask_data = np.array(mask_list)
  class_labels = {
                0 : "car",
                1 : "pedestrian",
                4 : "truck",
                5 : "tractor",
                7 : "barn",
                8 : "sign",
                }

  for i in range(0,100):
      class_labels[i] = "tag " + str(i)
  return mask_data, class_labels

def gen_mask_img(mask_data, class_labels):
  mask_img = wandb.Image(np.array(rand_image()), \
             masks = {"predictions" : {
               "mask_data" : mask_data,
               "class_labels" : class_labels
               }
             })
  return mask_img

def gen_mask_img2(mask_data, class_labels):
  mask_img = wandb.Image(np.array(rand_image()), masks={
        "predictions_0":
        {"mask_data": mask_data,
            "class_labels": class_labels },
        "predictions_1":
        {"mask_data": mask_data,
            "class_labels": class_labels }}) 

  return mask_img


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
x_label = []
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


  # include Nick's testing
print("X LABEL: ", x_label)
#mask_data, class_labels = gen_random_masks()
#print("MASK DATA: ", mask_data)
# new class labels
mask_data = x_label
class_labels = labels()
print("CLASS LABELS: ", class_labels)
 

wandb.log({
  "mask_img_single": gen_mask_img(mask_data, class_labels),
  "mask_img_multi_mask": gen_mask_img2(mask_data, class_labels),
  "mask_img_list": [gen_mask_img(mask_data, class_labels), gen_mask_img(mask_data, class_labels)]
})
   


# log arrays to W&B
#wandb.log({"camera view" : [wandb.Image(e) for e in raw],
#          "ground truth" : [wandb.Image(e) for e in ground_truth]})

