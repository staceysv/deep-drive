# Train a U-Net for semantic segmentation with fast.ai
from pathlib import Path
from fastai.vision import *
import wandb
from fastai.callbacks.hooks import *
from fastai.callback import Callback
import json

from wandb.fastai import WandbCallback
from functools import partialmethod
import PIL
import torch
import time

# Segmentation Classes extracted from dataset source code
# See https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/label.py
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

# convenience for wrapping masks
def wb_mask(bg_img, pred_mask, true_mask):
  return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask, "class_labels" : labels()},
    "ground truth" : {"mask_data" : true_mask, "class_labels" : labels()}})

class LogImagesCallback(Callback):

  def __init__(self, learn):
    self.learn = learn
   
  # log semantic segmentation masks
  def on_epoch_end(self, **kwargs):
    num_log = 5
    # save some training AND testing data
    train_batch = self.learn.data.train_ds[:num_log]
    valid_batch = self.learn.data.valid_ds[:num_log]
    train_masks = []
    valid_masks = []
    for batch_masks, batch in zip([train_masks, valid_masks], [train_batch, valid_batch]):
 
      for i, img in enumerate(batch):
        # log raw image as array
        orig_image = img[0]
        bg_image = image2np(orig_image.data*255).astype(np.uint8) 
        # our prediction
        prediction = learn.predict(img[0])[0]
        prediction_mask = image2np(prediction.data).astype(np.uint8)
      
        # ground truth
        ground_truth = img[1]
        true_mask = image2np(ground_truth.data).astype(np.uint8)
 
        batch_masks.append(wb_mask(bg_image, prediction_mask, true_mask))

    wandb.log({"train predictions" : train_masks, "validation predictions" : valid_masks})

# Initialize W&B project
wandb.init(project="cars_train_loop", entity="stacey")

# Define hyper-parameters
config = wandb.config           # for shortening
config.framework = "fast.ai"    # AI framework used (for when we create other versions)
config.img_size = (360, 640)    # dimensions of resized image - can be 1 dim or tuple

config.batch_size = 6           # Batch size during training
config.epochs = 10            # Number of epochs for training

config.encoder = "resnet34" #resnet34"
if config.encoder == "resnet18":
  encoder = models.resnet18     # encoder of unet (contracting path)
elif config.encoder == "resnet34":
  encoder = models.resnet34
elif config.encoder == "squeezenet1_0":
  encoder = models.squeezenet1_0
elif config.encoder == "squeezenet1_1":
  encoder = models.squeezenet1_1
elif config.encoder == "alexnet":
  encoder = models.alexnet

#config.encoder = encoder.__name__
#config.encoder = "resnet18"
#encoder = models.resnet18
config.pretrained = True        # whether we use a frozen pre-trained encoder


# SWEEPS UNCOMMENT
config.weight_decay = 0.08173 #0.097     # weight decay applied on layers
config.bn_weight_decay = True # whether weight decay is applied on batch norm layers
config.one_cycle = True         # use the "1cycle" policy -> https://arxiv.org/abs/1803.09820
# SWEEPS UNCOMMENT
config.learning_rate = 0.001311 #09084 #0.001     # learning rate
save_model = True

# Custom values to filter runs
# SWEEPS UNCOMMENT
config.training_stages = 2

# Data paths
path_data = Path('../../../../BigData/bdd100K/bdd100k/seg')
path_lbl = path_data / 'labels'
path_img = path_data / 'images'
#path_model = path_data / "mod" 

# Associate a label to an input
get_y_fn = lambda x: path_lbl / x.parts[-2] / f'{x.stem}_train_id.png'

# Load data into train & validation sets
src = (SegmentationItemList.from_folder(path_img).use_partial_data(0.02)
#src = (SegmentationItemList.from_folder(path_img)
       .split_by_folder(train='train', valid='val')
       .add_test_folder("test")
       .label_from_func(get_y_fn, classes=segmentation_classes))

# Resize, augment, load in batch & normalize (so we can use pre-trained networks)
data = (src.transform(get_transforms(), size=config.img_size, tfm_y=True)
        .databunch(bs=config.batch_size)
        .normalize(imagenet_stats))

config.num_train = len(data.train_ds)
config.num_valid = len(data.valid_ds)

########################################
# Accuracy metrics
#---------------------------------------
void_code = 19
# overall accuracy: across all classes, ignore unlabeled pixels
def acc(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    try:
      i = (input.argmax(dim=1)[mask] == target[mask]).float()
      m_i = i.mean()
      return m_i
    except:
      return torch.tensor([0.0])

def traffic_acc(input, target):
    target = target.squeeze(1)
    mask_pole = target == 5
    mask_light = target == 6
    mask_sign = taget = 7
    mask_traffic = mask_pole | mask_light | mask_sign
    try:
      i = (input.argmax(dim=1)[mask_traffic] == target[mask_traffic]).float()
      m_i = i.mean()
      return m_i
    except:
      return torch.tensor([0.0])

def road_acc(input, target):
    target = target.squeeze(1)
    mask = target == 0 
    try:
        intersection = input.argmax(dim=1)[mask] == target[mask]
        mean_intersection = intersection.float().mean()
        return mean_intersection
    except:
        return torch.tensor([0.0])


def car_acc(input, target):
    target = target.squeeze(1)
    mask = target == 13 
    try:
        intersection = input.argmax(dim=1)[mask] == target[mask]
        mean_intersection = intersection.float().mean()
        return mean_intersection
    except:
        return torch.tensor([0.0])

def human_acc(input, target):
    target = target.squeeze(1)
    mask_human = target == 11 
    #mask_rider = target == 12
    #mask_human = mask_person | mask_rider
    # this measures similarity of truth & guess on places where either has human pixels
    try:
        intersection = (input.argmax(dim=1)[mask_human] == target[mask_human]).float()
        mean_interesection = intersection.mean()
        print("GOT SOME HUMANS: ", mean_intersection)
        return mean_intersection
    except:
        return torch.tensor([0.0])

# cases we care about for human iou:
# 1. Truth: human, Guess: not human => most important
# 2. Truth: human, Guess: human => true positive, counts for accuracy as pixel intersection
# 3. Truth: not human, Guess: human => less important
# 4. Truth: not human, Guess: not human => true negative

########################################
# IoU metrics
#---------------------------------------
SMOOTH = 1e-6
def iou(input, target):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    target = target.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    intersection = (input.argmax(dim=1) & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (input.argmax(dim=1) | target).float().sum((1, 2))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    #return thresholded.mean()
    return iou.mean()

def human_iou(input, target):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    target = target.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    mask_human = target == 11
    intersection = (input.argmax(dim=1)[mask_human] == target[mask_human]).float()#.sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (input.argmax(dim=1)[mask_human] | target[mask_human]).float() #.sum((1, 2))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    #return thresholded.mean()
    return iou.mean()

# Create NN
learn = unet_learner(
    data,
    arch=encoder,
    pretrained=config.pretrained,
    metrics=[iou, acc, car_acc, traffic_acc, human_iou, road_acc],
    wd=config.weight_decay,
    bn_wd=config.bn_weight_decay,
    callback_fns=partial(WandbCallback, save_model=save_model, monitor='iou'))#, input_type='images'))

# load previous model and test data
unet_model = load_learner(path_img)

preds = unet_model.get_preds(ds_type=DataSetType.Test)
print("PREDS: ", preds)
exit()

#print("loaded: ", unet_model)
# get test data
# Load data into train & validation sets
#dt = ImageDataBunch.from_folder(path_img / "test", valid_pct=0, ds_tfms = get_transforms(), size=config.img_size, bs=config.batch_size).normalize(imagenet_stats) 
#print(dt)

#batch_masks = []

#test_batch = dt.train_dl
#print("test: ", test_batch)
#print(len(test_batch))
#for i, img in enumerate(test_batch):
        # log raw image as array
#  orig_image = img[0]
#  print("ORIG: ", orig_image)
#  bg_image = image2np(orig_image.data*255).astype(np.uint8)
        # our prediction
#  prediction = unet_model.predict(img[0])[0]
#  prediction_mask = image2np(prediction.data).astype(np.uint8)

        # ground truth
  #ground_truth = img[1]
  #true_mask = image2np(ground_truth.data).astype(np.uint8)

#batch_masks.append(wb_mask(bg_image, prediction_mask)) # true_mask))

#wandb.log({"test preditions" : batch_masks})

# Train
if config.one_cycle:
    learn.fit_one_cycle(
        config.epochs // 2,
        max_lr=slice(config.learning_rate),
        callbacks=[LogImagesCallback(learn)])
    learn.unfreeze()
    learn.fit_one_cycle(
        config.epochs // 2,
        max_lr=slice(config.learning_rate / 100,
                     config.learning_rate / 10),
        callbacks=[LogImagesCallback(learn)])
else:
    learn.fit(
        config.epochs,
        lr=slice(config.learning_rate))


# try to save learner
learn.export()
path = learn.save('final_model')
print(path)
