# Util file for common methods
# Created by Saklani Pankaj

import torch
import torch.nn as nn
import numpy as np
import cv2

from collections import namedtuple

# a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',
    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',
    # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',
    # Whether this label distinguishes between single instances or not

    'ignoreInEval',
    # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

# Our extended list of label types. Our train id is compatible with Cityscapes
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , [  0,  0,  0] ),
    Label(  'dynamic'              ,  1 ,      255 , 'void'            , 0       , False        , True         , [111, 74,  0] ),
    Label(  'ego vehicle'          ,  2 ,      255 , 'void'            , 0       , False        , True         , [  0,  0,  0] ),
    Label(  'ground'               ,  3 ,      255 , 'void'            , 0       , False        , True         , [ 81,  0, 81] ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , [  0,  0,  0] ),
    Label(  'parking'              ,  5 ,      255 , 'flat'            , 1       , False        , True         , [250,170,160] ),
    Label(  'rail track'           ,  6 ,      255 , 'flat'            , 1       , False        , True         , [230,150,140] ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , [128, 64,128] ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , [244, 35,232] ),
    Label(  'bridge'               ,  9 ,      255 , 'construction'    , 2       , False        , True         , [150,100,100] ),
    Label(  'building'             , 10 ,        2 , 'construction'    , 2       , False        , False        , [ 70, 70, 70] ),
    Label(  'fence'                , 11 ,        4 , 'construction'    , 2       , False        , False        , [190,153,153] ),
    Label(  'garage'               , 12 ,      255 , 'construction'    , 2       , False        , True         , [180,100,180] ),
    Label(  'guard rail'           , 13 ,      255 , 'construction'    , 2       , False        , True         , [180,165,180] ),
    Label(  'tunnel'               , 14 ,      255 , 'construction'    , 2       , False        , True         , [150,120, 90] ),
    Label(  'wall'                 , 15 ,        3 , 'construction'    , 2       , False        , False        , [102,102,156] ),
    Label(  'banner'               , 16 ,      255 , 'object'          , 3       , False        , True         , [250,170,100] ),
    Label(  'billboard'            , 17 ,      255 , 'object'          , 3       , False        , True         , [220,220,250] ),
    Label(  'lane divider'         , 18 ,      255 , 'object'          , 3       , False        , True         , [255, 165, 0] ),
    Label(  'parking sign'         , 19 ,      255 , 'object'          , 3       , False        , False        , [220, 20, 60] ),
    Label(  'pole'                 , 20 ,        5 , 'object'          , 3       , False        , False        , [153,153,153] ),
    Label(  'polegroup'            , 21 ,      255 , 'object'          , 3       , False        , True         , [153,153,153] ),
    Label(  'street light'         , 22 ,      255 , 'object'          , 3       , False        , True         , [220,220,100] ),
    Label(  'traffic cone'         , 23 ,      255 , 'object'          , 3       , False        , True         , [255, 70,  0] ),
    Label(  'traffic device'       , 24 ,      255 , 'object'          , 3       , False        , True         , [220,220,220] ),
    Label(  'traffic light'        , 25 ,        6 , 'object'          , 3       , False        , False        , [250,170, 30] ),
    Label(  'traffic sign'         , 26 ,        7 , 'object'          , 3       , False        , False        , [220,220,  0] ),
    Label(  'traffic sign frame'   , 27 ,      255 , 'object'          , 3       , False        , True         , [250,170,250] ),
    Label(  'terrain'              , 28 ,        9 , 'nature'          , 4       , False        , False        , [152,251,152] ),
    Label(  'vegetation'           , 29 ,        8 , 'nature'          , 4       , False        , False        , [107,142, 35] ),
    Label(  'sky'                  , 30 ,       10 , 'sky'             , 5       , False        , False        , [ 70,130,180] ),
    Label(  'person'               , 31 ,       11 , 'human'           , 6       , True         , False        , [220, 20, 60] ),
    Label(  'rider'                , 32 ,       12 , 'human'           , 6       , True         , False        , [255,  0,  0] ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , [119, 11, 32] ),
    Label(  'bus'                  , 34 ,       15 , 'vehicle'         , 7       , True         , False        , [  0, 60,100] ),
    Label(  'car'                  , 35 ,       13 , 'vehicle'         , 7       , True         , False        , [  0,  0,142] ),
    Label(  'caravan'              , 36 ,      255 , 'vehicle'         , 7       , True         , True         , [  0,  0, 90] ),
    Label(  'motorcycle'           , 37 ,       17 , 'vehicle'         , 7       , True         , False        , [  0,  0,230] ),
    Label(  'trailer'              , 38 ,      255 , 'vehicle'         , 7       , True         , True         , [  0,  0,110] ),
    Label(  'train'                , 39 ,       16 , 'vehicle'         , 7       , True         , False        , [  0, 80,100] ),
    Label(  'truck'                , 40 ,       14 , 'vehicle'         , 7       , True         , False        , [  0,  0, 70] ),
]


conversion = namedtuple('conversion', [
    'name',
    'apollo',
    'bdd',
])

conversions = [
    conversion('others',  0 ,  255),
    conversion('rover'  , 1  , 255),
    conversion('sky', 17 , 10),
    conversion('car', 33 , 13),
    conversion('car_groups'  ,161, 13),
    conversion('motorbicycle'  ,  34 , 17),
    conversion('motorbicycle_group'  ,162 ,17),
    conversion('bicycle' ,35 , 18),
    conversion('bicycle_group' , 163 ,18),
    conversion('person' , 36 , 11),
    conversion('person_group'  ,  164 ,11),
    conversion('rider' ,  37 , 12),
    conversion('rider_group' ,165,12),
    conversion('truck' ,  38 , 14),
    conversion('truck_group' ,166 ,14),
    conversion('bus' ,39 , 15),
    conversion('bus_group'  , 167, 15),
    conversion('tricycle'  ,  40 , 255),
    conversion('tricycle_group' , 168 ,255),
    conversion('road'   , 49 , 0),
    conversion('siderwalk'   ,50 , 1),
    conversion('traffic_cone'  ,  65 , 255),
    conversion('road_pile'  , 66 , 0),
    conversion('fence' ,  67 , 4),
    conversion('traffic_light'  , 81 , 6),
    conversion('pole'  ,  82,  5),
    conversion('traffic_sign'  ,  83 , 7),
    conversion('wall'  ,  84 , 3),
    conversion('dustbin', 85 , 255),
    conversion('billboard' ,  86 , 255),
    conversion('building' ,   97 , 2),
    conversion('bridge',  98,  255),
    conversion('tunnel',  99 , 255),
    conversion('overpass'  , 100, 255),
    conversion('vegetation' , 113, 8),
    conversion('unlabeled',  255, 255)
]

# returns the converted labelid from apollo to bdd dataset
def apollo_to_bdd(trainId):
    for conversion in conversions:
        if conversion.apollo == trainId:
            return conversion.bdd


# returns the color based on the trainId
def trainId_to_color(trainId):
    for label in labels:
        if label.trainId == trainId:
            return label.color

# returns color representation of a label
def label_img_to_color(img):
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            trainId = img[row, col]
            img_color[row, col] = np.array(trainId_to_color(trainId))

    return img_color


# return colourmap for all the labels
def get_colormaps():
    colormap = []
    for label in labels:
        colormap.append(label.color)

    return colormap

# returns the weight decays to various params in the network
def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


def apllo_label_to_bdd(labelPath, savePath):
    label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)
    label = np.asarray(label, np.float32)
    conv_label = np.zeros(label.shape, dtype=np.uint8)

    for r in range(label.shape[0]):
        for c in range(label.shape[1]):
            conv_label[r,c] = apollo_to_bdd(label[r,c])

    cv2.imwrite(savePath, conv_label)

def apllo_lbl_to_bdd(lbl, savePath):
    label = np.asarray(lbl, np.float32)
    conv_label = np.zeros(label.shape, dtype=np.uint8)
    # Iterate through all pixel semantic
    for r in range(label.shape[0]):
        for c in range(label.shape[1]):
            conv_label[r,c] = apollo_to_bdd(label[r,c])

    cv2.imwrite(savePath, conv_label)

def convert_apollo_to_bdd(labelDir,saveDir):
    for label in tqdm(os.listdir(labelDir)):
        labelPath = osp.join(labelDir,label)
        savePath = osp.join(saveDir,label)
        apllo_label_to_bdd(labelPath,savePath)


