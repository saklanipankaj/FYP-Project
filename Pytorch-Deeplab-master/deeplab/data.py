import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data

if __name__ == '__main__':
    wd = os.getcwd()

    root = "../../bdd100k/seg/"
    imgPath = osp.join(root, "images/train")
    labelPath = osp.join(root,"labels/train");
    count = 0
    for img in os.listdir(imgPath):
        img_file = osp.join(imgPath, img)
        #remove .png extension from the image
        name = img[:-4]
        label = "/%s_train_id.png"%name
        label_file = osp.join(labelPath,label)
    
