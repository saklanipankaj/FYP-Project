import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from model.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from model.aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):

    def __init__(self, model_id, project_dir, resnet):
        super(DeepLabV3, self).__init__()

        self.num_classes = 19

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        if resnet == "ResNet18_OS8":
            self.resnet = ResNet18_OS8()

        elif resnet == "ResNet34_OS8":
            self.resnet = ResNet34_OS8()

        elif resnet == "ResNet18_OS16":
            self.resnet = ResNet18_OS16()

        elif resnet == "ResNet34_OS16":
            self.resnet = ResNet34_OS16()

        elif resnet == "ResNet50_OS16":
            self.resnet = ResNet50_OS16()

        elif resnet == "ResNet101_OS16":
            self.resnet = ResNet101_OS16()

        elif resnet == "ResNet152_OS16":
            self.resnet = ResNet152_OS16()

        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
        print(feature_map.shape)
        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output


    # Create Model Directories if not created yet
    def create_model_dirs(self):
        self.model_dir = self.project_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
