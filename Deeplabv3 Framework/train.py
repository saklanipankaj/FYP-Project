from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

from model.deeplabv3 import DeepLabV3

from utils.utils import add_weight_decay

import torch
import os
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2

import time

# NOTE! NOTE! change this to not overwrite all log data when you train the model:
model_id = "1"

num_epochs = 1000
batch_size = 3
learning_rate = 0.0001

network = DeepLabV3(model_id, project_dir="./deeplabv3", resnet="ResNet18_OS8").cuda()

train_dataset = DatasetTrain(root="../bdd100k/seg/")
val_dataset = DatasetVal(root="../bdd100k/seg/")

num_train_batches = int(len(train_dataset)/batch_size)
num_val_batches = int(len(val_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)
print ("num_val_batches:", num_val_batches)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

params = add_weight_decay(network, l2_value=0.0001)
optimizer = torch.optim.Adam(params, lr=learning_rate)

with open("../bdd100k/seg/class_weights.pkl", "rb") as file: # (needed for python3)
    class_weights = np.array(pickle.load(file))
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# loss function
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

epoch_losses_train = []
epoch_losses_val = []

resume = True
if len(os.listdir(network.checkpoints_dir)) >0:
    checkpoint_path = network.checkpoints_dir + os.listdir(network.checkpoints_dir)[-1]
    checkpoint = torch.load(checkpoint_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_losses_train = checkpoint['epoch_losses_train']
    epoch_losses_val = checkpoint['epoch_losses_val']
    loss = checkpoint['loss']
else:
    resume = False

for epoch in range(num_epochs):

    if resume:
        epoch = checkpoint['epoch']+1

    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    ############################################################################
    # train:
    ############################################################################
    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label, img_ids) in enumerate(train_loader):
        #current_time = time.time()

        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
        label = Variable(label.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        # compute the loss:
        loss = loss_fn(outputs, label)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        # optimization step:
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)

        #print (time.time() - current_time)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)

    ############################################################################
    # val:
    ############################################################################
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label, img_ids) in enumerate(val_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
            label = Variable(label.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

            outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            loss = loss_fn(outputs, label)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)

    # save the model weights to disk:
    if epoch % 20 ==0:
        checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save({'epoch': epoch,
                    'epoch_losses_train':epoch_losses_train,
                    'epoch_losses_val': epoch_losses_val,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, checkpoint_path)


print ("train loss: %g" % epoch_loss)
plt.figure(1)
plt.plot(epoch_losses_train, "k^")
plt.plot(epoch_losses_train, "k")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("train loss per epoch")
plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
plt.close(1)

print ("val loss: %g" % epoch_loss)
plt.figure(1)
plt.plot(epoch_losses_val, "k^")
plt.plot(epoch_losses_val, "k")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("val loss per epoch")
plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
plt.close(1)
