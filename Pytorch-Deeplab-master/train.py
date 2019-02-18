import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from deeplab.model import Res_Deeplab
from deeplab.datasets import DataSetTrain, DataSetVal
import matplotlib.pyplot as plt
import timeit

BATCH_SIZE = 5
INPUT_SIZE = (321,321)
DATA_DIRECTORY = '../bdd100k/seg/'
IGNORE_LABEL = 255
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_TRAIN_FILES = 7000
NUM_STEPS = NUM_TRAIN_FILES/BATCH_SIZE
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './pretrain/MS_DeepLab_resnet_pretrained_COCO_init.pth'
SAVE_NUM_IMAGES = 2
SAVE_EVERY = 1
CHECKPOINT_DIR = './checkpoints/'
WEIGHT_DECAY = 0.0005
EPOCHS = 200

#Transform Params
RANDOM_SCALE = True
RANDOM_FLIP = True

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()
    
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
            
            
def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(LEARNING_RATE, i_iter, EPOCHS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


def main():
    with open("../bdd100k/seg/img_mean.pkl", "rb") as file: # (needed for python3)
        IMG_MEAN = np.array(pickle.load(file))

    start = timeit.default_timer()
    """Create the model and start the training."""
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
    # h, w = map(int, args.input_size.split(','))
    input_size = INPUT_SIZE

    cudnn.enabled = True

    # Create network.
    model = Res_Deeplab(num_classes=NUM_CLASSES)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    
    
    saved_state_dict = torch.load(RESTORE_FROM)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if not NUM_CLASSES == 19 or not i_parts[1]=='layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    model.load_state_dict(new_params)
    #model.float()
    #model.eval() # use_global_stats = True
    model.train()
    model.cuda()
    
    cudnn.benchmark = True

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)


    trainloader = data.DataLoader(DataSetTrain(DATA_DIRECTORY, max_iters=NUM_STEPS*BATCH_SIZE, crop_size=input_size, 
                    scale=RANDOM_SCALE, mirror=RANDOM_FLIP, mean=IMG_MEAN), 
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

    #Optimizer
    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': LEARNING_RATE }, 
                {'params': get_10x_lr_params(model), 'lr': 10*LEARNING_RATE}], 
                lr=LEARNING_RATE, momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    optimizer.zero_grad()

    #Upsampling Layer
    interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    train_epoch_loss = []
    start_epoch = 0

    # Resume Training
    if len(os.listdir(CHECKPOINT_DIR)) > 0:
        checkpoint_path = CHECKPOINT_DIR + os.listdir(CHECKPOINT_DIR)[-1]
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.keys())
        start_epoch = int(checkpoint['epoch'])+1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_epoch_loss = checkpoint['train_epoch_loss']
    

    print ("Starting Training!")
    print ("==================")
    for epoch in range(start_epoch, EPOCHS):

        #restore loss function
        if start_epoch > 0 and epoch == start_epoch:
            loss = checkpoint['loss']
 
        batch_losses = []
        for i_iter, batch in enumerate(trainloader):
            images, labels, _, _ = batch
            images = Variable(images).cuda()

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, epoch)
            pred =  nn.functional.interpolate((model(images)),size=input_size, mode='bilinear', align_corners=True)
            loss = loss_calc(pred, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.data.cpu().numpy())
            if i_iter%100==0:
                print('Batch ', i_iter, 'of', NUM_STEPS,' completed, loss = ', loss.data.cpu().numpy())

        epoch_loss = np.mean(batch_losses)
        train_epoch_loss.append(epoch_loss)
        print('epoch ', epoch, 'of', EPOCHS,' completed, loss = ', epoch_loss)

        #Removing Previous Saved Model
        if epoch >= EPOCHS-1 or (epoch % SAVE_EVERY == 0 and epoch!=0) and len(os.listdir(CHECKPOINT_DIR)) > 0:
            os.unlink(osp.join(CHECKPOINT_DIR,os.listdir(CHECKPOINT_DIR)[-1]))
            
        #Saving Model
        if epoch >= EPOCHS-1:
            print ('Training Completed Saving Model ...')
            torch.save({'epoch': epoch,
                    'train_epoch_loss':train_epoch_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, osp.join(CHECKPOINT_DIR, 'BDD_Train_Completed.pkl'))

        elif epoch % SAVE_EVERY == 0 and epoch!=0:
            print ('taking checkpoint ...')
            torch.save({'epoch': epoch,
                    'train_epoch_loss':train_epoch_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, osp.join(CHECKPOINT_DIR, 'BDD_Train_Completed.pkl'))    

    end = timeit.default_timer()
    print (end-start,'seconds')
    plt.figure(1)
    plt.plot(range(EPOCHS), train_epoch_loss)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Train Loss Per Epoch")
    plt.savefig("%s/epoch_losses_train.png" % CHECKPOINT_DIR)
    plt.close(1)

if __name__ == '__main__':
    main()
