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
from deeplab.datasets import DataSetTrain, DataSetVal, DataSetTarget
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import cycle
import timeit

# Dataset Variables
BATCH_SIZE = 5
INPUT_SIZE = (256,256)
IGNORE_LABEL = 255
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_TRAIN_FILES = 7000
NUM_TEST_FILES = 1900
NUM_STEPS = NUM_TRAIN_FILES/BATCH_SIZE
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_EVERY = 1

#Transform Params
RANDOM_SCALE = True
RANDOM_FLIP = True

WEIGHT_DECAY = 0.0005
EPOCHS = 500

# Maximum Mean Discrepancy Hyper Parameter
MMD_LAMDA = 0.25

# Paths
DATA_DIRECTORY = '../bdd100k/seg/'
TEST_MEAN_PATH = "../bdd100k/seg/test_mean.pkl"
TRAIN_MEAN_PATH = "../bdd100k/seg/train_mean.pkl"
CLASS_WEIGHT_PATH = "../bdd100k/seg/class_weights.pkl"
PRETRAIN_RESTORE_PATH = './pretrain/MS_DeepLab_resnet_pretrained_COCO_init.pth'
CHECKPOINT_DIR = './checkpoints/'

# Specify as NONE if not resuming
CHECKPOINT_RESTORE_PATH = './checkpoints/BDD_Train_Completed.pkl'

def loss_calc(pred, label, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights,ignore_index=IGNORE_LABEL).cuda()
    return criterion(pred, label)

# Adjsting based on polynomial decay lr based on the epochs becomes 0 by last iteration
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
    lr = lr_poly(LEARNING_RATE, i_iter, EPOCHS*NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


def mmd_linear(f_of_X, f_of_Y):
    meanX = torch.mean(f_of_X, (3))
    meanY = torch.mean(f_of_Y, (3))
    meanX = torch.mean(meanX, (2))
    meanY = torch.mean(meanY, (2))

    delta = meanX - meanY
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean

def main():
    # Load Dataset Means & Train Dataset Class Weights
    if os.path.isfile(TRAIN_MEAN_PATH):
        with open(TRAIN_MEAN_PATH, "rb") as file:
            TRAIN_IMG_MEAN = np.array(pickle.load(file))

        with open(TEST_MEAN_PATH, "rb") as file:
            TEST_IMG_MEAN = np.array(pickle.load(file))

        with open(CLASS_WEIGHT_PATH, "rb") as file:
            CLASS_WEIGHTS = torch.Tensor(pickle.load(file))

    else:
        print("Please run the preprocess_data.py file in utils first!")
        print("Exiting Training!")
        return
    
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

    model.train()
    model.cuda()
    
    cudnn.benchmark = True

    CHECKPOINT_SAVEPATH = CHECKPOINT_DIR+"/DC_"+str(EPOCHS)+"_models/"

    #Create Checkpoint Directory if not found
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    if not os.path.exists(CHECKPOINT_SAVEPATH):
        os.makedirs(CHECKPOINT_SAVEPATH)

    trainloader = data.DataLoader(DataSetTrain(DATA_DIRECTORY, max_iters=NUM_TRAIN_FILES, crop_size=input_size, 
                    scale=RANDOM_SCALE, mirror=RANDOM_FLIP, mean=TRAIN_IMG_MEAN), 
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

    testloader = data.DataLoader(DataSetTarget(DATA_DIRECTORY, max_iters=NUM_TEST_FILES, crop_size=input_size, 
                scale=RANDOM_SCALE, mirror=RANDOM_FLIP, mean=TEST_IMG_MEAN, train=True), 
                batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

    # Cyclically Iterate through testloader as testloader has lesser batches than trainloader
    test_iter = cycle(testloader)

    #Optimizer
    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': LEARNING_RATE }, 
                {'params': get_10x_lr_params(model), 'lr': 10*LEARNING_RATE}], 
                lr=LEARNING_RATE, momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    
    optimizer.zero_grad()

    train_epoch_loss = []
    start_epoch = 0

    # Resume Training
    if CHECKPOINT_RESTORE_PATH is not None:
        checkpoint = torch.load(CHECKPOINT_RESTORE_PATH)
        start_epoch = int(checkpoint['epoch'])+1
        print("Resuming Training! Current Epoch: ",start_epoch)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_epoch_loss = checkpoint['train_epoch_loss']
    
    # Restore first 4 layers from pre-trained Model for quicker training
    else:
        saved_state_dict = torch.load(PRETRAIN_RESTORE_PATH)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not NUM_CLASSES == 19 or not i_parts[1]=='layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)

    print ("Starting Training!")
    print ("==================")
    start = timeit.default_timer()
    """Create the model and start the training."""
    for epoch in tqdm(range(start_epoch, EPOCHS)):

        #restore loss function
        #if start_epoch > 0 and epoch == start_epoch:
            #loss = checkpoint['loss']
 
        batch_losses = []

        for i_iter, batch in enumerate(trainloader):
            images, labels, _ = batch
            test_images, _ = next(test_iter)

            images = Variable(images).cuda()
            test_images = Variable(test_images).cuda()

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, epoch*i_iter)

            pred = model(images)
            target = model(test_images)

            mmd = mmd_linear(pred,target)

            pred =  nn.functional.interpolate(pred,size=input_size, mode='bilinear', align_corners=True)

            # pred, mmd_loss =  model(images,test_images)

            # print("MMD_LOSS: "+str((MMD_LAMDA*mmd_linear(pred,target)).data.cpu().numpy()))

            loss = loss_calc(pred, labels, CLASS_WEIGHTS) + mmd

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.data.cpu().numpy())
            if i_iter%(NUM_STEPS/5)==0:
                print('Batch ', i_iter, 'of', NUM_STEPS,' completed, loss = ', loss.data.cpu().numpy())

        epoch_loss = np.mean(batch_losses)
        train_epoch_loss.append(epoch_loss)
        print('epoch ', epoch, 'of', EPOCHS,' completed, loss = ', epoch_loss)

        #Removing Previous Saved Model
        if epoch >= EPOCHS-1 or (epoch % SAVE_EVERY == 0 and epoch!=0) and len(os.listdir(CHECKPOINT_SAVEPATH)) > 0:
            os.unlink(osp.join(CHECKPOINT_SAVEPATH,os.listdir(CHECKPOINT_SAVEPATH)[-1]))
            
        #Saving Model
        if epoch >= EPOCHS-1:
            print ('Training Completed Saving Model ...')
            torch.save({'epoch': epoch,
                    'train_epoch_loss':train_epoch_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, osp.join(CHECKPOINT_SAVEPATH, 'BDD_Train_Completed.pkl'))

        elif epoch % SAVE_EVERY == 0 and epoch!=0:
            print ('taking checkpoint ...')
            torch.save({'epoch': epoch,
                    'train_epoch_loss':train_epoch_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, osp.join(CHECKPOINT_SAVEPATH, 'BDD_Train_'+str(epoch)+'.pkl'))    

    end = timeit.default_timer()
    total_time = end-start
    print (total_time,'seconds')

    savedata ={
    'epoch': EPOCHS,
    'total_time':total_time,
    'train_epoch_loss':train_epoch_loss,
    'time_per_epoch':total_time/EPOCHS,
    }

    
    # save
    with open(CHECKPOINT_DIR+"/completetion_params_"+str(EPOCHS)+".pkl", "wb") as file:
        pickle.dump(savedata, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)

    plt.figure(1)
    plt.plot(range(EPOCHS), train_epoch_loss)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Train Loss/Epoch")
    plt.savefig("%s/epoch_losses_train.png" % CHECKPOINT_DIR)
    plt.close(1)

if __name__ == '__main__':
    main()
