import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import pickle
import os
import os.path as osp
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

from deeplab.model import Res_Deeplab
from deeplab.datasets import DataSetTest
from collections import OrderedDict

from util.utils import label_img_to_color, apllo_lbl_to_bdd, convert_apollo_to_bdd
from util.bdd_evaluate import evaluate_segmentation

import matplotlib.pyplot as plt
import torch.nn as nn

GPU = 0
DATA_DIRECTORY = '../bdd100k/seg/'
IGNORE_LABEL = 255
NUM_CLASSES = 19
BATCH_SIZE = 1
NUM_TRAIN_FILES = 1938
NUM_STEPS = NUM_TRAIN_FILES/BATCH_SIZE # Number of images in the validation set.
RESTORE_FROM = ('./checkpoints/BDD_Train_Completed_(DA).pkl', './checkpoints/BDD_Train_Completed_500_(Normal).pkl')
TITLE = ('Domain Adaption', 'Normal')

PRED_SAVE_DIR = ('./results/DA_2/pred/','./results/Normal_2/pred/')
LBL_SAVE_DIR = ('./results/DA_2/label/','./results/Normal_2/label/')
LBL_CHG_SAVE_DIR = './results/label_changed/'

VISUAL_SAVE_DIR = "./results/"
VISUAL_COLOR=""
VISUAL_PRED=""
VISUAL_OVERLAY=""

IMAGE_SIZE = (1280, int(2710/3382*1280))
TEST_MEAN_PATH = "../bdd100k/seg/apollo_mean.pkl"
TEST_IMG_MEAN = 0

CLASS_NAMES = ('Road','Sidewalk','Building','Wall','Fence',
    'Pole', 'Traffic Light', 'Traffic Sign', 'Vegetation', 
    'Terrain', 'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus',
    'Train', 'Motorcycle', 'Bicycle')

def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool 
    from deeplab.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list)+'\n')
            f.write(str(M)+'\n')



def create_save_visualisation(imgs, pred, img_ids, overlay=False):
    ########################################################################
    # save data for visualization:
    ########################################################################
    print("Starting Visualisation")
    pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
    pred_label_imgs = pred_label_imgs.astype(np.uint8)

    color_dir = osp.join(VISUAL_SAVE_DIR, VISUAL_COLOR)
    pred_dir = osp.join(VISUAL_SAVE_DIR, VISUAL_PRED)
    overlay_dir = osp.join(VISUAL_SAVE_DIR, VISUAL_OVERLAY)

    if not os.isdir(color_dir):
        os.mkdir(color_dir)
        os.mkdir(pred_dir)
        os.mkdir(overlay_dir)

    for i in range(pred_label_imgs.shape[0]):
        if i == 0:
            pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
            img_id = img_ids[i]
            img = imgs[i] # (shape: (3, img_h, img_w))

            output = output.transpose(1,2,0)
            cv2.imwrite(osp.join(pred_dir, img_id+'.png'), pred_label_img)

            # Save color pred
            if overlay:
                pred_label_img_color = label_img_to_color(pred_label_img)
                cv2.imwrite(osp.join(color_dir, img_id), pred_label_img_color)

                img = img.data.cpu().numpy()
                img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                img += TEST_IMG_MEAN
                img = img.astype(np.uint8)

                overlayed_img = 0.45*img + 0.55*pred_label_img_color
                overlayed_img = overlayed_img.astype(np.uint8)
                cv2.imwrite(osp.join(overlay_dir, img_id + "_overlayed.png"), overlayed_img)

def save_results(labels, pred, img_ids):
    print("Starting Visualisation!")
    outputs = pred.data.cpu().numpy()
    pred_label

# Create Video From computed images
def createVideo(img_h, img_w):
    imgDir = osp.join(DATA_DIRECTORY, "images/apollo_test")

    out = cv2.VideoWriter("%s/video_combined.avi" % (VISUAL_SAVE_DIR), cv2.VideoWriter_fourcc(*"MJPG"), 20, (2*img_w, 2*img_h))
    for img_id in tqdm(os.listdir(imgDir)):
        
        img = cv2.imread(osp.join(imgDir,img_id), -1)
        img_id = img_id[:-4]

        pred_img = cv2.imread(osp.join(VISUAL_SAVE_DIR, img_id + "_pred.png"), -1)
        overlayed_img = cv2.imread(osp.join(VISUAL_SAVE_DIR_2, img_id + "_overlayed.png"), -1)


        img = cv2.resize(img, (img_w, img_h))

        combined_img = np.zeros((2*img_h, 2*img_w, 3), dtype=np.uint8)
        combined_img[0:img_h, 0:img_w] = img
        combined_img[0:img_h, img_w:(2*img_w)] = pred_img
        combined_img[img_h:(2*img_h), (int(img_w/2)):(img_w + int(img_w/2))] = overlayed_img

        out.write(combined_img)

    out.release()

def main():

    """Create the model and start the evaluation process."""
    # gpu0 = GPU
    # torch.cuda.empty_cache()

    # model = Res_Deeplab(num_classes=NUM_CLASSES)
    
    # for i, res in enumerate(RESTORE_FROM):
    #     checkpoint = torch.load(res)
    #     model.load_state_dict(checkpoint['model_state_dict'])

    #     # Load Dataset Means & Train Dataset Class Weights
    #     if osp.isfile(TEST_MEAN_PATH):
    #         with open(TEST_MEAN_PATH, "rb") as file:
    #             TEST_IMG_MEAN = np.array(pickle.load(file))
    #     else:
    #         print("Please run the preprocess_data.py file in utils first!")
    #         print("Exiting Training!")
    #         return

    #     model.eval()
    #     model.cuda(gpu0)

    #     testloader = data.DataLoader(DataSetTest(DATA_DIRECTORY, max_iters=None, crop_size=IMAGE_SIZE, mean=TEST_IMG_MEAN), 
    #                 batch_size=BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True)

    #     data_list = []
    #     for step, (imgs, label, img_ids) in tqdm(enumerate(testloader)):
    #         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)

    #             print(TITLE[i]," Step: ",step,"/",len(testloader))
    #             imgs = Variable(imgs).cuda(gpu0)

    #             pred = nn.functional.interpolate((model(imgs)),size= (int(2710/3382*1280),1280), mode='bilinear', align_corners=True)
    #             output = np.argmax(F.softmax(pred, dim=1).cpu().data[0].numpy(),axis=0)

    #             lbl = np.asarray(label[0].numpy(), dtype=np.int32)

    #             cv2.imwrite(osp.join(PRED_SAVE_DIR[i], img_ids[0] + ".png"), output)

    #             data_list.append([lbl.flatten(), output.flatten()])
    #             # cv2.imwrite(osp.join(LBL_SAVE_DIR, img_ids[0] + ".png"), np.asarray(label[0], np.uint8))
    #             if i == 0:
    #                 apllo_lbl_to_bdd(label[0],osp.join(LBL_SAVE_DIR[i], img_ids[0] + ".png"))
    #                 print("Saving BDD Label")
    #             else:
    #                 cv2.imwrite(osp.join(LBL_SAVE_DIR[i], img_ids[0] + ".png"), lbl)


    #     with open("./results/data_list_"+TITLE[i]+".pkl", "wb") as file:
    #         pickle.dump(data_list, file, protocol=2)
        

        # convert_apollo_to_bdd(LBL_SAVE_DIR, LBL_CHG_SAVE_DIR)
                
    # miou, ious = evaluate_segmentation("./results/DA_2/label/","./results/DA_2/pred/",19,200)
    # print("DA: ",miou)

    # p1 = plt.barh(np.arange(len(CLASS_NAMES)), ious, align='center', color="red")
    # miou, ious = evaluate_segmentation("./results/DA_2/label/","./results/Normal_2/pred/",19,200)
    # print("Normal: ",miou)
    # p2 = plt.barh(np.arange(len(CLASS_NAMES)), ious, align='center', color="black")

    # plt.title("IoU per Class")
    # plt.ylabel('Classes')
    # plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    # plt.xticks(np.arange(0,100,step=10))
    # plt.xlabel('IoU')
    # plt.legend((p1[0],p2[0]), ('Domain Adaption', 'Deeplabv3'))
    # plt.savefig("./results/NormalsvDA_Bar.png")
    # plt.show()

    img = cv2.imread("C:/Users/Home/Desktop/Test Pictures/170927_071013518_Camera_5.jpg", -1)
    pred = cv2.imread("C:/Users/Home/Desktop/Test Pictures/170927_071013518_Camera_5.png", -1)
    img = np.asarray(img, np.float32)

    pred_label_img = pred.astype(np.uint8)
    print("Starting Colouring")
    pred_label_img_color = label_img_to_color(pred_label_img)
    print("End Colouring")
    cv2.imwrite("C:/Users/Home/Desktop/Test Pictures/170927_071013518_Camera_5_color.png", pred_label_img_color)
    
    overlayed_img = 0.45*img + 0.55*pred_label_img_color
    overlayed_img = overlayed_img.astype(np.uint8)
    cv2.imwrite("C:/Users/Home/Desktop/Test Pictures/170927_071013518_Camera_5_overlay.png", overlayed_img)



    create_save_visualisation((img),(pred),("170927_074018569_Camera_5"),overlay = True)

    # createVideo(576, 720)

    # data_list = []

    # for index, batch in enumerate(testloader):
    #     if index % 100 == 0:
    #         print('%d processd'%(index))
    #     image, label, size, name = batch
    #     size = size[0].numpy()
    #     output = model(Variable(image, volatile=True).cuda(gpu0))
    #     output = interp(output).cpu().data[0].numpy()

    #     output = output[:,:size[0],:size[1]]
    #     gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
        
    #     output = output.transpose(1,2,0)
    #     output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

    #     # show_all(gt, output)
    #     data_list.append([gt.flatten(), output.flatten()])

    # get_iou(data_list, args.num_classes)


if __name__ == '__main__':
    main()
