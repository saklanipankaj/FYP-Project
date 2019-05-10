# Preprocess Data 
# Created By Saklani Pankaj
import pickle
import numpy as np
import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
# from util.utils import apollo_to_bdd
from tqdm import tqdm

from utils import label_img_to_color, apllo_lbl_to_bdd, convert_apollo_to_bdd


def dataset_mean(datasetPath, savepath, name):
    mean_arr=[]

    # Enumerate through the image directory
    for num, img_path in enumerate(os.listdir(datasetPath)):
        completed = num/len(os.listdir(datasetPath))*100
        if completed%10 == 0:
            print ("Computing %s Mean: %0.2f%% Completed"%(str(name),completed))

        img = cv2.imread(osp.join(datasetPath,img_path), -1)
        mean_arr.append(np.mean(img,axis=0))

    mean = np.mean(mean_arr,axis=0)
    mean = np.mean(mean,axis=0)
    print(mean)

    # save class weights to file
    with open(savepath, "wb") as file:
        pickle.dump(mean, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)

def dataset_sd(datasetPath, savepath,name, mean):
    arr=[]
    count = 0
    total = 0
    # Enumerate through the image directory
    for num, img_path in tqdm(enumerate(os.listdir(datasetPath))):
        completed = num/len(os.listdir(datasetPath))*100
        if completed%10 == 0:
            print ("Computing %s SD: %0.2f%% Completed"%(str(name),completed))
        img = cv2.imread(osp.join(datasetPath,img_path), -1)

        for i in range(img.shape[0]):
            for r in range(img.shape[1]):
                total += abs(img[i,r] - mean)**2
                count+=1
    print(count)
    std = (total/count)**0.5
    print(std)

    # save class weights to file
    with open(savepath, "wb") as file:
        pickle.dump(mean, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)

def get_class_weights(datasetPath):
    num_classes = 19

    trainId_to_count = {}
    for trainId in range(num_classes):
        trainId_to_count[trainId] = 0

    # get the total number of pixels in all train label_imgs that are of each object class:
    for num, label_path in enumerate(os.listdir(datasetPath)):
        completed = num/len(os.listdir(datasetPath))*100
        if int(completed)%10 == 0:
            print ("Computing weights: {0:.2f}% Completed".format(completed))

        label_img = cv2.imread(datasetPath+label_path, -1)

        for trainId in range(num_classes):
            # count how many pixels in label_img which are of object class trainId:
            trainId_mask = np.equal(label_img, trainId)
            trainId_count = np.sum(trainId_mask)

            # add to the total count:
            trainId_to_count[trainId] += trainId_count

    # compute the class weights according to the ENet paper:
    class_weights = []
    total_count = sum(trainId_to_count.values())
    for trainId, count in trainId_to_count.items():
        trainId_prob = float(count)/float(total_count)
        trainId_weight = 1/np.log(1.02 + trainId_prob)
        class_weights.append(trainId_weight)

    print (class_weights)
    print("Completed Class Weight Calculation")

    # save class weights to file
    with open("../../bdd100k/seg/class_weights.pkl", "wb") as file:
        pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)


# Generating mean of images
if __name__ == '__main__':

    # dataset_mean("C:/Users/Home/Desktop/Camera 5/images/test/", "../../bdd100k/seg/apollo_test_mean.pkl", "ApolloTest")
    # dataset_sd("C:/Users/Home/Desktop/Camera 5/images/test/", "../../bdd100k/seg/apollo_test_sd.pkl", "ApolloTest", mean)

    # trainPath = "../../bdd100k/seg/images/train/"
    # dataset_mean("../../bdd100k/seg/images/train/", "../../bdd100k/seg/train_mean.pkl", "Train")
    # dataset_mean("../../bdd100k/seg/images/test_apollo/", "../../bdd100k/seg/apollo_mean.pkl", "Apollo Test")
    test = os.listdir('../results/Normal_2/label')
    # test = os.listdir('C:/Users/Home/Documents/FYP-Project/bdd100k/seg/labels/test_apollo/')


    for i, data in enumerate(test):
        print(data)
        path = osp.join('../results/Normal_2/label', data)
        img = cv2.imread(path)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        path = osp.join('../results/DA_2/pred', data)
        img = cv2.imread(path)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    # dataset_mean("../../bdd100k/seg/images/val/", "../../bdd100k/seg/val_mean.pkl", "Validation")
    
    # print("\nCompleted Calculation of Mean of all Datasets!")

    # Print("Calculating Class Weights of Training Dataset...")
    # get_class_weights(trainPath)
    # print("\nCompleted Class Weights of Training Dataset!")

    # Relabel Apollo labels to BDD labels
    # convert_apollo_to_bdd("../../bdd100k/seg/labels/test_apollo_unchanged/", "../../bdd100k/seg/labels/test_apollo/")
    


    # dataset_mean("../../bdd100k/seg/images/train/", "../../bdd100k/seg/train_mean.pkl", "Train")

    # dataset_sd("../../bdd100k/seg/images/train/", "../../bdd100k/seg/train_sd.pkl", "Train",TRAIN_IMG_MEAN)

    # if not os.path.exists("../pretrain"):
    #     os.makedirs("../pretrain")

    # if not os.path.exists("../checkpoints"):
    #     os.makedirs("../checkpoints")


