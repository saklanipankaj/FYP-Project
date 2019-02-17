# Preprocess Data 
# Created By Saklani Pankaj
import pickle
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# Generating mean of images
if __name__ == '__main__':
    datasetPath = "../../bdd100k/seg/images/train/"
    mean_arr = []
    # Enumerate through the image directory
    for num, img_path in enumerate(os.listdir(datasetPath)):
        completed = num/len(os.listdir(datasetPath))*100
        if int(completed)%10 == 0:
            print ("Computing weights: {0:.2f}% Completed".format(completed))

        img = cv2.imread(datasetPath+img_path, -1).numpy()
        mean_arr.append(np.mean(img),axis=0)

    mean = np.mean(mean_arr,axis=0)
    print(mean)

    # save class weights to file
    with open("../../bdd100k/seg/img_mean.pkl", "wb") as file:
        pickle.dump(mean, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)


        
