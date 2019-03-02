import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import torchvision.transforms as TF
import cv2
from torch.utils import data
from itertools import cycle

class DataSetTrain(data.Dataset):
    #Default load train
    def __init__(self, root, max_iters= None, crop_size=(1280,720), mean=(128, 128, 128), mirror=False, scale=False):
        self.root = root
        self.ignore_label = 255
        self.mean = mean
        self.is_mirror = mirror
        self.crop_w, self.crop_h = crop_size

        imgPath = osp.join(self.root, "images/train")
        labelPath = osp.join(self.root,"labels/train")
        
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = os.listdir(imgPath)
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters)/len(self.img_ids)))

        self.files = []
        self.scale = scale

        #List iamges in the directory
        for img in self.img_ids:
            img_file = osp.join(imgPath, img)
            
            #remove .png extension from the image
            name = img[:-4]
            label = "%s_train_id.png"%name
            label_file = osp.join(labelPath,label)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def random_scale(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.random_scale(image, label)
        
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        img_h, img_w = label.shape
        #Pad with 0 for image and ignore_label for label if size < cropsize 
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), name

class DataSetVal(data.Dataset):
    #Default load train
    def __init__(self, root, max_iters= None, crop_size=(1280,720), mean=(128, 128, 128), mirror=False, scale=False):
        self.root = root
        self.ignore_label = 255
        self.mean = mean
        self.is_mirror = mirror
        self.crop_w, self.crop_h = crop_size
        
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

        self.files = []
        self.scale = scale

        imgPath = root + "images/val/"
        labelPath = root +"labels/val/"

        #List iamges in the directory
        for img in os.listdir(imgPath):
            img_file = imgPath+img
            
            #remove .png extension from the image
            name = img[:-4]
            label = "%s_train_id.png"%name
            label_file = osp.join(labelPath,label)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def random_scale(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.random_scale(image, label)
        image = np.asarray(image, np.float32)

        image -= self.mean
        
        img_h, img_w = label.shape
        #Pad with 0 for image and ignore_label for label if size < cropsize 
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        print ("Pad_H is"+(pad_h > 0 or pad_w > 0))
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), name

class DataSetTest(data.Dataset):
    def __init__(self, root, max_iters= None, crop_size=(1280,720), mean=(128, 128, 128), mirror=False, scale=False, train=True):
        self.root = root
        self.ignore_label = 255
        self.mean = mean
        self.is_mirror = mirror
        self.crop_w, self.crop_h = crop_size
        self.train = train

        imgPath = osp.join(self.root, "images/test")
        
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = os.listdir(imgPath)
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters)/len(self.img_ids)))

        self.files = []
        self.scale = scale

        # Image Path for Test Images
        imgPath = root + "images/test/"

        # Dataset has all Apollo Test Images for Training
        if self.train:
            for i, img in enumerate(os.listdir(imgPath)):
                if i == max_iters:
                    break

                img_file = imgPath+img

                #remove .png extension from the image
                name = img[:-4]
                label = "%s_train_id.png"%name
                self.files.append({
                    "img": img_file,
                    "name": name
                })

        # Dataset has only required 179 images
        else:
            with open(osp.join(self.root, "apollo_test_list.txt")) as f:
                names = f.read().splitlines()

            with open(osp.join(self.root, "apollo_test_list_md5.txt")) as f:
                names_md5 = f.read().splitlines()

            #List iamges in the directory
            for i, img in enumerate(names_md5):

                img += ".jpg"
                img_file = osp.join(imgPath, img)

                # Name is in video sequence
                name = names[i]

                self.files.append({
                    "img": img_file,
                    "name": name
                })

    def __len__(self):
        return len(self.files)

    def random_scale(self, image):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image

    def image_scale(self, image):
        f_scale = 1270 / 3384
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)

        # Making Test Images as close as possible to train images for training
        if self.train:
            image = self.image_scale(image)

        size = image.shape
        name = datafiles["name"]

        image = np.asarray(image, np.float32)

        image -= self.mean

        #Random Scaling
        if self.scale:
            image = self.random_scale(image)
        
        img_h, img_w, _ = size
        #Pad with 0 for image and ignore_label for label if size < cropsize 
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
        else:
            img_pad = image

        img_h, img_w, _ = img_pad.shape

        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        
        image = image.transpose((2, 0, 1))
        
        # Random Flipping
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]

        return image.copy(), name


if __name__ == '__main__':

    IMG_MEAN = np.array((128,128,128), dtype=np.float32)

    BATCH_SIZE = 10
    IMAGE_SIZE = (100,100)
    DATA_DIRECTORY = '../../bdd100k/seg/'
    IGNORE_LABEL = 255
    INPUT_SIZE = '321,321'
    LEARNING_RATE = 2.5e-4
    MOMENTUM = 0.9
    NUM_CLASSES = 19
    NUM_STEPS = 700
    POWER = 0.9
    RANDOM_SEED = 1234
    RESTORE_FROM = './checkpoints/MS_DeepLab_resnet_pretrained_COCO_init.pth'
    SAVE_NUM_IMAGES = 2
    SAVE_EVERY = 100
    CHECKPOINT_DIR = './checkpoints/'
    WEIGHT_DECAY = 0.0005
    EPOCHS = 1000

    #Transform Params
    RANDOM_SCALE = False
    RANDOM_FLIP = False


    # root = "../../bdd100k/seg"
    # imgPath = osp.join(root, "images/train")
    # labelPath = osp.join(root, "labels/train")
    #count = 0
    #for img in os.listdir(imgPath):
        #img_file = osp.join(imgPath, img)
        # remove .png extension from the image
        #name = img[:-4]
        #label = "%s_train_id.png" % name
        #label_file = osp.join(labelPath, label)
        #if count == 0:
            #print(img_file," exists ", os.path.exists(img_file))
            #print(label_file, " exists ", os.path.exists(label_file))
            #lbl = cv2.imread(label_file)
            #plt.imshow(lbl)
            #plt.show()
        #count += 1
        # self.files.append({
        #     "img": img_file,
        #     "label": label_file,
        #     "name": name
        # })
    
    trainloader = data.DataLoader(DataSetTrain(DATA_DIRECTORY, max_iters=NUM_STEPS*BATCH_SIZE, crop_size=IMAGE_SIZE, 
                    scale=RANDOM_SCALE, mirror=RANDOM_FLIP, mean=IMG_MEAN), 
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True)

    testloader = data.DataLoader(DataSetTest(DATA_DIRECTORY, max_iters=NUM_STEPS*BATCH_SIZE, crop_size=IMAGE_SIZE, 
                    scale=RANDOM_SCALE, mirror=RANDOM_FLIP, mean=IMG_MEAN), 
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True)


    # for epoch in range(4):
    #     for i, data in enumerate(trainloader):
    #         imgs, labels, size , name  = data
    #         if i == 0:
    #             print(imgs.size())
    #             img = imgs[0].numpy()#torchvision.utils.make_grid(imgs).numpy()
    #             img = np.transpose(img, (1, 2, 0))
    #             img += IMG_MEAN
    #             img = img/255.0
    #             plt.subplot(1,7,epoch+1)
    #             plt.axis('off')
    #             plt.imshow(img)
    #             break

    # plt.show()

    test_iter = cycle(testloader)
    # for i in range(0,579):
    for i, data in enumerate(trainloader):
        data_test = next(test_iter)
        imgs, names = data_test
        if i%193==0:
            print("i: %d %d"%(i, int(i/193)+1))
            img = imgs[0].numpy()#torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img += IMG_MEAN
            img = img/255.0
            plt.subplot(1,4, int(i/193)+1)
            plt.axis('off')
            plt.imshow(img)
   
    plt.show()
                
