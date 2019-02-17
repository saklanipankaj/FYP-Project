import torch
import torch.utils.data as data
import torchvision
import numpy as np
import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt


class DatasetTrain(torch.utils.data.Dataset):
    # resize the image can be specified
    def __init__(self, root, size=(720, 1280), resize=(720, 1280), transform=True):
        self.root = root
        self.img_dir = root + "images/train/"
        self.label_dir = root + "labels/train/"

        self.size = size

        self.resize = resize

        self.files = []
        self.transform = transform

        # List iamges in the directory
        for img in os.listdir(self.img_dir):
            img_id = img.split(".jpg")[0]

            img_path = self.img_dir + img

            label_path = self.label_dir + img_id + "_train_id.png"

            file = {}
            file["img"] = img_path
            file["label"] = label_path
            file["img_id"] = img_id
            self.files.append(file)

        self.num_files = len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        img_path = file["img"]
        label_img_path = file["label"]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (shape: (720, 1280, 3))
        label_img = cv2.imread(label_img_path, -1)  # (shape: (720, 1280))

        if self.resize != self.size:
            # resize img without interpolation (want the image to still match
            # label_img, which we resize below):
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_NEAREST)  # (shape: (720, 1280, 3))

            # resize label_img without interpolation (want the resulting image to
            # still only contain pixel values corresponding to an object class):
            label_img = cv2.resize(label_img, self.resize, interpolation=cv2.INTER_NEAREST)  # (shape: (720, 1280))

        if self.transform:
            # flip the img and the label with 0.5 probability:
            flip = np.random.randint(low=0, high=2)
            if flip == 1:
                img = cv2.flip(img, 1)
                label_img = cv2.flip(label_img, 1)

            ########################################################################
            # randomly scale the img and the label:
            ########################################################################
            scale = np.random.uniform(low=0.7, high=2.0)
            new_img_h = int(scale * self.resize[0])
            new_img_w = int(scale * self.resize[1])

            # resize img without interpolation (want the image to still match
            # label_img, which we resize below):
            img = cv2.resize(img, (new_img_w, new_img_h),
                             interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w, 3))

            # resize label_img without interpolation (want the resulting image to
            # still only contain pixel values corresponding to an object class):
            label_img = cv2.resize(label_img, (new_img_w, new_img_h),
                                   interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w))
            ########################################################################

            # # # # # # # # debug visualization START
            # print (scale)
            # print (new_img_h)
            # print (new_img_w)
            #
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            #
            # cv2.imshow("test", label_img)
            # cv2.waitKey(0)
            # # # # # # # # debug visualization END

            ########################################################################
            # select a 256x256 random crop from the img and label:
            ########################################################################
            start_x = np.random.randint(low=0, high=(new_img_w - 256))
            end_x = start_x + 256
            start_y = np.random.randint(low=0, high=(new_img_h - 256))
            end_y = start_y + 256

            img = img[start_y:end_y, start_x:end_x]  # (shape: (256, 256, 3))
            label_img = label_img[start_y:end_y, start_x:end_x]  # (shape: (256, 256))
            ########################################################################

            # # # # # # # # debug visualization START
            # print (img.shape)
            # print (label_img.shape)
            #
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            #
            # cv2.imshow("test", label_img)
            # cv2.waitKey(0)
            # # # # # # # # debug visualization END

            # normalize the img (with the mean and std for the pretrained ResNet):
            mean = np.array(img.mean(axis=0).mean(axis=0))
            img = img - mean  # img-mean
            img = img / np.array([0.229, 0.224, 0.225])  # img/std (shape: (256, 256, 3))

        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 256, 256))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 256, 256))
        label_img = torch.from_numpy(label_img)  # (shape: (256, 256))

        return (img, label_img, file['img_id'])

    def __len__(self):
        return self.num_files

    def meanNstd(self):
        mean = 0
        std = []
        for i, file in enumerate(self.files):
            img = cv2.imread(file["img"])
            val = np.reshape(image[:, :, 0], -1)
            mean += np.mean(val)
            std

        return


class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, root, size=(720, 1280), resize=(720, 1280), transform=False):
        self.root = root
        self.img_dir = root + "images/val/"
        self.label_dir = root + "labels/val/"

        self.size = size

        self.resize = resize

        self.files = []
        self.transform = transform

        # List iamges in the directory
        for img in os.listdir(self.img_dir):
            img_id = img.split(".jpg")[0]

            img_path = self.img_dir + img

            label_path = self.label_dir + img_id + "_train_id.png"

            file = {}
            file["img"] = img_path
            file["label"] = label_path
            file["img_id"] = img_id
            self.files.append(file)

        self.num_files = len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        img_path = file["img"]
        label_img_path = file["label"]
        img_id = file['img_id']

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (shape: (720, 1280, 3))
        label_img = cv2.imread(label_img_path, -1)  # (shape: (1024, 2048))

        if self.resize != self.size:
            # resize img without interpolation (want the image to still match
            # label_img, which we resize below):
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_NEAREST)  # (shape: (720, 1280, 3))

            # resize label_img without interpolation (want the resulting image to
            # still only contain pixel values corresponding to an object class):
            label_img = cv2.resize(label_img, self.resize, interpolation=cv2.INTER_NEAREST)  # (shape: (720, 1280))
            # # # # # # # # debug visualization START
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            #
            # cv2.imshow("test", label_img)
            # cv2.waitKey(0)
            # # # # # # # # debug visualization END

            # normalize the img (with the mean and std for the pretrained ResNet):
        img = img / 255.0
        # img = img - np.array([0.485, 0.456, 0.406])
        # img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 512, 1024))
        label_img = torch.from_numpy(label_img)  # (shape: (512, 1024))

        return (img, label_img, img_id)

    def __len__(self):
        return self.num_files

class DatasetSeq(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, sequence):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/demoVideo/stuttgart_" + sequence + "/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []

        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split("_leftImg8bit.png")[0]

            img_path = self.img_dir + file_name

            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # (shape: (1024, 2048, 3))
        # resize img without interpolation:
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples


class DatasetThnSeq(torch.utils.data.Dataset):
    def __init__(self, thn_data_path):
        self.img_dir = thn_data_path + "/"

        self.examples = []

        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split(".png")[0]

            img_path = self.img_dir + file_name

            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # (shape: (512, 1024, 3))

        # normalize the img (with mean and std for the pretrained ResNet):
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples


# Testing of Dataset
if __name__ == '__main__':
    root = "../../bdd100k/seg"
    imgPath = osp.join(root, "images/train")
    labelPath = osp.join(root, "labels/train")
    # count = 0
    # for img in os.listdir(imgPath):
    # img_file = osp.join(imgPath, img)
    # remove .png extension from the image
    # name = img[:-4]
    # label = "%s_train_id.png" % name
    # label_file = osp.join(labelPath, label)
    # if count == 0:
    # print(img_file," exists ", os.path.exists(img_file))
    # print(label_file, " exists ", os.path.exists(label_file))
    # lbl = cv2.imread(label_file)
    # plt.imshow(lbl)
    # plt.show()
    # count += 1
    # self.files.append({
    #     "img": img_file,
    #     "label": label_file,
    #     "name": name
    # })

    dst = DatasetVal("../bdd100k/seg/")
    valloader = data.DataLoader(dst, batch_size=2)

    for i, data in enumerate(valloader):
        imgs, labels, img_id = data
        if i == 0:
            print(labels[0])
            img = imgs[0]  # torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            print(img.size())
            plt.imshow(labels[0])
            plt.show()
