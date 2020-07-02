import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import config
from utils import utils

class Dataset_OCR(Dataset):
    def __init__(self, image_root, label_path, alphabet, resize_shape, transform=None):
        super(Dataset_OCR).__init__()
        self.image_root = image_root
        self.labels = self.get_labels(label_path)
        self.alphabet = alphabet
        self.height, self.width = resize_shape
        self.transform = transform

    @staticmethod
    def get_labels(label_path_):
        with open(label_path_) as f:
            labels = [{a.split(' ', 1)[0]: a.strip().split(' ', 1)[1]}for a in f.readlines()]
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_name = list(self.labels[item].keys())[0]
        image = cv2.imread(self.image_root + os.sep + image_name)
        if image is None:
            print('{} not exit!'.format(image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        image = cv2.resize(image, (0, 0), fx=self.height / h, fy=self.height / h, interpolation=cv2.INTER_CUBIC)
        # 不足的，补充白色区域
        image = self.padding_image(image)
        # cv2.imshow('image {}'.format(image_name), image)
        # cv2.waitKey(0)
        image = (np.reshape(image, (32, self.width, 1))).transpose(2, 0, 1)
        # 预处理，转换为torchTensor
        image = self.preprocess(image)
        return image, item

    def padding_image(self, image_):
        h, w = image_.shape
        img = 255. * np.ones((self.height, self.width))
        if w < self.width:
            img[:, :w] = image_
        else:
            img = cv2.resize(image_, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        img = np.uint8(img)
        return img

    def preprocess(self, image_):
        image = image_.astype(np.float32) / 255.
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image.sub_(config.mean).div_(config.std)
        return image


if __name__ == '__main__':
    image_root = 'images_sentences/images'
    label_path = 'images_sentences/labels/sentences_label.txt'
    # image_root = 'Synthetic_Chinese_3_6M/train_tiny_images'
    # label_path = 'Synthetic_Chinese_3_6M/label/train_tiny.txt'

    alphabet_path = './alphabets.txt'
    alphabet = utils.generate_alphabets(alphabet_path)
    resize_shape = (32, 560)
    dataset = Dataset_OCR(image_root, label_path, alphabet, resize_shape)

    for i in range(len(dataset)):
        dataset[i]

    # datasetLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_works)
    #
    # for i_batch, (img, index) in enumerate(datasetLoader):
    #     print(i_batch)
    #     print(img.shape)
    #     print(index)

    # images = os.listdir(dataset.image_root)
    # heights = []
    # widths = []
    # scales = []
    # for image_name in images:
    #     image = cv2.imread(image_root+'/'+image_name)
    #     h, w, c = image.shape
    #     heights.append(h)
    #     widths.append(w)
    #     scales.append(w/h)
    # print(images)

