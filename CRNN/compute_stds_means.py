import os
import cv2
import random
import numpy as np
from tqdm import tqdm

from config import config
from dataset import dataset
from utils import utils


def cal_std_mean(dataset, num):
    img, _ = dataset[0]
    imgs = img[np.newaxis, :, :, :]
    for i in range(1, num):
        if i % 100 == 0:
            print(i)
        img, _ = dataset[i]
        img_ = img[np.newaxis, :, :, :]
        imgs = np.concatenate((imgs, img_), axis=0)
    print(imgs.shape)
    imgs = imgs.astype(np.float32) / 255.
    img_flat = imgs.flatten()
    print('mean:', np.mean(img_flat))
    print('std:', np.std(img_flat))
    return 0


if __name__ == '__main__':
    #### compute stds and means
    # image_root = '/home/lyb/ocr/CRNN/dataset/images_sentences/images'
    # label_path = '/home/lyb/ocr/CRNN/dataset/images_sentences/labels/sentences_label.txt'

    image_root = '/home/lyb/dataset/OCR/Sythetic_Chinese_Character_Dataset/images'
    label_path = '/home/lyb/crnn_chinese_characters_rec/train.txt'

    alphabet_path = '/home/lyb/ocr/CRNN/dataset/alphabets.txt'
    alphabet = utils.generate_alphabets(alphabet_path)
    resize_shape = (32, 560)
    dataset = dataset.Dataset_OCR(image_root, label_path, alphabet, resize_shape)

    cal_std_mean(dataset, num=2000)
