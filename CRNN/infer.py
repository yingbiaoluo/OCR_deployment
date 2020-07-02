import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
from torch.autograd import Variable

sys.path.append(os.getcwd())
from .utils import utils
from .Net import crnn_vgg
from .config import config

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str,
                    default='/home/lyb/ocr/text_det_reg/dataset/images_sentences1/images/000000_00_00.jpg',
                    help='the path to your image')
opt = parser.parse_args()


def get_alphabets(alphabet_path):
    # 获取字符表
    alphabets = utils.generate_alphabets(alphabet_path=alphabet_path)
    nclass = len(alphabets) + 1
    return nclass, alphabets


def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    ret, image = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = image.shape
    image = cv2.resize(image, (0, 0), fx=config.imgH / h, fy=config.imgH / h, interpolation=cv2.INTER_CUBIC)
    # 不足的，补充白色区域
    image = padding_image(image)

    image = (np.reshape(image, (32, config.imgW, 1))).transpose(2, 0, 1)
    # 预处理，转换为torchTensor
    image = preprocess(image)
    return image

def padding_image(image_):
    h, w = image_.shape
    img = 255. * np.ones((config.imgH, config.imgW))
    if w < config.imgW:
        img[:, :w] = image_
    else:
        img = cv2.resize(image_, (config.imgW, config.imgH), interpolation=cv2.INTER_CUBIC)
    img = np.uint8(img)
    return img

def preprocess(image_):
    image = image_.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(config.mean).div_(config.std)
    return image


def crnn_recognition(image, model, device, alphabets):
    converter = utils.strLabelConverter(alphabets)
    image = Variable(image)
    image = image.to(device)

    model.eval()
    preds = model(image)  # (141, batch_size, 6773)

    _, preds = preds.max(2)  # (141, batch_szie)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = torch.IntTensor([preds.size(0)])
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))
    return 0


def infer_recognition(image_batch):
    return 0


if __name__ == '__main__':
    nclass, alphabets = get_alphabets(alphabet_path='./dataset/alphabets.txt')

    model = crnn_vgg.CRNN(32, 1, nclass, 256)
    device = torch.device('cpu')
    # device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # crnn_model_path = './models/crnn_best.pth'
    crnn_model_path = './models/crnn_Rec_ocr_50_0.00478125.pth'
    print('===> loading pretrained model from {}'.format(crnn_model_path))
    model.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))
    # model.load_state_dict(torch.load(crnn_model_path))

    start_time = time.time()
    # ------在此输入图片路径-------
    image = cv2.imread(opt.image_path)
    image = process_image(image)
    image = image.view(1, *image.size())

    crnn_recognition(image, model, device, alphabets)

    finish_time = time.time()
    print('elapsed time of recognition: {0}'.format(finish_time - start_time))



