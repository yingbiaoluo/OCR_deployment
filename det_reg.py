import os
import cv2
import argparse
import yaml
import time
from pprint import pprint
from easydict import EasyDict as edict
import numpy as np

import torch

from detect_text.models.model import DetRegModel
from detect_text.utils import get_device, load_checkpoint, scale_img
from CRNN.Net import crnn_vgg
from CRNN.infer import get_alphabets, process_image, crnn_recognition

model_path = os.environ.get("model_path")


def parse_arg():
    parser = argparse.ArgumentParser(description="Text detection and recognition")
    parser.add_argument('--config_file', type=str, default='./detect_text/config/ocr_dataset.yaml',
                        help='configuration filename')
    args = parser.parse_args()
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    return config


def infer_detection(img):
    start_time = time.time()

    img_scaled, f_scale = scale_img(img)  # (640, 640, 3)
    img_scaled = img_scaled.transpose((2, 0, 1)).astype(np.float32)
    img_scaled = torch.unsqueeze(torch.from_numpy(img_scaled), 0)

    # -------get config-------
    config = parse_arg()
    # pprint(config)

    # -------get device-------
    device = torch.device('cpu')
    print('Using device: ', device)

    model = DetRegModel(model_config=config['arch']['args'], device=device).to(device)

    # load checkpoint
    checkpoint_path = model_path + '/model_last.pth'
    print('===> loading weights to detection model from checkpoint: {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()
    start_time1 = time.time()
    socres, classes, pred_boxes = model(x=img_scaled.to(device))
    end_time = time.time()
    print('load model time:', start_time1 - start_time)
    print('infer time of detection:', end_time - start_time1)
    print('total time:', end_time - start_time)

    return pred_boxes.detach().numpy() / f_scale


def get_img_batch(image, pred_boxes):
    batch_size = pred_boxes.shape[0]
    cropped_images = []
    for i in range(batch_size):
        pred_box = pred_boxes[i]
        cropped_images.append(image[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2], :])
    return cropped_images


def infer_recognition(cropped_images):
    batch_size = len(cropped_images)
    image_batch = torch.zeros((batch_size, 1, 32, 560))
    for i in range(batch_size):
        cropped_image = cropped_images[i]
        cropped_image = process_image(cropped_image)
        image_batch[i] = cropped_image

    nclass, alphabets = get_alphabets(alphabet_path='./CRNN/dataset/alphabets.txt')

    model = crnn_vgg.CRNN(32, 1, nclass, 256)
    device = torch.device('cpu')
    model.to(device)

    crnn_model_path = model_path + '/crnn_Rec_ocr_50_0.00478125.pth'
    print('===> loading pretrained model from {}'.format(crnn_model_path))
    model.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))
    # model.load_state_dict(torch.load(crnn_model_path))

    start_time = time.time()
    crnn_recognition(image_batch, model, device, alphabets)

    finish_time = time.time()
    print('predict time of recognition: {0}'.format(finish_time - start_time))

    return 0


def handler(environ, context):
    # print('environ:', environ)
    # print('context:', context)
    # print(dir(context))
    # print(context.account_id, context.region, context.function.name)

    print('torch version:', torch.__version__)

    image_path = './test_images/000000_00.jpg'
    img = cv2.imread(image_path)
    print('input shape:', img.shape)

    # 文本检测
    pred_boxes = infer_detection(img=img).astype(np.int)
    print(pred_boxes)

    cropped_images = get_img_batch(img, pred_boxes)

    # 文本识别
    infer_recognition(cropped_images)

    return 'ending\n'


if __name__ == '__main__':
    image_path = '/Users/biaobiao/Downloads/project/OCR/dataset/OCR_pic/images_question/train_images/000000_00.jpg'
    img = cv2.imread(image_path)
    print(img.shape)

    # 文本检测
    pred_boxes = infer_detection(img=img).astype(np.int)
    print(pred_boxes)

    cropped_images = get_img_batch(img, pred_boxes)

    # 文本识别
    infer_recognition(cropped_images)
