import torch

# dataset
std = 0.2360883
mean = 0.8901977
num_works = 2
batch_size = 64
imgW = 560
imgH = 32

val_batch_size = 32

# training phase
nh = 256
lr = 0.00005

crnn = '/home/lyb/ocr/CRNN/expr/crnn_best.pth'

displayInterval = 1
n_test_disp = 100

adam = False
adadelta = False
