import os
import random
import argparse
import numpy as np
from sys import exit
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from config import config
from utils import utils
from dataset import dataset
from Net import crnn_vgg


# 构建命令行参数解析器
def init_args():
    parser = argparse.ArgumentParser(description="Train CRNN for text recognition")
    parser.add_argument("--trainroot", default="./dataset/images/train", help="path to dataset")
    parser.add_argument("--valroot", default="./dataset/images/val", help="path to dataset")
    parser.add_argument("--cuda", action="store_true", help="enable cuda", default=True)

    return parser.parse_args()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    # print('classname:', classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0  # replace all nan/inf in gradients to zero


def val(crnn, val_loader, criterion, iteration, max_i=1000):
    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    for i_batch, (image, index) in enumerate(val_loader):
        image = image.to(device)
        label = utils.get_batch_label(val_dataset, index)
        preds = crnn(image)
        batch_size = image.size(0)
        index = np.array(index.data.numpy())
        text, length = conveter.encode(label)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = conveter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, label):
            print('&&&&&&', pred, target)
            edit_distance = lev_ratio(pred, target)
            print(edit_distance)
            if pred == target:
                n_correct += 1

        if (i_batch + 1) % config.displayInterval == 0:
            print('[%d/%d][%d/%d]' %
                  (epoch, epochs, i_batch, len(val_loader)))

        if i_batch == max_i:
            break
    raw_preds = conveter.decode(preds.data, preds_size.data, raw=True)[:config.n_test_disp]
    print(raw_preds)
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct)
    print(max_i * config.val_batch_size)
    accuracy = n_correct / float(max_i * config.val_batch_size)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

    return accuracy


def lev_ratio(str_a, str_b):
    """
    ED距离，用来衡量单词之间的相似度
    :param str_a:
    :param str_b:
    :return:
    """
    str_a = str_a.lower()
    str_b = str_b.lower()
    matrix_ed = np.zeros((len(str_a) + 1, len(str_b) + 1), dtype=np.int)
    matrix_ed[0] = np.arange(len(str_b) + 1)
    matrix_ed[:, 0] = np.arange(len(str_a) + 1)
    for i in range(1, len(str_a) + 1):
        for j in range(1, len(str_b) + 1):
            # 表示删除a_i
            dist_1 = matrix_ed[i - 1, j] + 1
            # 表示插入b_i
            dist_2 = matrix_ed[i, j - 1] + 1
            # 表示替换b_i
            dist_3 = matrix_ed[i - 1, j - 1] + (2 if str_a[i - 1] != str_b[j - 1] else 0)
            # 取最小距离
            matrix_ed[i, j] = np.min([dist_1, dist_2, dist_3])
    # print(matrix_ed)
    levenshtein_distance = matrix_ed[-1, -1]
    sum = len(str_a) + len(str_b)
    levenshtein_ratio = (sum - levenshtein_distance) / sum
    return levenshtein_ratio


if __name__ == "__main__":

    # -------配置参数-------
    args = init_args()

    # 随机种子
    manualSeed = 10
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True  # 为网络的每个卷积层搜索最优的卷积实现算法，实现网络的加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取字符表
    alphabet_path = 'dataset/alphabets.txt'
    alphabets = utils.generate_alphabets(alphabet_path=alphabet_path)

    # 模型保存地址
    model_save_path = './models'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # -------准备数据集--------
    # images_root = './dataset/images_sentences/images'
    # train_labels_path = './dataset/images_sentences/labels/train_labels.txt'
    # val_labels_path = './dataset/images_sentences/labels/val_labels.txt'
    images_root = './dataset/images_sentences1/images_process'
    train_labels_path = './dataset/images_sentences1/labels/train_labels1.txt'
    val_labels_path = './dataset/images_sentences1/labels/val_labels1.txt'
    # images_root = '/home/lyb/dataset/OCR/Sythetic_Chinese_Character_Dataset/images'
    # train_labels_path = '/home/lyb/crnn_chinese_characters_rec/train.txt'
    # val_labels_path = '/home/lyb/crnn_chinese_characters_rec/test.txt'
    train_dataset = dataset.Dataset_OCR(image_root=images_root, label_path=train_labels_path,
                                        alphabet=alphabets, resize_shape=(config.imgH, config.imgW))
    val_dataset = dataset.Dataset_OCR(image_root=images_root, label_path=val_labels_path,
                                      alphabet=alphabets, resize_shape=((config.imgH, config.imgW)))
    train_dataLoader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    val_dataLoader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=1)

    conveter = utils.strLabelConverter(alphabet_=train_dataset.alphabet)

    # -------搭建网络----------
    nc = 1  # 输入通道数
    nclass = len(alphabets) + 1  # 6773
    crnn = crnn_vgg.CRNN(imgH=32, nc=nc, nclass=nclass, nh=256)
    crnn.apply(weights_init)  # 初始化权重

    if config.crnn != '':
        print('loading pretrained model from {}'.format(config.crnn))
        crnn.load_state_dict(torch.load(config.crnn, map_location='cpu'))

    crnn.register_backward_hook(backward_hook)
    criterion = torch.nn.CTCLoss(reduction='sum')
    optimizer = optim.RMSprop(crnn.parameters(), lr=config.lr)

    crnn = crnn.to(device)
    criterion = criterion.to(device)

    epochs = 300
    best_accuracy = 0.5
    for epoch in range(epochs):
        # --------训练阶段---------
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
        loss_avg = utils.averager()

        for i_batch, (image, index) in enumerate(train_dataLoader):
            image = image.to(device)
            label = utils.get_batch_label(train_dataset, index)
            # print('label:', label)
            text, length = conveter.encode(label)
            # print('text:', text, '\nlength:', length)
            preds = crnn(image)
            # print('preds:', preds)
            batch_size = image.size(0)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            cost = criterion(log_probs=preds, targets=text, input_lengths=preds_size,
                             target_lengths=length) / batch_size
            crnn.zero_grad()
            cost.backward()
            optimizer.step()
            loss_avg.add(cost)

            if (i_batch + 1) % config.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, epochs, i_batch, len(train_dataLoader), loss_avg.val()))
                loss_avg.reset()

        accuracy = val(crnn, val_dataLoader, criterion, epoch, max_i=1000)
        for p in crnn.parameters():
            p.requires_grad = True
        if (epoch + 1) % 10 == 0:
            torch.save(crnn.state_dict(),
                       '{0}/crnn_Rec_ocr_{1}_{2}.pth'.format(model_save_path, epoch, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(crnn.state_dict(), '{0}/crnn_ocr_best.pth'.format(model_save_path))
        print("is best accuracy: {0}".format(accuracy > best_accuracy))
