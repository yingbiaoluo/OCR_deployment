import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BidiretionalLSTM(nn.Module):
    """After CNN backbone
    Args：
        nIn:(int)
    """

    def __init__(self, nIn, nHidden, nOut):
        super(BidiretionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    """
    Args:
        imgH (int): image height
        nc (int): the input channels
    """

    def __init__(self, imgH, nc, nclass, nh, leaky_relu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        kernel_size = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        padding_size = [1, 1, 1, 1, 1, 1, 0]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def ConvReLU(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_size[i], strides[i], padding_size[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(inplace=True))

        ConvReLU(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        ConvReLU(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        ConvReLU(2, True)
        ConvReLU(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        ConvReLU(4, True)
        ConvReLU(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        ConvReLU(6, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidiretionalLSTM(512, nh, nh),
            BidiretionalLSTM(nh, nh, nclass))

    def forward(self, input):
        conv = self.cnn(input)  # batch * 1 * 32 * 256
        b, c, h, w = conv.size()  # batch * 512 * 1 * 65
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)  # batch * 512 * 65
        conv = conv.permute(2, 0, 1)  # 65 * batch * 512  [width, batch_size, channel]
        output = F.log_softmax(self.rnn(conv), dim=2)  # 65 * batch * 37
        return output


if __name__ == '__main__':
    # batch_size * channel * imgH * imgW
    torch.manual_seed(10)
    a = torch.rand(2, 1, 32, 256)
    crnn = CRNN(imgH=32, nc=1, nclass=37, nh=256)
    feature_map = crnn(a)
    # print(feature_map.shape)

    # 统计参数
    print(crnn.parameters)
    for x in crnn.parameters():
        # print(x)
        # print(x.shape)
        print(x.numel())
    print(sum(x.numel() for x in crnn.parameters()))

    # 可视化特征图 visualize the feature map
    # print(feature_map.detach().numpy()[0, 1, :, :])
    # plt.imshow(feature_map.detach().numpy()[0, 1, :, :])
    # plt.show()
