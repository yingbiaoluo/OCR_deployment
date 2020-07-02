import numpy as np
import torch
from torch.autograd import Variable

class strLabelConverter(object):
    def __init__(self, alphabet_):
        self.alphabet = alphabet_ + 'Ω'
        self.dict = {}
        for i, char in enumerate(self.alphabet):
            self.dict[char] = i + 1

    def encode(self, text):
        length = []
        result = []

        for item in text:
            item = item.replace(' ', '').replace('\t', '')
            length.append(len(item))
            for char in item:
                # if char == ' ' or '\t':
                #     continue
                if char not in self.alphabet:
                    print('char {} not in alphabets!'.format(char))
                    char = '-'
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0].replace(' ', ''))
    return label


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def generate_alphabets(alphabet_path):
    """
    读取文本标签，生成字符表。
    :param alphabet_path: 文本标签.
    :return: 字符表.
    """
    with open(alphabet_path, 'r', encoding='utf-8') as file:
        alphabet = sorted(list(set(repr(''.join(file.readlines())))))
        if ' ' in alphabet:
            alphabet.remove(' ')
        alphabet = ''.join(alphabet)
    return alphabet