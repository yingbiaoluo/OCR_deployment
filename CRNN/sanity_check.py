import os
import cv2


def get_labels(label_path_):
    with open(label_path_) as f:
        labels = [{a.split(' ', 1)[0]: a.strip().split(' ', 1)[1]} for a in f.readlines()]
    return labels


images_root = '/home/lyb/ocr/CRNN/dataset/images_sentences1/images'
train_labels_path = '/home/lyb/ocr/CRNN/dataset/images_sentences1/labels/train_labels.txt'
val_labels_path = '/home/lyb/ocr/CRNN/dataset/images_sentences1/labels/val_labels.txt'

images = os.listdir(images_root)
print(len(images))
train_labels = get_labels(train_labels_path)
print(len(train_labels))
val_labels = get_labels(val_labels_path)
print(len(val_labels))

empty_train_label = []
for i in range(len(train_labels)):
    image_name = list(train_labels[i].keys())[0]
    image_path = images_root + os.sep + image_name
    image = cv2.imread(image_path)
    if image is None:
        print('{} not exit!'.format(image_name))
        print(train_labels[i])
        empty_train_label.append(train_labels[i])
        os.system('rm {}'.format(image_path))
for j in range(len(empty_train_label)):
    train_labels.remove(empty_train_label[j])
print(len(train_labels))

empty_val_label = []
for i in range(len(val_labels)):
    image_name = list(val_labels[i].keys())[0]
    image_path = images_root + os.sep + image_name
    image = cv2.imread(image_path)
    if image is None:
        print('{} not exit!'.format(image_name))
        print(val_labels[i])
        empty_val_label.append(val_labels[i])
        os.system('rm {}'.format(image_path))
for j in range(len(empty_val_label)):
    val_labels.remove(empty_val_label[j])
print(len(val_labels))

train_labels_path_mo = '/home/lyb/ocr/CRNN/dataset/images_sentences1/labels/train_labels1.txt'
val_labels_path_mo = '/home/lyb/ocr/CRNN/dataset/images_sentences1/labels/val_labels1.txt'
with open(train_labels_path_mo, 'w') as f:
    for i in range(len(train_labels)):
        image_name = list(train_labels[i].keys())[0]
        content = list(train_labels[i].values())[0]
        f.write('{} {}\n'.format(image_name, content))

with open(val_labels_path_mo, 'w') as f:
    for i in range(len(val_labels)):
        image_name = list(val_labels[i].keys())[0]
        content = list(val_labels[i].values())[0]
        f.write('{} {}\n'.format(image_name, content))
