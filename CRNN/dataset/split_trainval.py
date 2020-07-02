import os
import random


if __name__ == '__main__':
    train_image_root = './images_sentences/train_images'
    train_label_path = './images_sentences/train_labels'
    val_image_root = './images_sentences/val_images'
    val_label_path = './images_sentences/val_labels'
    if not os.path.exists(train_image_root):
        os.mkdir(train_image_root)
    if not os.path.exists(train_label_path):
        os.mkdir(train_label_path)
    if not os.path.exists(val_image_root):
        os.mkdir(val_image_root)
    if not os.path.exists(val_label_path):
        os.mkdir(val_label_path)

    images_root = './images_sentences/images'
    images = os.listdir(images_root)
    label_path = './images_sentences/labels/sentences_label.txt'
    with open(label_path) as f:
        a = f.readlines()
    labels = {b.split(' ', 1)[0]: b.split(' ', 1)[1] for b in a}

    random.shuffle(images)
    train_images = sorted(images[:-500])
    val_images = sorted(images[-500:])
    print(train_images)
    print(len(train_images))
    print(val_images)
    print(len(val_images))

    for i in range(len(train_images)):
        os.system('cp {} {}'.format(images_root + os.sep + train_images[i], train_image_root))
        with open(train_label_path+'/'+'train_labels.txt', 'a') as f:
            f.write(train_images[i] + ' ' + labels[train_images[i]])

    for i in range(len(val_images)):
        os.system('cp {} {}'.format(images_root + os.sep + val_images[i], val_image_root))
        with open(val_label_path+'/'+'val_labels.txt', 'a') as f:
            f.write(val_images[i] + ' ' + labels[val_images[i]])

