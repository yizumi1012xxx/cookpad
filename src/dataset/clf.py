# -*- coding: utf-8 -*-
'''
clf.py
'''

import os
import logging
import pandas as pd
import numpy as np
from PIL import Image
from chainer.datasets import ImageDataset, LabeledImageDataset, split_dataset


def _read_image_as_array(path, dtype, img_size, img_type):
    image = Image.open(path)
    width, height = image.size

    if img_type == 'warp':
        image = image.resize((img_size, img_size))
    elif img_type == 'crop':
        if width > height:
            diff = (width - height) / 2
            box = (diff, 0, height + diff, height)
            image.crop(box)
        elif height > width:
            diff = (height - width) / 2
            box = (0, diff, width, width + diff)
            image.crop(box)
        image = image.resize((img_size, img_size))
    elif img_type == 'natural':
        w = int(width / min(width, height) * img_size)
        h = int(height / min(width, height) * img_size)
        image = image.resize((w, h))
    # return np.asarray(image, dtype=dtype) / 255.
    return np.asarray(image, dtype=dtype)


class WarpedImageDataset(ImageDataset):
    def __init__(
            self, paths, root='.', dtype=np.float32,
            use_memory=True, img_size=224, img_type='warp', img_average=None):
        super(WarpedImageDataset, self).__init__(
            paths, root=root, dtype=dtype)
        self.img_size = img_size
        self.img_type = img_type
        self.img_ave = None
        self.use_memory = False

        if img_average is not None:
            self.img_ave = _read_image_as_array(
                img_average, self._dtype, self.img_size, self.img_type)

        if use_memory:
            self.use_memory = True
            self.data = []
            for path in self._paths:
                path = os.path.join(self._root, path)
                image = _read_image_as_array(
                    path, self._dtype, self.img_size, self.img_type)
                self.data.append(image)

    def get_example(self, i):
        if self.use_memory:
            image = self.data[i]
        else:
            path = os.path.join(self._root, self._paths[i])
            image = _read_image_as_array(
                path, self._dtype, self.img_size, self.img_type)

        if self.img_ave is not None:
            image -= self.img_ave

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, np.newaxis]
        return image.transpose(2, 0, 1)


class WarpedLabeledImageDataset(LabeledImageDataset):
    def __init__(
            self, pairs, root='.', dtype=np.float32, label_dtype=np.int32,
            use_memory=True, img_size=224, img_type='warp', img_average=None):
        super(WarpedLabeledImageDataset, self).__init__(
            pairs, root=root, dtype=dtype, label_dtype=label_dtype)
        self.img_size = img_size
        self.img_type = img_type
        self.img_ave = None
        self.use_memory = False

        if img_average is not None:
            self.img_ave = _read_image_as_array(
                img_average, self._dtype, self.img_size, self.img_type)

        if use_memory:
            self.use_memory = True
            self.data = []
            for path, _ in self._pairs:
                path = os.path.join(self._root, path)
                image = _read_image_as_array(
                    path, self._dtype, self.img_size, self.img_type)
                self.data.append(image)

    def get_example(self, i):
        path, int_label = self._pairs[i]
        if self.use_memory:
            image = self.data[i]
        else:
            full_path = os.path.join(self._root, path)
            image = _read_image_as_array(
                full_path, self._dtype, self.img_size, self.img_type)

        if self.img_ave is not None:
            image -= self.img_ave

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, np.newaxis]
        label = np.array(int_label, dtype=self._label_dtype)
        return image.transpose(2, 0, 1), label


def get_clf_data(use_memory=True, img_size=224, img_type='warp', split_val=0.9):

    def __get_train_list():
        train_list_path = 'data/clf/train_master.tsv'
        dataframe = pd.read_csv(train_list_path, sep='\t', usecols=['file_name', 'category_id'])
        train_data_list = pd.DataFrame(dataframe).to_records(index=False)
        return train_data_list

    def __get_test_list():
        test_list_path = 'data/clf/test.tsv'
        test_data_list = pd.read_csv(test_list_path, sep='\t', usecols=['file_name'])
        test_data_list = pd.DataFrame(test_data_list).to_records(index=False)
        test_data_list = [data[0] for data in test_data_list]
        return test_data_list

    img_average = None
    # if use_average_image:
    #     img_average = 'data/clf/ave_%s_%s.png' % (img_size, img_type)

    # train, val
    logging.info('Loading train, val dataset...')
    labeled = WarpedLabeledImageDataset(
        __get_train_list(),
        root='data/clf/train_images_labeled',
        use_memory=use_memory,
        img_size=img_size,
        img_type=img_type,
        img_average=img_average
    )
    logging.info('Done.')

    # test
    logging.info('Loading test dataset...')
    test = WarpedImageDataset(
        __get_test_list(),
        root='data/clf/test_images',
        use_memory=use_memory,
        img_size=img_size,
        img_type=img_type,
        img_average=img_average
    )
    logging.info('Done.')

    if split_val is not False:
        train, val = split_dataset(labeled, int(len(labeled)*split_val))
        return train, val, test
    else:
        train = labeled
        return train, test

if __name__ == '__main__':
    pass
    # def _show(ary):
    #     ary = ary.transpose((1, 2, 0))
    #     ary = (ary - np.min(ary)) / (np.max(ary) - np.min(ary)) * 255.
    #     ary = np.uint8(ary)
    #     image = Image.fromarray(ary)
    #     image.show()
    # train, val, test = get_clf_data(use_memory=False, use_average_image=True)
    # train1, val1, test1 = get_clf_data(use_memory=False, use_average_image=False)
    # _show(train.get_example(0)[0])
    # _show(val.get_example(0)[0])
    # _show(test.get_example(0))
    # _show(train1.get_example(0)[0])
    # _show(val1.get_example(0)[0])
    # _show(test1.get_example(0))
