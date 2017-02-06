# -*- coding: utf-8 -*-
'''
clf.py
'''

import os
import pandas as pd
import numpy as np
from PIL import Image

class ImageData():
    '''ImageData'''

    def __init__(self, img_dir, meta_data):
        self.img_dir = img_dir
        assert meta_data.endswith('.tsv')
        self.meta_data = pd.read_csv(meta_data, sep='\t')
        self.index = np.array(self.meta_data.index)
        self.split = False

    def shuffle(self):
        assert self.split == True
        self.train_index = np.random.permutation(self.train_index)

    def split_train_val(self, train_size):
        self.train_index = np.random.choice(self.index, train_size, replace=False)
        self.val_index = np.array([i for i in self.index if i not in self.train_index])
        self.split = True

    def generate_minibatch(self, batchsize, img_size = 224, mode = None):
        i = 0
        if mode == 'train':
            assert self.split == True
            meta_data = self.meta_data.ix[self.train_index]
            index = self.train_index

        elif mode == 'val':
            assert self.split == True
            meta_data = self.meta_data.ix[self.val_index]
            index = self.val_index
        else:
            meta_data = self.meta_data
            index = self.index

        while i < len(index):
            data = meta_data.iloc[i:i + batchsize]
            images = []
            for f in list(data['file_name']):
                image = Image.open(os.path.join(self.img_dir, f))
                image = image.resize((img_size, img_size))
                images.append(np.array(image))
            images = np.array(images)
            images = images.transpose((0, 3, 1, 2))
            images = images.astype(np.float32)

            if 'category_id' in data.columns:
                labels = np.array(list(data['category_id']))
                labels = labels.astype(np.int32)
                yield images, labels
            else:
                yield images
            i += batchsize
