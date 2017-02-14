# -*- coding: utf-8 -*-
'''
image_info.py
'''

import os
import logging
import pandas as pd
import numpy as np
from PIL import Image


logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(module)s::%(funcName)s called:(%(lineno)d) - %(message)s',
    level=logging.INFO
)


def size_check():
    labeled_list_path = 'data/clf/train_master.tsv'
    labeled_image_folder_path = 'data/clf/train_images_labeled'
    labeled_list = pd.read_csv(labeled_list_path, sep='\t', usecols=['file_name', 'category_id'])
    labeled_list = pd.DataFrame(labeled_list).to_records(index=False)

    file_name = []
    file_width = []
    file_height = []

    for f_name, _ in labeled_list:
        image_path = os.path.join(labeled_image_folder_path, f_name)
        image = Image.open(image_path)
        file_name.append(f_name)
        file_width.append(image.size[0])
        file_height.append(image.size[1])

    data = {
        'file_name' : file_name,
        'file_width' : file_width,
        'file_height' : file_height
    }

    pd.DataFrame(data=data).to_csv('result/size.csv')


def get_average_image(img_size=224, img_type='warp'):

    logging.info('Get average image...')

    average_image = np.zeros((224, 224, 3))

    train_list_path = 'data/clf/train_master.tsv'
    train_image_path = 'data/clf/train_images_labeled'
    labeled_list = pd.read_csv(train_list_path, sep='\t', usecols=['file_name', 'category_id'])

    records = pd.DataFrame(labeled_list).to_records(index=False)
    record_num = len(records)

    for file_name, _ in records:
        image_path = os.path.join(train_image_path, file_name)
        image = Image.open(image_path)
        if img_type == 'warp':
            image = image.resize((img_size, img_size))
        elif img_type == 'crop':
            width, height = image.size
            if width > height:
                diff = (width - height) / 2
                box = (diff, 0, height + diff, height)
                image.crop(box)
            elif height > width:
                diff = (height - width) / 2
                box = (0, diff, width, width + diff)
                image.crop(box)
            image = image.resize((img_size, img_size))
        image = np.asarray(image, dtype=np.float)
        average_image += image / record_num
    Image.fromarray(np.uint8(average_image)).save('result/ave_%s_%s.png' % (img_size, img_type))
    logging.info('Done.')


if __name__ == '__main__':
    get_average_image(img_type='crop')
