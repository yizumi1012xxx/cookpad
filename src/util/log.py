# -*- coding: utf-8 -*-
'''
image_info.py
'''

import logging
import pandas as pd


logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(module)s::%(funcName)s called:(%(lineno)d) - %(message)s',
    level=logging.INFO
)


def to_csv(file_path, output_path):
    pd.read_json(file_path).to_csv(output_path)


if __name__ == '__main__':
    # to_csv('result/alexnet_pretrain_crop/log', 'result/alexnet_pretrain_crop/log.csv')
    # to_csv('result/alexnet_pretrain_warp/log', 'result/alexnet_pretrain_warp/log.csv')
    # to_csv('result/alexnet_pretrain_warp_b50/log', 'result/alexnet_pretrain_warp_b50/log.csv')
    # to_csv('result/alexnet_scratch_warp/log', 'result/alexnet_scratch_warp/log.csv')
    # to_csv('result/resnet50_pretrain_warp/log', 'result/resnet50_pretrain_warp/log.csv')
    to_csv('result/resnet101_pretrain_warp/log', 'result/resnet101_pretrain_warp/log.csv')
