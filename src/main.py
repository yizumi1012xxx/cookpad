# -*- coding: utf-8 -*-
'''
main.py
'''


import os
import logging
import random
import chainer
import numpy as np
import pandas as pd
from dataset.clf import get_clf_data
from run.train import train_network
from run.validation import val_network
from run.test import test_network
from optimizer.optimizer import get_optimizer
from network.model import get_model
if chainer.cuda.available:
    import cupy as cp


logging.basicConfig(
    format='%(asctime)s [%(process)5d] %(levelname)8s %(module)s::%(funcName)s called:(%(lineno)d) - %(message)s',
    level=logging.INFO
)

# set Python, Numpy, Cupy random seed
random.seed(100)
np.random.seed(100)


GPU_ID = -1

if chainer.cuda.available:
    cp.random.seed(100)
    GPU_ID = 0
    logging.info('GPU is available')


def __run(dataset, model, optimizer, output_path='', resume_path='', batch_size=100, epoch=50, execute_train=True):

    logging.info('output_path -> %s', output_path)
    logging.info('resume_path -> %s', resume_path)
    logging.info('batch_size  -> %s', batch_size)
    logging.info('epoch       -> %s', epoch)
    logging.info('train       -> %s', execute_train)
    # run
    train, val, test = dataset
    if execute_train:
        train_network(
            (train, val), model, optimizer,
            output_path=output_path,
            resume_path=resume_path,
            batch_size=batch_size,
            epoch=epoch,
            device_id=GPU_ID
        )
    val_network(
        val, model,
        output_path=output_path,
        batch_size=batch_size,
        device_id=GPU_ID
    )
    test_network(
        test, model,
        output_path=output_path,
        batch_size=batch_size,
        device_id=GPU_ID
    )


def __run_from_config(config):
    dataset_config = config['dataset']
    model_config = config['model']
    test_model_config = config['test_model']
    optimizer = config['optimizer']
    output_path = config['output_path']
    resume_path = config['resume_path']
    batch_size = config['batch_size']
    epoch = config['epoch']
    train = config['train']
    # dataset
    dataset = get_clf_data(
        use_memory=dataset_config['use_memory'],
        img_size=dataset_config['img_size'],
        img_type=dataset_config['img_type']
    )
    # model
    if train:
        model = get_model(
            model_config['type'],
            model_config['path']
        )
    else:
        model = get_model(
            test_model_config['type'],
            test_model_config['path']
        )

    # optimizer
    optimizer = get_optimizer(
        optimizer['type'],
        model
    )
    #run
    __run(
        dataset,
        model,
        optimizer,
        output_path=output_path,
        resume_path=resume_path,
        batch_size=batch_size,
        epoch=epoch,
        execute_train=train
    )


def __main(root_path):
    config_path = os.path.join(root_path, 'src/config.json')
    project = pd.read_json(config_path).to_dict()['project']
    # __run_from_config(project['alex_scratch_warp'])
    # __run_from_config(project['alex_scratch_crop'])           # no calculated
    # __run_from_config(project['alexnet_pretrain_warp'])
    # __run_from_config(project['alexnet_pretrain_crop'])
    # __run_from_config(project['resnet50_pretrain_warp'])
    # __run_from_config(project['resnet50_pretrain_warp_adam']) # calculating
    __run_from_config(project['resnet101_pretrain_warp'])     # calculating
    # __run_from_config(project['resnet152_pretrain_warp'])     # calculating

if __name__ == '__main__':
    __main(os.path.normpath(os.path.join(os.path.abspath(__file__), '../../')))
