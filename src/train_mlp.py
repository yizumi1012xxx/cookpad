# -*- coding: utf-8 -*-
'''
main.py
'''


import os
import logging
import random
import numpy as np
import pandas as pd
import chainer
from chainer.datasets import split_dataset, tuple_dataset
from chainer import Variable
from chainer import cuda
from dataset.clf import get_clf_data
from network.model import get_model
from optimizer.optimizer import get_optimizer
from run.train import train_network
from run.validation import val_network
if chainer.cuda.available:
    import cupy as cp


logging.basicConfig(
    format='%(asctime)s [%(process)5d] %(levelname)8s %(module)s::%(funcName)s called:(%(lineno)d) - %(message)s',
    level=logging.INFO
)


# set Python, Numpy, Cupy random seed
random.seed(100)
np.random.seed(100)
if chainer.cuda.available:
    cp.random.seed(100)


GPU_ID = -1
if chainer.cuda.available:
    GPU_ID = 0


def __get_dataset(root_path):

    def __get_hidden_layer_value(val, model, batch_size):
        # set model
        model.train = False
        if GPU_ID > -1:
            model.to_gpu()
        # set dataset
        outputs = []
        labels = []
        for i in range(0, len(val), batch_size):
            logging.info('forwarding... [%s / %s]', i+batch_size, len(val))
            data = [_data for _data, _ in val[i:i+batch_size]]
            label = [_label for _, _label in val[i:i+batch_size]]
            x = Variable(np.array(data))
            if GPU_ID > -1:
                x.to_gpu()
            output = model.predictor.get_features(x).data
            if GPU_ID > -1:
                output = cuda.to_cpu(output)
            outputs.extend(output)
            labels.extend(label)
        return outputs, labels

    output_path = os.path.join(root_path, 'result/resnet50_pretrain_warp/dataset.npz')
    if output_path != '' and os.path.isfile(output_path):
        np_file = np.load(output_path)
        data = np_file['data']
        label = np_file['label']
    else:
        # get root dataset
        train, _ = get_clf_data(use_memory=False, img_size=224, img_type='warp', split_val=False)
        # model
        model_path = os.path.join(root_path, 'result/resnet50_pretrain_warp/model_epoch_100')
        model = get_model('ResNet50-cls', pretrained_path=model_path)
        # get data and label
        data, label = __get_hidden_layer_value(train, model, 10)
        # save data and label
        if output_path != '':
            logging.info('saving...')
            np.savez_compressed(output_path, data=np.array(data), label=np.array(label))
    return tuple_dataset.TupleDataset(data, label)


def __filter_class(dataset, extract_class):
    target_data = []
    target_label = []
    for data, label in dataset:
        if label in extract_class:
            target_data.append(data)
            target_label.append(extract_class.index(label))
    target_data = np.array(target_data)
    target_label = np.array(target_label, dtype=np.int32)

    dataset = tuple_dataset.TupleDataset(target_data, target_label)
    train, val = split_dataset(dataset, int(len(dataset) * 0.9))
    return train, val


def __train(root_path):
    category = {
        'category_1' : [14, 11, 12, 13, 5, 4, 17, 15],
        'category_2' : [18, 7],
        'category_3' : [9, 8, 6, 10],
        'category_4' : [23, 22, 21, 16, 0, 24, 19],
        'category_5' : [20, 1, 3, 2]
    }
    dataset = __get_dataset(root_path)
    for category_name, category_ids in category.items():
        output_path = os.path.join(
            root_path, 'result/resnet50_pretrain_warp/%s' % category_name)
        train, val = __filter_class(dataset, category_ids)
        model = get_model(
            'MLP2-cls',
            output=len(category_ids)
        )
        optimizer = get_optimizer('MomentumSGD', model)
        train_network(
            (train, val),
            model,
            optimizer,
            output_path=output_path,
            save_epoch=10
        )
        val_network(val, model, output_path=output_path)


def __val(root_path):

    def __get_hidden_layer_values(dataset, model, batch_size=100):
        # set model
        model.train = False
        if GPU_ID > -1:
            model.to_gpu()
        # set dataset
        outputs = []
        for i in range(0, len(dataset), batch_size):
            logging.info('forwarding... [%s / %s]', i+batch_size, len(dataset))
            data = [_data for _data, _ in dataset[i:i+batch_size]]
            x = Variable(np.array(data))
            if GPU_ID > -1:
                x.to_gpu()
            output = model.predictor.get_features(x).data
            if GPU_ID > -1:
                output = cuda.to_cpu(output)
            outputs.extend(output)
        return outputs

    def __get_output_layer_values(dataset, model, batch_size=100):
        # set model
        model.train = False
        if GPU_ID > -1:
            model.to_gpu()
        # set dataset
        outputs = []
        for i in range(0, len(dataset), batch_size):
            logging.info('forwarding... [%s / %s]', i+batch_size, len(dataset))
            data = [_data for _data, _ in dataset[i:i+batch_size]]
            x = Variable(np.array(data))
            if GPU_ID > -1:
                x.to_gpu()
            output = model.predictor(x).data
            if GPU_ID > -1:
                output = cuda.to_cpu(output)
            outputs.extend(np.argmax(output, axis=1))
        return outputs


    train, val, test = get_clf_data(use_memory=False, img_size=224, img_type='warp')

    val_label = []
    for _data, _label in val:
        val_label.append(_label)

    # model
    model_path = os.path.join(root_path, 'result/resnet50_pretrain_warp/model_epoch_100')
    model = get_model('ResNet50-cls', pretrained_path=model_path)
    hidden_layer_data = __get_hidden_layer_values(val, model, batch_size=10)
    outputs_class = __get_output_layer_values(val, model, batch_size=10)
    # predict
    predictions = __test2(hidden_layer_data, outputs_class)
    output_path = os.path.join(root_path, 'result/resnet50_pretrain_warp')
    if output_path != '':
        file_name = os.path.join(output_path, 'val.csv')
        pd.DataFrame({
            'test1' : outputs_class,
            'test2' : predictions,
            'label' : val_label
        }).to_csv(file_name)
    logging.info('Validation network... Done.')


def __test(root_path):

    def __get_hidden_layer_values(dataset, model, batch_size=100):
        # set model
        model.train = False
        if GPU_ID > -1:
            model.to_gpu()
        # set dataset
        outputs = []
        for i in range(0, len(dataset), batch_size):
            logging.info('forwarding... [%s / %s]', i+batch_size, len(dataset))
            data = [_data for _data in dataset[i:i+batch_size]]
            x = Variable(np.array(data))
            if GPU_ID > -1:
                x.to_gpu()
            output = model.predictor.get_features(x).data
            if GPU_ID > -1:
                output = cuda.to_cpu(output)
            outputs.extend(output)
        return outputs

    def __get_output_layer_values(dataset, model, batch_size=100):
        # set model
        model.train = False
        if GPU_ID > -1:
            model.to_gpu()
        # set dataset
        outputs = []
        for i in range(0, len(dataset), batch_size):
            logging.info('forwarding... [%s / %s]', i+batch_size, len(dataset))
            data = [_data for _data in dataset[i:i+batch_size]]
            x = Variable(np.array(data))
            if GPU_ID > -1:
                x.to_gpu()
            output = model.predictor(x).data
            if GPU_ID > -1:
                output = cuda.to_cpu(output)
            outputs.extend(np.argmax(output, axis=1))
        return outputs

    _, test = get_clf_data(use_memory=False, img_size=224, img_type='warp', split_val=False)
    # model
    model_path = os.path.join(root_path, 'result/resnet50_pretrain_warp/model_epoch_100')
    model = get_model('ResNet50-cls', pretrained_path=model_path)
    hidden_layer_data = __get_hidden_layer_values(test, model, batch_size=10)
    outputs_class = __get_output_layer_values(test, model, batch_size=10)
    # predict
    predictions = __test2(hidden_layer_data, outputs_class)
    output_path = os.path.join(root_path, 'result/resnet50_pretrain_warp')
    if output_path != '':
        file_name = os.path.join(output_path, 'prediction-t1.csv')
        pd.DataFrame(outputs_class).to_csv(file_name, header=None)
        file_name = os.path.join(output_path, 'prediction-t2.csv')
        pd.DataFrame(predictions).to_csv(file_name, header=None)
    logging.info('Test network... Done.')


def __test2(hidden_layer_data, outputs_class):

    def __get_output_layer_values(data, model):
        # set model
        model.train = False
        if GPU_ID > -1:
            model.to_gpu()
        # set dataset
        x = Variable(np.array([data]))

        if GPU_ID > -1:
            x.to_gpu()
        output = model.predictor(x).data
        if GPU_ID > -1:
            output = cuda.to_cpu(output)
        return np.argmax(output, axis=1)

    category = {
        'category_1' : [14, 11, 12, 13, 5, 4, 17, 15],
        'category_2' : [18, 7],
        'category_3' : [9, 8, 6, 10],
        'category_4' : [23, 22, 21, 16, 0, 24, 19],
        'category_5' : [20, 1, 3, 2]
    }

    models = {}
    for key, val in category.items():
        path = 'result/resnet50_pretrain_warp/%s/model_epoch_100' % key
        models[key] = get_model(
            'MLP2-cls',
            pretrained_path=path,
            output=len(val)
        )

    preds = []
    for data, category_id in zip(hidden_layer_data, outputs_class):
        for key, val in category.items():
            if category_id in val:
                local_pred = __get_output_layer_values(data, models[key])
                pred = val[local_pred]
                break
        preds.append(pred)
    return preds

def __main(root_path):
    # config_path = os.path.join(root_path, 'src/config.json')
    # config = pd.read_json(config_path).to_dict()
    __train(root_path)
    __val(root_path)
    __test(root_path)

if __name__ == '__main__':
    __main(os.path.normpath(os.path.join(os.path.abspath(__file__), '../../')))
