# -*- coding: utf-8 -*-
'''
validation.py
'''


import os
import logging
from chainer import Variable
from chainer import cuda
import numpy as np
import pandas as pd
import sklearn.metrics


def __forward(dataset, model, batch_size, device_id):
    predictions = []

    model.train = False
    if device_id > -1:
        model.to_gpu()

    for i in range(0, len(dataset), batch_size):
        data = [_data for _data, _ in dataset[i:i+batch_size]]
        # label = [_label for _, _label in dataset[i:i+batch_size]]

        x = Variable(np.array(data))
        if device_id > -1:
            x.to_gpu()

        prediction = model.predictor(x).data

        if device_id > -1:
            prediction = cuda.to_cpu(prediction)

        predictions.extend(prediction)

    return predictions

def val_network(val, model, output_path='', batch_size=100, device_id=-1):
    '''
    Validation network.
    Args:
        val (chainer.dataset): Dataset
        model (chainer.links): Model
        output_path (str): Output path
        batch_size (int): Batch size
        device_id (int): Device id
    '''
    logging.info('Validation network... Start.')

    preds = []
    for prediction in __forward(val, model, batch_size, device_id):
        order = prediction.argsort()[::-1]
        preds.append(order)
    preds = np.asarray(preds).transpose((1, 0))

    labels = []
    for _, label in val:
        labels.append(label)

    output = {}
    output['label'] = labels
    for i, pred in enumerate(preds):
        if i > 4:
            break
        key = 'pred%s' % (i + 1)
        val = pred
        output[key] = val

    if output_path != '':
        # classification result (Top 5)
        file_path = os.path.join(output_path, 'validation.csv')
        pd.DataFrame(output).to_csv(file_path)
        # confusion matrix
        file_path = os.path.join(output_path, 'validation-cf.csv')
        c_mat = sklearn.metrics.confusion_matrix(labels, preds[0])
        np.savetxt(file_path, c_mat, fmt='%d', delimiter=',')
    logging.info('Validation network... Done.')
