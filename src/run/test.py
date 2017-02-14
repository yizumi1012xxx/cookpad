# -*- coding: utf-8 -*-
'''
test.py
'''


import os
import logging
from chainer import Variable
import numpy as np
import pandas as pd
import sklearn.metrics


def test_network(test, model, output_path='', batch_size=100, device_id=-1):
    '''
    Test.
    Args:
        dataset (chainer.dataset): Dataset
        model (chainer.links): Model
        output_path (str): Output path
        batch_size (int): Batch size
        device_id (int): Device id
    '''
    logging.info('Test network... Start.')
    prediction = []
    # set model
    model.train = False
    if device_id > -1:
        model.to_gpu()
    # set dataset
    for i in range(0, len(test), batch_size):
        data = test[i:i+batch_size]
        x = Variable(np.array(data))
        if device_id > -1:
            x.to_gpu()
        prediction.extend(np.argmax(model.predictor(x).data, axis=1))
    if output_path != '':
        pd.DataFrame(prediction).to_csv(os.path.join(output_path, 'prediction.csv'), header=None)
    logging.info('Test network... Done.')
