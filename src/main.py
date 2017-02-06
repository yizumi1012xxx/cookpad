# -*- coding: utf-8 -*-
'''
main.py
'''

import os
import logging
import chainer
from chainer.training import extensions
import numpy as np
import sklearn.metrics
from dataset.clf import ImageData
from network.SPPnet import SPPnet
from common import json_read


logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(module)s::%(funcName)s called:(%(lineno)d) - %(message)s',
    level=logging.DEBUG
)


class TestModeEvaluator(extensions.Evaluator):
    '''TestModeEvaluator'''

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def train_network(dataset, model, output_path='', model_path='', resume_path='', epoch=100):
    '''
    Train.
    Args:
        dataset (str): Dataset
        model (str): Model
        output_path (str): Output path
        model_path (str): Load model path
        resume_path (str): Load resume path
        epoch (int): Epoch
    Returns:
        Confusion matrix (array)
    '''
    batch_size = 100
    # set dataset
    train, test = dataset
    # set model
    if model_path != '':
        logging.info('Load model. -> %s', model_path)
        chainer.serializers.load_npz(model_path, model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    # set config
    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=output_path)
    trainer.extend(TestModeEvaluator(test_iter, model, device=-1))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport([
            'epoch',
            'main/loss',
            'main/accuracy',
            # 'validation/main/loss',
            # 'validation/main/accuracy'
        ])
    )
    trainer.extend(extensions.ProgressBar(update_interval=5))
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'),
        trigger=(5, 'epoch')
    )
    trainer.extend(
        extensions.snapshot_object(model, filename='model_epoch_{.updater.epoch}'),
        trigger=(5, 'epoch')
    )
    if resume_path != '':
        chainer.serializers.load_npz(resume_path, trainer)
    # execute
    trainer.run()


def test_network(dataset, model, model_path):
    '''
    Test.
    Args:
        dataset (str): Dataset
        model (str): Model
        output_path (str): Output path
        model_path (str): Load model path
    Returns:
        Confusion matrix (array)
    '''

    def get_confusion_matrix(references, predictions, labels=None):
        '''
        Get coufusion matrix.
        Args:
            references (1d-array): Reference data
            predictions (1d-array): Prediction data
            labels (1d-array): Labels
        Returns:
            Confusion matrix (2d-array)
        '''
        if labels is None:
            if len(np.unique(references)) > 7:
                labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                labels = [0, 1, 2, 3, 4, 5, 6]
        return sklearn.metrics.confusion_matrix(references, predictions, labels=labels)

    # set dataset
    _, test = dataset
    data, label = test._datasets
    # set model
    logging.info('Load model. -> %s', model_path)
    model.train = False
    chainer.serializers.load_npz(model_path, model)
    # test
    model(data, label)
    prediction_probability = model.y.data
    prediction = np.argmax(prediction_probability, axis=1)
    return get_confusion_matrix(label, prediction)


def __main(root_path):
    config = json_read(os.path.join(root_path, 'src/config.json'))
    # dataset = get_clf_dataset(config)
    # model = SPPnet(channel=3, c1=16, c2=32, f1=672, f2=512, output=25)
    # output_path = os.path.join(root_path, 'result')
    # train_network(dataset, model, output_path)

if __name__ == '__main__':
    __main(os.path.normpath(os.path.join(os.path.abspath(__file__), '../../')))
