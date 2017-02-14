# -*- coding: utf-8 -*-
'''
train.py
'''


import logging
import chainer
from chainer.training import extensions


class TestModeEvaluator(extensions.Evaluator):
    '''TestModeEvaluator'''

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def train_network(dataset, model, optimizer, output_path='', resume_path='', epoch=100, batch_size=100, device_id=-1, save_epoch=5):
    '''
    Train.
    Args:
        dataset (str): Dataset
        model (str): Model
        output_path (str): Output path
        model_path (str): Load model path
        resume_path (str): Load resume path
        epoch (int): Epoch
    '''

    if device_id > -1:
        chainer.cuda.get_device(device_id).use()
        model.to_gpu()
        logging.info('model to gpu.')
    # set dataset
    train, val = dataset
    # set config
    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    val_iter = chainer.iterators.SerialIterator(val, batch_size, repeat=False, shuffle=False)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device_id)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=output_path)
    trainer.extend(TestModeEvaluator(val_iter, model, device=device_id))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport([
            'epoch',
            'main/loss',
            'main/accuracy',
            'validation/main/loss',
            'validation/main/accuracy'
        ])
    )
    trainer.extend(extensions.ProgressBar(update_interval=5))
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'),
        trigger=(save_epoch, 'epoch')
    )
    trainer.extend(
        extensions.snapshot_object(model, filename='model_epoch_{.updater.epoch}'),
        trigger=(save_epoch, 'epoch')
    )
    if resume_path != '':
        chainer.serializers.load_npz(resume_path, trainer)
    trainer.run()
