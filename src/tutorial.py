# -*- coding: utf-8 -*-
'''
tutorial.py
'''

import logging
import pandas as pd
import numpy as np
import chainer
import chainer.functions as F
# from chainer.links import caffe
from chainer import optimizers, Variable
from dataset.clf import ImageData
from network.Alex import Alex
# from network.fine_tuning import download_model, copy_model

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(module)s::%(funcName)s called:(%(lineno)d) - %(message)s',
    level=logging.DEBUG
)


def train_val(train_data, classifier, optimizer, num_train=9000, epochs=1, gpu=True):
    '''
    train
    '''
    batchsize = 100
    # split data to train and val
    train_data.split_train_val(num_train)
    for epoch in range(epochs):
        # train
        classifier.predictor.train = True
        num_samples = 0
        train_cum_loss = 0
        train_cum_acc = 0
        for data in train_data.generate_minibatch(batchsize, mode='train'):
            num_samples += len(data[0])
            optimizer.zero_grads()
            x, y = Variable(data[0]), Variable(data[1])
            if gpu:
                x.to_gpu()
                y.to_gpu()
            loss = classifier(x, y)
            train_cum_acc += classifier.accuracy.data * batchsize
            train_cum_loss += classifier.loss.data * batchsize
            loss.backward()
            optimizer.update()
            logging.info('%s/%s(epoch:%s)', num_samples, len(train_data.train_index), (epoch+1))
        train_accuracy = train_cum_acc / num_samples
        train_loss = train_cum_loss / num_samples
        # validation
        classifier.predictor.train = False
        num_samples = 0
        val_cum_loss = 0
        val_cum_acc = 0
        for data in train_data.generate_minibatch(batchsize, mode='val'):
            num_samples += len(data[0])
            x, y = Variable(data[0]), Variable(data[1])
            if gpu:
                x.to_gpu()
                y.to_gpu()
            loss = classifier(x, y)
            val_cum_acc += classifier.accuracy.data*batchsize
            val_cum_loss += classifier.loss.data*batchsize
            logging.info('%s/%s(epoch:%s)', num_samples, len(train_data.train_index), (epoch+1))
        val_accuracy = val_cum_acc/num_samples
        val_loss = val_cum_loss/num_samples

        logging.info('-----------------epoch:%s-----------------', epoch+1)
        logging.info('train_accuracy:%strain_loss:%s', train_accuracy, train_loss)
        logging.info('val_accuracy:%sval_loss:%s', val_accuracy, val_loss)
        # shuffle train data
        train_data.shuffle()
    return classifier, optimizer


def predict(test_data, classifier, batchsize=5, gpu=True):
    '''
    predict
    '''
    if gpu:
        classifier.predictor.to_gpu()
    else:
        classifier.predictor.to_cpu()
    classifier.predictor.train = False
    num_samples = 0
    predictions = np.zeros((len(test_data.index), 25))
    for data in test_data.generate_minibatch(batchsize):
        num_samples += len(data)
        logging.info('%s / %s', num_samples, len(test_data.index))
        x = Variable(data)
        if gpu:
            x.to_gpu()
        yhat = classifier.predictor(x)
        yhat = F.softmax(yhat)
        yhat.to_cpu()
        predictions[num_samples-len(data):num_samples, :] = yhat.data
    return predictions


def __main():
    # network
    alex = Alex()
    # copy_model(caffe.CaffeFunction(download_model('alexnet')), alex)
    alex.to_gpu()
    classifier = chainer.links.Classifier(alex)
    optimizer = optimizers.MomentumSGD(lr=0.0005)
    optimizer.setup(classifier)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # dataset
    train_data = ImageData(
        'data/clf/train_images_labeled',
        'data/clf/train_master.tsv'
    )
    test_data = ImageData(
        'data/clf/test_images',
        'data/clf/test.tsv'
    )

    # train
    classifier, optimizer = train_val(train_data, classifier, optimizer, gpu=True)

    # test
    p = predict(test_data, classifier, gpu=True)
    pd.DataFrame(p.argmax(axis=1)).to_csv('result/sample_submit.csv', header=None)

if __name__ == '__main__':
    __main()
