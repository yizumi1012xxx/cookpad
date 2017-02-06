# -*- coding: utf-8 -*-
'''
tutorial.py
'''

import os
import logging
import zipfile
import six
from chainer.links import caffe
from chainer import link

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(module)s::%(funcName)s called:(%(lineno)d) - %(message)s',
    level=logging.DEBUG
)

def load_caffe_model(file_path):
    return caffe.CaffeFunction(file_path)


def download_model(model_name):
    if model_name == 'alexnet':
        url = 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
        name = 'bvlc_alexnet.caffemodel'
    elif model_name == 'caffenet':
        url = 'http://dl.caffe.berkeleyvision.org/' \
              'bvlc_reference_caffenet.caffemodel'
        name = 'bvlc_reference_caffenet.caffemodel'
    elif model_name == 'googlenet':
        url = 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
        name = 'bvlc_googlenet.caffemodel'
    elif model_name == 'resnet':
        url = 'http://research.microsoft.com/en-us/um/people/kahe/resnet/' \
              'models.zip'
        name = 'models.zip'
    else:
        raise RuntimeError('Invalid model. Choose from alexnet, caffenet, googlenet and resnet.')
    if not os.path.isfile(name):
        logging.info('Downloading model file...')
        six.moves.urllib.request.urlretrieve(url, name)
        if model_name == 'resnet':
            with zipfile.ZipFile(name, 'r') as zf:
                zf.extractall('.')
        logging.info('Done.')
    return name


def copy_model(src, dst):
    '''
    Copy model
    Args:
        src (model): copy source model
        dst (model): copy destination model
    '''
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for param_a, param_b in zip(child.namedparams(), dst_child.namedparams()):
                if param_a[0] != param_b[0]:
                    match = False
                    break
                if param_a[1].data.shape != param_b[1].data.shape:
                    match = False
                    break
            if not match:
                logging.info('Ignore %s because of parameter mismatch', child.name)
                continue
            for param_a, param_b in zip(child.namedparams(), dst_child.namedparams()):
                param_b[1].data = param_a[1].data
            logging.info('Copy %s', child.name)
