# -*- coding: utf-8 -*-
'''
caffe.py
'''

import os
import logging
import six
import chainer
from chainer.links import caffe
from chainer import link
from AlexNet import AlexNet
from CaffeNet import CaffeNet
from GoogLeNet import GoogLeNet
from ResNet50 import ResNet50
from ResNet101 import ResNet101
from ResNet152 import ResNet152
from SppNet import SppNet


def load_caffe_model(model_name, output_folder_path='model', copy_model=None):
    '''
    Load caffe model and save as chainer model.
    Args:
        model_name (str): Model name
            Choose from alexnet, caffenet, googlenet, resnet.
        output_path (str): Output path
    '''
    def __download_model(model_name):
        model = {
            'alexnet' : {
                'url' : 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
                'name' : 'bvlc_alexnet.caffemodel'
            },
            'caffenet' : {
                'url' : 'http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel',
                'name' : 'bvlc_reference_caffenet.caffemodel'
            },
            'googlenet' : {
                'url' : 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
                'name' : 'bvlc_googlenet.caffemodel'
            },
            'resnet50' : {
                'url' : None,
                'name' : 'ResNet-50-model.caffemodel'
            }
        }
        if not model_name in model:
            raise RuntimeError('Choose from alexnet, caffenet, googlenet and resnet.')

        url = model[model_name]['url']
        name = model[model_name]['name']

        if not os.path.isdir(os.path.join(output_folder_path, model_name)):
            os.mkdir(os.path.join(output_folder_path, model_name))

        output_path = os.path.join(output_folder_path, model_name, name)
        if not os.path.isfile(output_path) and url is not None:
            six.moves.urllib.request.urlretrieve(url, output_path)
        else:
            logging.info('Already model file exist.')
        return output_path

    logging.info('Downloading caffe model...')
    file_path = __download_model(model_name)
    logging.info('Done.')

    logging.info('Converting to chainer model...')
    model = caffe.CaffeFunction(file_path)
    logging.info('Done.')

    if output_folder_path:
        logging.info('Saving chainer model...')
        output_path = os.path.join(output_folder_path, model_name, 'import.npz')
        chainer.serializers.save_npz(output_path, model)
        logging.info('Done.')

    if output_folder_path and copy_model is not None:
        logging.info('Saving chainer model...')
        __copy_model(model, copy_model)
        output_path = os.path.join(output_folder_path, model_name, 'import2.npz')
        chainer.serializers.save_npz(output_path, copy_model)
        logging.info('Done.')

    return model


def __copy_model(src, dst):
    '''
    Copy model.
    Args:
        src (model): Copy source model
        dst (model): Copy destination model
    '''
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    # logging.info(dst.__dict__)
    for child in src.children():
        logging.info(child.name)
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, link.Chain):
            __copy_model(child, dst_child)
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


# def spp():
#     alex = AlexNet()
#     chainer.serializers.load_npz('model/alexnet/import2.npz', alex)
#     dst = SppNet()
#     logging.info('1Done.')
#     __copy_model(alex, dst)
#     logging.info('2Done.')
#     chainer.serializers.save_npz('model/import.npz', dst)
#     logging.info('3Done.')


def resnet():
    # resnet 50
    src = ResNet50(output=1000)
    dst = ResNet50(output=25)
    src_filename = 'model/resnet50/import.hdf5'
    dst_filename = 'model/resnet50/import2.npz'
    chainer.serializers.load_hdf5(src_filename, src)
    __copy_model(src, dst)
    chainer.serializers.save_npz(dst_filename, dst)

    # resnet 101
    src = ResNet101(output=1000)
    dst = ResNet101(output=25)
    src_filename = 'model/resnet101/import.hdf5'
    dst_filename = 'model/resnet101/import2.npz'
    chainer.serializers.load_hdf5(src_filename, src)
    __copy_model(src, dst)
    chainer.serializers.save_npz(dst_filename, dst)

    # resnet 152
    src = ResNet152(output=1000)
    dst = ResNet152(output=25)
    src_filename = 'model/resnet152/import.hdf5'
    dst_filename = 'model/resnet152/import2.npz'
    chainer.serializers.load_hdf5(src_filename, src)
    __copy_model(src, dst)
    chainer.serializers.save_npz(dst_filename, dst)


def __main():
    root_path = os.path.normpath(os.path.join(
        os.path.abspath(__file__),
        os.pardir,
        os.pardir,
        os.pardir))
    os.chdir(root_path)
    # load_caffe_model('alexnet', copy_model=AlexNet())
    # load_caffe_model('caffenet', copy_model=CaffeNet())
    # load_caffe_model('googlenet')
    # load_caffe_model('resnet50', copy_model=ResNet50())
    resnet()

if __name__ == '__main__':
    __main()
