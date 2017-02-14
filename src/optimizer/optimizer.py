# -*- coding: utf-8 -*-
'''
optimizer.py
'''


import chainer


def get_optimizer(name, model):
    if name == 'AdaDelta':
        optimizer = chainer.optimizers.AdaDelta(rho=0.95, eps=1e-6)
        optimizer.setup(model)
    elif name == 'AdaGrad':
        optimizer = chainer.optimizers.AdaGrad(lr=0.001, eps=1e-8)
        optimizer.setup(model)
    elif name == 'Adam':
        optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        optimizer.setup(model)
    elif name == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD(lr=0.0005)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
    elif name == 'NesterovAG':
        optimizer = chainer.optimizers.NesterovAG(lr=0.01, momentum=0.9)
        optimizer.setup(model)
    elif name == 'RMSprop':
        optimizer = chainer.optimizers.RMSprop(lr=0.01, alpha=0.99, eps=1e-08)
        optimizer.setup(model)
    elif name == 'RMSpropGraves':
        optimizer = chainer.optimizers.RMSpropGraves(
            lr=0.0001, alpha=0.95, momentum=0.9, eps=0.0001)
        optimizer.setup(model)
    elif name == 'SGD':
        optimizer = chainer.optimizers.SGD(lr=0.01)
        optimizer.setup(model)
    elif name == 'SMORMS3':
        optimizer = chainer.optimizers.SMORMS3(lr=0.001, eps=1e-16)
        optimizer.setup(model)
    else:
        raise Exception('optimizer name is invalid. -> %s' % name)
    return optimizer
