# -*- coding: utf-8 -*-
'''
model.py
'''


import chainer
from network.AlexNet import AlexNet
from network.ResNet50 import ResNet50
from network.ResNet101 import ResNet101
from network.ResNet152 import ResNet152
from network.MLP2 import MLP2

def get_model(name, pretrained_path='', output=25):
    classifier = False

    if name == 'AlexNet':
        model = AlexNet()
    elif name == 'AlexNet-cls':
        model = chainer.links.Classifier(AlexNet())
        classifier = True
    elif name == 'ResNet50':
        model = ResNet50(output=output)
    elif name == 'ResNet50-cls':
        model = chainer.links.Classifier(ResNet50(output=output))
        classifier = True
    elif name == 'ResNet101':
        model = ResNet101(output=output)
    elif name == 'ResNet101-cls':
        model = chainer.links.Classifier(ResNet101(output=output))
        classifier = True
    elif name == 'ResNet152':
        model = ResNet152(output=output)
    elif name == 'ResNet152-cls':
        model = chainer.links.Classifier(ResNet152(output=output))
        classifier = True
    elif name == 'MLP2':
        model = MLP2(output=output)
    elif name == 'MLP2-cls':
        model = chainer.links.Classifier(MLP2(output=output))
        classifier = True
    else:
        raise Exception('model name is invalid -> %s' % name)

    if pretrained_path != '':
        chainer.serializers.load_npz(pretrained_path, model)

    if not classifier:
        model = chainer.links.Classifier(model)
    return model
