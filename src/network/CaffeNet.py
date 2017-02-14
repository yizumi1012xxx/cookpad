# -*- coding: utf-8 -*-
'''
CaffeNet.py
'''

import chainer.functions as F
import chainer.links as L
from chainer import Chain

class CaffeNet(Chain):
    '''CaffeNet'''

    def __init__(self):
        super(CaffeNet, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=4),
            conv2=L.Convolution2D(96, 256, 5, pad=2),
            conv3=L.Convolution2D(256, 384, 3, pad=1),
            conv4=L.Convolution2D(384, 384, 3, pad=1),
            conv5=L.Convolution2D(384, 256, 3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 25),
        )
        self.train = True

    def __call__(self, x):
        h = F.local_response_normalization(F.max_pooling_2d(F.relu(self.conv1(x))), 3, stride=2)    
        h = F.local_response_normalization(F.max_pooling_2d(F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        y = self.fc8(h)
        return y
