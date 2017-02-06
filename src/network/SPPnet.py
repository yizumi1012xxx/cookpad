# -*- coding: utf-8 -*-
'''
SPPnet.py
'''

import chainer.functions as F
import chainer.links as L
from chainer import Chain

class SPPnet(Chain):
    '''SPPnet'''

    def __init__(self, channel=1, c1=16, c2=32, f1=672, f2=512, output=10):
        super(SPPnet, self).__init__(
            conv1=L.Convolution2D(channel, c1, 3),
            conv2=L.Convolution2D(c1, c2, 3),
            l1=L.Linear(f1, f2),
            l2=L.Linear(f2, output)
        )
        self.train = True

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.dropout(F.relu(self.l1(h)), train=self.train)
        y = self.l2(h)
        return y
