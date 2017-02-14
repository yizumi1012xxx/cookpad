# -*- coding: utf-8 -*-
'''
MLP2.py
'''

import chainer
import chainer.functions as F
import chainer.links as L


class MLP2(chainer.Chain):

    insize = 2048

    def __init__(self, output=25):
        super(MLP2, self).__init__(
            fc1=L.Linear(2048, 4096),
            fc2=L.Linear(4096, output)
        )
        self.train = True

    def __call__(self, x):
        h = F.dropout(F.relu(self.fc1(x)), train=self.train)
        y = self.fc2(h)
        return y
