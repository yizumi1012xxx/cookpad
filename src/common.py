# -*- coding: utf-8 -*-
'''
common.py
'''


import os
import codecs
import json
import logging
import numpy as np
import pandas as pd


logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(module)s::%(funcName)s called:(%(lineno)d) - %(message)s',
    level=logging.DEBUG
)


def json2csv(file_path):
    root, name = os.path.split(file_path)
    file_name, _ = os.path.splitext(name)
    output_path = os.path.join(root, file_name + '.csv')
    data = json_read(file_path)
    pd.DataFrame.from_dict(data).to_csv(output_path)


def csv_write(file_path, data):
    np.savetxt(file_path, data, delimiter=',', fmt='%d')


def csv_read(file_path):
    return np.loadtxt(file_path, delimiter=',', dtype=np.int32)


def json_write(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True, separators=(',', ': '))


def json_read(file_path):
    with codecs.open(file_path, 'r', 'utf-8') as file_object:
        return json.loads(file_object.read())
