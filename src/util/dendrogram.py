# -*- coding: utf-8 -*-
'''
dendrogram.py
'''

import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt



CONFUSION_MATRIX = [
    [32, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 2, 0, 0, 1, 0, 0],
    [2, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 23, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0],
    [1, 1, 5, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 18, 3, 1, 0, 4, 2, 1, 1, 2, 0, 1, 3, 3, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 27, 3, 1, 3, 0, 0, 1, 1, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 26, 0, 6, 6, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 33, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 2, 3, 0, 15, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 2, 0, 7, 0, 10, 24, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 3, 0, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 2, 0, 8, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 3, 3, 27, 2, 4, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 35, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 2],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 2, 34, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 4, 23, 0, 5, 1, 0, 0, 0, 1, 0, 1],
    [2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 3, 2, 1, 0, 0, 18, 0, 0, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 26, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 33, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 29, 0, 0, 2, 0, 7],
    [0, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 28, 3, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 23, 0, 1, 1],
    [2, 0, 4, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 3, 29, 2, 2],
    [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 31, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 6, 2, 3, 0, 0, 0, 30],
]

LABELS = [
    'bread_sandwich',
    'bread_sliced',
    'bread_sweets',
    'bread_table',
    'noodle_somen',
    'noodle_udon',
    'pasta_cream',
    'pasta_gratin',
    'pasta_japanese',
    'pasta_oil',
    'pasta_tomato',
    'rice_boiled',
    'rice_bowl',
    'rice_curry',
    'rice_fried',
    'rice_risotto',
    'rice_sushi',
    'soup_miso',
    'soup_potage',
    'sweets_cheese',
    'sweets_cookie',
    'sweets_muffin',
    'sweets_pie',
    'sweets_pound',
    'sweets_pudding'
]


def get_dendrogram(confusion_mat, labels=None, a=0.5):

    # calculate distance matrix
    distance_mat = np.zeros_like(confusion_mat, dtype=np.float)
    row, col = distance_mat.shape
    for r in range(row):
        for c in range(col):
            if r < c:
                val = 1 / (confusion_mat[c][r] + confusion_mat[r][c] + a)
                distance_mat[r][c] = val
                distance_mat[c][r] = val

    # change format
    dist_vec = squareform(distance_mat)
    result = linkage(dist_vec, method='average')

    # draw dendrogram
    dendrogram(result, labels=labels, orientation='right', color_threshold=0.4, leaf_font_size=6)
    plt.savefig('result/dendrogram.png')

if __name__ == '__main__':
    get_dendrogram(CONFUSION_MATRIX, labels=LABELS)
