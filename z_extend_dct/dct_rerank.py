# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors.dist_metrics import DistanceMetric

from generatedct import DCT_binaray


def dct_rerank(query_img, img_list, num_output):
    dist = DistanceMetric.get_metric('euclidean')
    feats = []
    # query_feat = get_dct_feature(query_img)
    query_feat = DCT_binaray(query_img)
    for img in img_list:
        # dct_feat = get_dct_feature(img)
        dct_feat = DCT_binaray(img)
        feats.append(dct_feat)
    feats = np.array(feats)

    # scores = np.dot(query_feat,feats.T)
    a = []
    a.append(query_feat)
    scores = dist.pairwise(feats, a)
    scores = scores.flatten()
    rank_ID = np.argsort(scores)
    rank_score = scores[rank_ID]

    imlist = [img_list[index] for i, index in enumerate(rank_ID[0:num_output])]
    print rank_score
    return imlist


if __name__ == '__main__':
    img_list = ['./all_souls_000000.jpg', './all_souls_000000_blur.jpg', './all_souls_000000_blur1.jpg',
                './all_souls_000001.jpg',
                './all_souls_000002.jpg']

    print dct_rerank('./all_souls_000000.jpg', img_list, 20)
