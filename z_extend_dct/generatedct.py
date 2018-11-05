# -*- coding: utf-8 -*-
'''
提取dct特征：获取图片整体的dc和ac系数特征。
'''
import sys

import cv2


def DCT_binaray(image1, size=(32, 32), part_size=(16, 16)):
    """
     original dct xishu, not binaray .
    'Size' is parameter what the image will resize to it and then image will be compared by the pHash.
    It's 32 * 32 when it default.
    'part_size' is a size of a part of the matrix after Discrete Cosine Transform,which need to next steps.
    It's 8 * 8 when it default.

    The function will return the hamming code,less is correct.
    """
    assert size[0] == size[1], "size error"
    assert part_size[0] == part_size[1], "part_size error"

    image1 = cv2.resize(cv2.imread(image1), size)
    img_YUV = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)[:, :, 0]
    img1 = img_YUV.astype('float')  # 将uint8转化为float类型
    img_dct = cv2.dct(img1)  # 进行离散余弦变换
    # img_dct_log = np.log(abs(img_dct))  # 进行log处理
    sub_dct_list = sub_matrix_to_list(img_dct, part_size)
    # code1 = get_code_linear(sub_dct_list)
    code1 = sub_dct_list[1::]
    return code1


def sub_matrix_to_list(DCT_matrix, part_size):
    w, h = part_size
    sub_dct_list = []
    for i in range(0, h):
        for j in range(0, w):
            sub_dct_list.append(DCT_matrix[i][j])
    return sub_dct_list


def get_code_linear(lists):
    '''
    比较列表中相邻元素大小
    '''
    j = len(lists) - 1
    hash_list = []
    m, n = 0, 1
    for i in range(j):
        if lists[m] > lists[n]:
            hash_list.append(1)
        else:
            hash_list.append(0)
        m += 1
        n += 1
    hash_list.append(0) if lists[0] > lists[j] else hash_list.append(1)
    return hash_list


def code_compare(code1, code2):
    num = 0
    for index in range(0, len(code1)):
        if str(code1[index]) != str(code2[index]):
            num += 1
    return num


def apply_blur(image_name):
    image = cv2.imread(image_name)
    image = cv2.GaussianBlur(image, (99, 99), sigmaX=0)
    cv2.imwrite(image_name.replace('.jpg', '_blur.jpg'), image, )


sys.path.append('/shihuijie/software/faiss/python/')

import faiss
import numpy as np
from numpy import linalg as LA

if __name__ == '__main__':
    # print code_compare(DCT_binaray('./all_souls_000000.jpg'),
    #                    DCT_binaray('./all_souls_000000_blur1.jpg'))
    #
    # print code_compare(DCT_binaray('./all_souls_000000.jpg'),
    #                    DCT_binaray('./all_souls_000000_blur.jpg'))
    # apply_blur('./all_souls_000000.jpg')
    feats = []
    imgNames = []
    result_rank = {}
    feat_a = DCT_binaray('./all_souls_000000.jpg')
    feats.append(feat_a / LA.norm(feat_a))
    feat_b = DCT_binaray('./all_souls_000000_blur.jpg')
    feats.append(feat_b / LA.norm(feat_b))
    feat_c = DCT_binaray('./all_souls_000000_blur1.jpg')
    feats.append(feat_c / LA.norm(feat_c))
    feat_d = DCT_binaray('./all_souls_000001.jpg')
    feats.append(feat_d / LA.norm(feat_d))
    feat_e = DCT_binaray('./all_souls_000002.jpg')
    feats.append(feat_e / LA.norm(feat_e))
    imgNames.append('./all_souls_000000.jpg')
    imgNames.append('./all_souls_000000_blur.jpg')
    imgNames.append('./all_souls_000000_blur1.jpg')
    imgNames.append('./all_souls_000001.jpg')
    imgNames.append('./all_souls_000002.jpg')

    feats = np.asarray(feats)

    feat_q = DCT_binaray('./all_souls_000000.jpg')
    feats_query = [feat_q / LA.norm(feat_q)]
    names_query = ['./all_souls_000000.jpg']
    d = feats.shape[1]  # dimension
    # index = faiss.IndexFlatL2(d)  # build the index
    index = faiss.IndexFlatIP(d)  # build the index
    print(index.is_trained)
    feats = feats.astype('float32')
    feats_query = np.array(feats_query[:]).astype('float32')
    index.add(feats)  # add vectors to the index
    print(index.ntotal)

    k = 5  # we want to see 4 nearest neighbors
    # D, I = index.search(xb[:5], k)  # sanity check
    # print(I) #相似位置
    # print(D)# 相似度
    D, I = index.search(feats_query, k)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])

    [rows, cols] = I.shape
    for i_row in range(rows):
        rank_ID = I[i_row, :]
        num_output = 5

        # print rank_ID[:num_output]

        imlist = [imgNames[index] for j_rank, index in enumerate(rank_ID[0:num_output])]
        result_rank[names_query[i_row]] = imlist
    print result_rank
