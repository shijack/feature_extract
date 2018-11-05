# -*- coding: utf-8 -*-
# Author: shijack
import sys

import time

sys.path.append("..")
# this is used to import faiss made from source code ,else you can use anocado2 faiss
sys.path.append('/shihuijie/software/faiss/python/')

import faiss
import os
import scipy.spatial.distance
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import h5py
from util import utils


def cos_dist(matrix, vector):
    '''
    matrix:(number,feature)
    vector:(1,feature)
    '''
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)


def result_generator_vae(dir_query, h5_file, file_result):
    '''
    this function runs too slow,please use result_generator_cnn_new function !
    :param dir_query:
    :param h5_file:
    :param file_result:
    :return:
    '''
    # read in indexed images' feature vectors and corresponding image names
    h5f = h5py.File(h5_file, 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()
    print "--------------------------------------------------"
    print "               searching starts"
    print "--------------------------------------------------"
    # read and show query image
    img_list = utils.get_all_files_suffix(dir_query)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../model_new/vae-59000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../model_new'))
        graph = tf.get_default_graph()
        x_input = graph.get_tensor_by_name('encoder/input_img:0')
        latent_feature = graph.get_tensor_by_name('variance/latent_feature:0')

        # extract query image's feature, compute simlarity score and sort
        # here is baoli search
        result_rank = {}
        for i, img_path in enumerate(img_list):
            img = utils.img_process(img_path)
            queryVec = sess.run(latent_feature, feed_dict={x_input: img})

            a = []
            a.append(queryVec)
            scores = cos_dist(feats, a[0])
            scores = scores.flatten()

            rank_ID = np.argsort(scores)
            rank_score = scores[rank_ID]
            num_output = 20
            # print rank_ID[:num_output]
            print rank_score[:num_output]

            imlist = [imgNames[index] for j, index in enumerate(rank_ID[0:num_output])]
            # threshold = 0.8
            # imlist = []
            # for j in range(len(rank_score)):
            #     if rank_score[j] < (1.0 - threshold):
            #         imlist.append(imgNames[rank_ID[j]])
            #     else:
            #         break
            # print imlist
            result_rank[img_path] = imlist
            print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list))
        with open(file_result, 'w') as f:
            for key, value in result_rank.items():
                f.write(key)
                for k in range(len(value)):
                    f.write(' ' + str(k) + ' ' + value[k])
                f.write('\n')


def result_generator_densenet(dir_query, h5_file, file_result):
    '''
    this function runs too slow,please use result_generator_cnn_new function !
    :param dir_query:
    :param h5_file:
    :param file_result:
    :return:
    '''
    # read in indexed images' feature vectors and corresponding image names
    h5f = h5py.File(h5_file, 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    print "--------------------------------------------------"
    print "               searching starts"
    print "--------------------------------------------------"

    # init VGGNet16 model
    model = DenseNETMAX()

    # read and show query image
    img_list = utils.get_all_files_suffix(dir_query)
    # extract query image's feature, compute simlarity score and sort
    # here is baoli search
    result_rank = {}
    for i, img_path in enumerate(img_list):
        queryVec = model.extract_feat(img_path)

        a = []
        a.append(queryVec)
        scores = cos_dist(feats, a[0])
        scores = scores.flatten()

        rank_ID = np.argsort(scores)
        rank_score = scores[rank_ID]
        num_output = 20
        # print rank_ID[:num_output]
        print rank_score[:num_output]

        imlist = [imgNames[index] for j, index in enumerate(rank_ID[0:num_output])]
        # threshold = 0.8
        # imlist = []
        # for j in range(len(rank_score)):
        #     if rank_score[j] < (1.0 - threshold):
        #         imlist.append(imgNames[rank_ID[j]])
        #     else:
        #         break
        # print imlist
        result_rank[img_path] = imlist

        print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list))

    with open(file_result, 'w') as f:
        for key, value in result_rank.items():
            f.write(key)
            for k in range(len(value)):
                f.write(' ' + str(k) + ' ' + value[k])
            f.write('\n')


def result_generator_cnn_new(h5_file_query, h5_file, file_result):
    h5f = h5py.File(h5_file, 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()
    print "--------------------------------------------------"
    print "               searching starts"
    print "--------------------------------------------------"

    h5f_query = h5py.File(h5_file_query, 'r')
    feats_query = h5f_query['dataset_1'][:]
    names_query = h5f_query['dataset_2'][:]
    h5f_query.close()

    # img_list = zip(feats_query, names_query)
    # here is baoli search
    result_rank = {}
    start_time_all = time.time()

    d = feats.shape[1]  # dimension
    # index = faiss.IndexFlatL2(d)  # build the index
    index = faiss.IndexFlatIP(d)  # build the index
    print(index.is_trained)
    feats = feats.astype('float32')
    feats_query = np.array(feats_query[:]).astype('float32')
    index.add(feats)  # add vectors to the index
    print(index.ntotal)

    k = 400  # we want to see 4 nearest neighbors
    # D, I = index.search(xb[:5], k)  # sanity check
    # print(I) #相似位置
    # print(D)# 相似度
    D, I = index.search(feats_query, k)  # actual search
    # print(I[:5])  # neighbors of the 5 first queries
    # print(I[-5:])

    [rows, cols] = I.shape
    for i_row in range(rows):
        rank_ID = I[i_row, :]
        # num_output = 20
        #
        # # print rank_ID[:num_output]
        #
        # imlist = [imgNames[index] for j_rank, index in enumerate(rank_ID[0:num_output])]
        # result_rank[names_query[i_row]] = imlist

        threshold = 0.8
        imlist = []
        rank_score = D[i_row, :]
        for j in range(len(rank_score)):
            if rank_score[j] >= threshold:
                imlist.append(imgNames[rank_ID[j]])
            else:
                break
        # print imlist
        result_rank[names_query[i_row]] = imlist

    end_time_all = time.time()
    print ("total extract time:", (end_time_all - start_time_all))

    with open(file_result, 'w') as f:
        for key, value in result_rank.items():
            f.write(key.replace('/opt/dongsl', '/data/datasets'))
            for k in range(len(value)):
                f.write(' ' + str(k) + ' ' + value[k].replace('/opt/dongsl', '/data/datasets'))
            f.write('\n')


def result_generator_sift_color(file_bow, file_img_names, h5_file, file_result):
    '''
    需要使用paiss库，mac:anaconda有这个库
    :param file_bow:
    :param file_img_names:
    :param h5_file:
    :param file_result:
    :return:
    '''
    # read in indexed images' feature vectors and corresponding image names
    h5f = h5py.File(h5_file, 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()
    print "--------------------------------------------------"
    print "               searching starts"
    print "--------------------------------------------------"

    feats_query = []
    names_query = []
    with open(file_bow) as f:
        hang_a = f.readlines()
        for item_hang in hang_a:
            a = np.array(item_hang.split(' ')[:-1])
            feats_query.append(a)
    with open(file_img_names) as f:
        a = f.readlines()[0].split('.jpg')
        for i in range(len(a)):
            if i != len(a) - 1:

                if '_zoom_20+' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/zoom_up') + '.jpg')
                elif '_zoom_20-' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/zoom_down') + '.jpg')
                elif '_crop_1' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/crop') + '.jpg')
                elif '_con_25-' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/contrast_down') + '.jpg')
                elif '_con_25+' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/contrast_up') + '.jpg')
                elif '_rot_45+' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/rot_45') + '.jpg')
                elif '_blur' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/blur') + '.jpg')
                elif '_shift' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/shift') + '.jpg')
                elif '_letterbox' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/letterbox') + '.jpg')
                elif '_text' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/add_text') + '.jpg')
                elif '_flip_ho' in a[i]:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/flip') + '.jpg')
                else:
                    names_query.append(a[i].replace('/opt/dongsl/query_imgs/trans_imgs_query',
                                                    '/opt/dongsl/trans_imgs/origin') + '.jpg')

    # img_list = zip(feats_query, names_query)
    # here is baoli search
    result_rank = {}
    start_time_all = time.time()

    d = feats.shape[1]  # dimension
    # index = faiss.IndexFlatL2(d)  # build the index
    index = faiss.IndexFlatIP(d)  # build the index
    print(index.is_trained)
    feats = feats.astype('float32')
    feats_query = np.array(feats_query[:]).astype('float32')
    index.add(feats)  # add vectors to the index
    print(index.ntotal)

    k = 40  # we want to see 4 nearest neighbors
    # D, I = index.search(xb[:5], k)  # sanity check
    # print(I) #相似位置
    # print(D)# 相似度
    D, I = index.search(feats_query, k)  # actual search
    # print(I[:5])  # neighbors of the 5 first queries
    # print(I[-5:])

    [rows, cols] = I.shape
    for i_row in range(rows):
        rank_ID = I[i_row, :]
        num_output = 20

        # print rank_ID[:num_output]

        imlist = [imgNames[index] for j_rank, index in enumerate(rank_ID[0:num_output])]
        result_rank[names_query[i_row]] = imlist

    end_time_all = time.time()
    print ("total extract time:", (end_time_all - start_time_all))
    # threshold = 0.8
    # imlist = []
    # for j in range(len(rank_score)):
    #     if rank_score[j] < (1.0 - threshold):
    #         imlist.append(imgNames[rank_ID[j]])
    #     else:
    #         break
    # print imlist
    # result_rank[img_path] = imlist

    with open(file_result, 'w') as f:
        for key, value in result_rank.items():
            f.write(key.replace('/opt/dongsl', '/data/datasets'))
            for k in range(len(value)):
                f.write(' ' + str(k) + ' ' + value[k].replace('/opt/dongsl', '/data/datasets'))
            f.write('\n')


if __name__ == '__main__':

    # result_generator_sift_color('./features/feature_query_sift/trans_imgs_query.bow', './features/feature_query_sift/trans_imgs_query_names.txt',
    #                             './features/feature_sift_color_trans_imgs_ccweb.h5',
    #                             './results/result_sift_color_trans_imgs_ccweb.dat')
    valid_small = True
    if valid_small:
        file_img = '/shihuijie/project/vae/data/image_list_valid_small.txt'
        # h5_file_query = './features/feature_vae_resnetv2_101_query_small_136000.h5'
        h5_file_query = './features/feature_sift_color_query_ccweb_v2.h5'
        file_result = './results/result_sift_color_query_ccweb_v2.dat'
    else:
        file_img = '/shihuijie/project/vae/data/image_list_valid.txt'
        h5_file_query = './features/feature_vae_resnetv2_101_query_136000.h5'
        file_result = './results/result_resnetv2_101_136000.dat'

    file_ckpt = '/shihuijie/project/densenet/model_new/model_vae_resnetv2_101/vae-136000'
    file_meta_graph = file_ckpt + '.meta'
    # h5_file_db = './features/feature_vae_resnetv2_101_trans_imgs_136000.h5'
    h5_file_db = './features/feature_sift_color_query_ccweb_v2.h5'

    # index.feature_generator_basenet(file_img=file_img,
    #                       checkpoints_dir='/shihuijie/project/vae/checkpoints/resnet_v2_101/',file_feature_output=h5_file_query)
    # index.feature_generator_vae(file_img, file_meta_graph, file_ckpt, h5_file_query)
    result_generator_cnn_new(h5_file_query, h5_file_db, file_result)
