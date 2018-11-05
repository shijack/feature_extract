# coding=utf-8
import time

import h5py
import numpy as np

from util import utils

'''
将线上系统生成的所有视频帧特征转为h5格式，在本系统中只需要对视频库进行处理，不需要对查询视频特征进行转换，
只需要在result_generator.py脚本中直接产生结果。
'''


def bow2h5f(dir_name, file_feature_output):
    bows = utils.get_all_files_suffix(dir_name, '.bow')
    img_names = utils.get_all_files_suffix(dir_name, '.txt')
    print "--------------------------------------------------"
    print "         feature extraction starts"
    print "--------------------------------------------------"
    feats = []
    names = []
    start_time = time.time()
    for i, bow_path in enumerate(bows):
        if bow_path[:-4] + '_img_names.txt' in set(img_names):
            with open(bow_path[:-4] + '_img_names.txt') as f:
                a = f.readlines()[0].split('.jpg')
                for j in range(len(a)):
                    if j != len(a) - 1:
                        names.append(a[j] + '.jpg')
        else:
            print 'can not find the img_names .txt'
            return -1
        with open(bow_path) as f:
            hang_a = f.readlines()
            for item_hang in hang_a:
                a = np.array(item_hang.split(' ')[:-1])
                feats.append(a)
        print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(bows))

    end_time = time.time()
    print ("final_feature extract time:", (end_time - start_time))
    feats = np.array(feats)

    print "--------------------------------------------------"
    print "      writing feature extraction results ..."
    print "--------------------------------------------------"
    # directory for storing extracted features
    output = file_feature_output

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=names)
    h5f.close()


def feature_generator_sift_color(dir_bow, file_feature_output):
    bow2h5f(dir_bow, file_feature_output)


if __name__ == '__main__':
    dir_bow = '/opt/dongsl/tmp2/tmp/'
    file_feature_output = '/opt/dongsl/tmp2/tmp/feature_sift_color_query_ccweb_v2.h5'

    feature_generator_sift_color(dir_bow=dir_bow, file_feature_output=file_feature_output)
