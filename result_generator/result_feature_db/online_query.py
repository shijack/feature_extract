# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os

import scipy.spatial.distance
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from extract_cnn_densenet_keras import DenseNETMAX

import numpy as np
import h5py
from util import utils

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

'''
matrix:(number,feature)
vector:(1,feature)
'''


def cos_dist(matrix, vector):
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)


'''
ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True,
	help = "Path to query which contains image to be queried")
ap.add_argument("-index", required = True,
	help = "Path to index")
ap.add_argument("-result", required = True,
	help = "Path for output retrieved images")
args = vars(ap.parse_args())
'''

args = {'query': '/data/datasets/trans_imgs/origin/5618_29_RKF.jpg', 'result': '',
        'index': './features/feature_vae_trans_imgs.h5'}

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"], 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

print "--------------------------------------------------"
print "               searching starts"
print "--------------------------------------------------"

# read and show query image
queryDir = args["query"]
# queryImg = mpimg.imread(queryDir)
# plt.title("Query Image")
# plt.imshow(queryImg)
# plt.show()

# init VGGNet16 model
# model = DenseNETMAX()
# model = VGG16_MODIFIED()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model_new/vae-59000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model_new'))
    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name('encoder/input_img:0')
    latent_feature = graph.get_tensor_by_name('variance/latent_feature:0')

    # extract query image's feature, compute simlarity score and sort
    # here is baoli search
    # queryVec = model.extract_feat(queryDir)
    img = utils.img_process(queryDir)
    queryVec = sess.run(latent_feature, feed_dict={x_input: img})

    # dct_feat = np.multiply(np.array(DCT_binaray(queryDir)),np.full((1,256),0.01))
    # dct_feat = get_dct_feature(queryDir)
    # dct_feat = DCT_binaray(queryDir)
    # norm_feat = np.hstack((queryVec,np.zeros([32,],dtype=np.float32)))
    norm_feat = queryVec
# final_feat = np.append(norm_feat,queryVec)
# dist = DistanceMetric.get_metric('euclidean')


a = []
a.append(norm_feat)

# scores = dist.pairwise(feats, a)
scores = cos_dist(feats, a[0])
scores = scores.flatten()
rank_ID = np.argsort(scores)

# scores = np.dot(queryVec, feats.T)
# rank_ID = np.argsort(scores)[::-1]


rank_score = scores[rank_ID]
num_output = 40
# print rank_ID[:num_output]
print rank_score[:num_output]

threshold = 0.8
imlist = []
# # number of top retrieved images to show
# maxres = num_output
# imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
# print "top %d images in order are: " % maxres, imlist

for i in range(len(rank_score)):
    if rank_score[i] < (1.0 - threshold):
        imlist.append(imgNames[rank_ID[i]])
    else:
        break

print imlist
