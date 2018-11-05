# -*- coding: utf-8 -*-
# Author: shijack
import sys
import time

sys.path.append('../../')

import os

from nets import resnet_v2
from net_model.extract_cnn_vgg16 import VGG16_MODIFIED

import h5py
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from util import utils
from net_model.extract_cnn_densenet_keras import DenseNETMAX
from z_extend_rmac.rmac import rmac
from z_extend_rmac.get_regions import rmac_regions, get_size_vgg_feat_map


def feature_generator_densenet(file_img, file_feature_output):
    tmp_img_list = []
    img_list = []
    with open(file_img, 'r') as f:
        tmp_img_list = f.readlines()

    for item_img in tmp_img_list:
        img_list.append(item_img.split(' ')[0])
    print "--------------------------------------------------"
    print "         feature extraction starts"
    print "--------------------------------------------------"
    feats = []
    names = []
    start_time = time.time()
    model = DenseNETMAX()
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        # dct_feat = np.multiply(np.array(DCT_binaray(img_path)),np.full((1,256),0.01))
        # dct_feat = get_dct_feature(img_path)
        # dct_feat = DCT_binaray(img_path)
        # final_feat = np.append(dct_feat,norm_feat)
        img_name = img_path
        # norm_feat = np.hstack((norm_feat,np.zeros([32,],dtype=np.float32)))
        feats.append(norm_feat)
        names.append(img_name)
        print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list))
    end_time = time.time()
    print ("final_feature extract time:", (end_time - start_time))
    feats = np.array(feats)
    # directory for storing extracted features
    output = file_feature_output
    print "--------------------------------------------------"
    print "      writing feature extraction results ..."
    print "--------------------------------------------------"
    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=names)
    h5f.close()


def feature_generator_rmac_vgg16(dir_img, file_feature_output, is_split_dir=False):
    '''
    按照文件夹目录，每个目录生成一个文件夹所有图片特征的集合.bow文件，format：每行一个图片的特征。
    :param dir_img:
    :param file_feature_output:
    :param is_split_dir:
    :return:
    '''
    path = dir_img
    print "--------------------------------------------------"
    print "         feature extraction starts"
    print "--------------------------------------------------"

    if is_split_dir:
        model = rmac.rmac(20)
        for child_dirs in utils.get_dirs_child(path):
            img_list = utils.get_all_files_suffix(child_dirs)
            start_time = time.time()
            feats = []
            names = []
            for i, img_path in enumerate(img_list):
                img = image.load_img(img_path)

                # Resize
                scale = utils.IMG_SIZE / max(img.size)
                new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
                # print('Original size: %s, Resized image: %s' % (str(img.size), str(new_size)))
                img = img.resize(new_size)

                # Mean substraction
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_image(x)

                # Load RMAC model
                Wmap, Hmap = get_size_vgg_feat_map(x.shape[2], x.shape[1])
                regions = rmac_regions(Wmap, Hmap, 3)

                # Compute RMAC vector
                # print('Extracting RMAC from image...')
                # print (len(regions))
                norm_feat = model.predict([x, np.expand_dims(regions, axis=0)])

                norm_feat = norm_feat.reshape((-1,))
                img_name = os.path.split(img_path)[1]
                final_feat = np.hstack((norm_feat.reshape((-1,)), np.zeros([288, ], dtype=np.float32)))
                feats.append(final_feat)
                names.append(img_name)
                print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list))
            feats = np.array(feats)

            print "--------------------------------------------------"
            print "      writing feature extraction results ..."
            print "--------------------------------------------------"
            feats_6 = feats.astype('float32')
            np.savetxt(child_dirs + "/" + child_dirs.split("/")[-1] + '.bow', feats_6, fmt='%f')
        end_time = time.time()
        print ('the total time cnsumed is %d\n', (end_time - start_time))
    else:
        feats = []
        names = []
        start_time = time.time()
        model = rmac.rmac(20)
        img_list = utils.get_all_files_suffix(dir_img)
        for i, img_path in enumerate(img_list):
            img = image.load_img(img_path)

            # Resize
            scale = utils.IMG_SIZE / max(img.size)
            new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
            # print('Original size: %s, Resized image: %s' % (str(img.size), str(new_size)))
            img = img.resize(new_size)

            # Mean substraction
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_image(x)

            # Load RMAC model
            Wmap, Hmap = get_size_vgg_feat_map(x.shape[2], x.shape[1])
            regions = rmac_regions(Wmap, Hmap, 3)

            # Compute RMAC vector
            # print('Extracting RMAC from image...')
            # print (len(regions))
            norm_feat = model.predict([x, np.expand_dims(regions, axis=0)])

            norm_feat = norm_feat.reshape((-1,))
            img_name = os.path.split(img_path)[1]
            final_feat = np.hstack((norm_feat.reshape((-1,)), np.zeros([288, ], dtype=np.float32)))
            feats.append(final_feat)
            names.append(img_name)
            print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list))
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


def feature_generator_vae(file_img, file_meta_graph, file_ckpt, file_feature_output):
    print os.path.abspath(file_meta_graph)
    print file_ckpt
    tmp_img_list = []
    img_list = []
    with open(file_img, 'r') as f:
        tmp_img_list = f.readlines()

    for item_img in tmp_img_list:
        img_list.append(item_img.split(' ')[0])

    print "--------------------------------------------------"
    print "         feature extraction starts"
    print "--------------------------------------------------"
    feats = []
    names = []
    start_time = time.time()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(file_meta_graph)
        saver.restore(sess, file_ckpt)
        graph = tf.get_default_graph()
        x_input = graph.get_tensor_by_name('encoder/input_img:0')
        latent_feature = graph.get_tensor_by_name('variance/latent_feature:0')

        for i, img_path in enumerate(img_list):
            img = utils.img_process(img_path)
            norm_feat = sess.run(latent_feature, feed_dict={x_input: img})
            img_name = img_path
            # norm_feat = np.hstack((norm_feat,np.zeros([160,],dtype=np.float32)))
            feats.append(norm_feat.flatten())
            names.append(img_name)
            print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list))
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


def feature_generator_basenet(file_img, checkpoints_dir, file_feature_output):
    tmp_img_list = []
    img_list = []
    with open(file_img, 'r') as f:
        tmp_img_list = f.readlines()

    for item_img in tmp_img_list:
        img_list.append(item_img.split(' ')[0])

    print "--------------------------------------------------"
    print "         feature extraction starts"
    print "--------------------------------------------------"
    feats = []
    names = []
    from tensorflow.contrib import slim

    x_input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_img')

    # latent_mean, latent_stddev = encoder(x_input, train_logical=True, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_vgg16(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_vgg19(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_inceptionv1(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_inceptionv4(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_inception_resnetv2(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_resnetv2_152(x_input, latent_dim=LATENT_DIM)#参数过多，训练很慢

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_101(x_input, num_classes=None, is_training=False)
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'resnet_v2_101.ckpt'),
        slim.get_model_variables('resnet_v2_101'))
    start_time = time.time()
    with tf.Session() as sess:
        init_fn(sess)
        latent_feature = logits

        for i, img_path in enumerate(img_list):
            img = utils.img_process_vgg_tf(img_path)
            norm_feat = sess.run(latent_feature, feed_dict={x_input: img})
            img_name = img_path
            # norm_feat = np.hstack((norm_feat,np.zeros([160,],dtype=np.float32)))
            feats.append(norm_feat.flatten())
            names.append(img_name)
            print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list))
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


def feature_generator_basenet_vgg(file_img, file_feature_output):
    tmp_img_list = []
    img_list = []
    with open(file_img, 'r') as f:
        tmp_img_list = f.readlines()

    for item_img in tmp_img_list:
        img_list.append(item_img.split(' ')[0])

    print "--------------------------------------------------"
    print "         feature extraction starts"
    print "--------------------------------------------------"
    feats = []
    names = []

    # model = DenseNETMAX()
    model = VGG16_MODIFIED()

    start_time = time.time()

    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        # dct_feat = DCT_binaray(img_path)
        # final_feat = np.append(dct_feat, norm_feat)
        img_name = img_path
        # norm_feat = np.hstack((norm_feat,np.zeros([160,],dtype=np.float32)))
        feats.append(norm_feat.flatten())
        names.append(img_name)
        print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list))

    end_time = time.time()
    print ("final_feature extract time:", (end_time - start_time))
    feats = np.array(feats)

    print "--------------------------------------------------"
    print "      writing feature extraction results ..."
    print "--------------------------------------------------"
    # directory for storing extracted features

    h5f = h5py.File(file_feature_output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=names)
    h5f.close()


if __name__ == "__main__":
    args = {'index_basenet': './result_generator/features/feature_densenet169_trans_imgs_basenet.h5',
            'index': '../features/feature_vae_resnetv2_101_trans_imgs_136000_basenet.h5',
            'database': '/data/datasets/trans_imgs'}

    # feature_generator_densenet(dir_img=args["database"], file_feature_output=args["index"])
    # feature_generator_rmac_vgg16(dir_img=args["database"], file_feature_output=args["index"])
    # file_ckpt = '/shihuijie/project/densenet/model_new/model_vae_resnetv2_101/vae-136000'
    # feature_generator_vae(file_img='/shihuijie/project/vae/data/image_list.txt',
    #                       file_meta_graph=file_ckpt + '.meta',
    #                       file_ckpt=file_ckpt,
    #                       file_feature_output=args["index"])
    feature_generator_basenet(file_img='/shihuijie/project/vae/data/image_list.txt',
                              checkpoints_dir='/shihuijie/project/vae/checkpoints/resnet_v2_101/',
                              file_feature_output=args["index_basenet"])
    # feature_generator_basenet_vgg(file_img='/shihuijie/project/vae/data/image_list.txt',
    #                           file_feature_output=args["index_basenet"])
    # feature_generator_densenet(file_img='/shihuijie/project/vae-system/data/image_list.txt',
    #                           file_feature_output=args["index_basenet"])
