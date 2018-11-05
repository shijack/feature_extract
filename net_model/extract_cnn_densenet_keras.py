# -*- coding: utf-8 -*-
# Author: yongyuan.name
import cv2
import numpy as np
from keras.models import Model
from numpy import linalg as LA

from net_model.densenet169 import DenseNet


class DenseNETMAX:
    def __init__(self):
        '''
        确保使用TensorFlow作为back
        '''
        weights_path = './imagenet_models/densenet169_weights_tf.h5'
        base_model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
        # sgd = SGD(lr=1e-2,decay=1e-6,momentum=0.9,nesterov=True)
        # base_model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
        # make the feature maps to a featrue us globalmaxpooling

        x = base_model.get_layer('fc6').output
        self.model = Model(inputs=base_model.input, outputs=[x])
        print self.model.summary()

        self.input_shape = (224, 224, 3)
        self.model.predict(np.zeros((1, 224, 224, 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''

    def extract_feat(self, img_path):
        im = cv2.resize(cv2.imread(img_path), (self.input_shape[0], self.input_shape[1])).astype(np.float32)

        # Subtract mean pixel and multiple by scaling constant
        # Reference: https://github.com/shicai/DenseNet-Caffe
        im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
        im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
        im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017

        img = np.expand_dims(im, axis=0)
        feat = self.model.predict(img)
        norm_feat = (feat / LA.norm(feat)).reshape((-1,))
        return norm_feat
