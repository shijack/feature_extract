# -*- coding: utf-8 -*-
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from numpy import linalg as LA


class VGG16_MODIFIED:
    def __init__(self):
        base_model = VGG16(weights='imagenet', include_top=True)

        # make the feature maps to a featrue us globalmaxpooling

        x_1 = base_model.get_layer('fc2').output
        # x_1 = GlobalMaxPooling2D()(x_1)
        self.model = Model(inputs=base_model.input, outputs=x_1)
        print self.model.summary()
        # print base_model.summary()


        self.input_shape = (224, 224, 3)
        self.model.predict(np.zeros((1, 224, 224, 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''

    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)

        norm_feat = (feat / LA.norm(feat)).reshape((-1,))
        return norm_feat
