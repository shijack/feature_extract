import os
import pickle
import platform

import cv2
import matplotlib
import numpy as np

if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing import image

DATA_DIR = 'data/'
WEIGHTS_FILE = 'vgg16_weights_th_dim_ordering_th_kernels.h5'
PCA_FILE = 'PCAmatrices.mat'
IMG_SIZE = 1024


def save_obj(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    print "Object saved to %s." % filename


def load_obj(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    print "Object loaded from %s." % filename
    return obj


def preprocess_image(x):
    # Substract Mean
    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68

    # 'RGB'->'BGR'
    x = x[:, ::-1, :, :]

    return x


def get_dirs_child(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def get_all_files_suffix(path, file_suffix='.jpg'):
    all_file = []
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            if name.endswith(file_suffix):
                all_file.append(os.path.join(dirpath, name))
    return all_file


def img_process_vgg_tf(img_path):
    img = image.load_img(img_path)

    img = img.resize((224, 224))

    # Mean substraction
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68

    return x


def img_process(img_path, depth=3):
    img = cv2.imread(img_path)
    if depth == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (112, 144)) / 255.0
    img = cv2.resize(img, (224, 224))
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b]) / 255.0
    train_total_data = np.expand_dims(rgb_img, axis=0)
    return train_total_data


def display_imgs(imlist):
    if imlist is None:
        imlist = ['/Users/shijack/Desktop/trans_imgs/trans_imgs/origin/5228_13_RKF.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/origin/6087_13_RKF.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_up/5228_13_RKF_zoom_20+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_down/5228_13_RKF_con_25-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_up/6087_13_RKF_zoom_20+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_down/6087_13_RKF_con_25-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/add_text/5228_13_RKF_text.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_up/5228_13_RKF_con_25+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_up/6087_13_RKF_con_25+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/add_text/6087_13_RKF_text.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/blur/5228_13_RKF_blur.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/blur/6087_13_RKF_blur.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_down/6087_13_RKF_zoom_20-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_down/5228_13_RKF_zoom_20-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/add_text/5475_22_RKF_text.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/add_text/5280_18_RKF_text.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/add_text/5658_18_RKF_text.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/add_text/5532_16_RKF_text.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_up/5475_22_RKF_zoom_20+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_up/5658_18_RKF_zoom_20+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/origin/5475_22_RKF.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_down/5658_18_RKF_con_25-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/add_text/5634_20_RKF_text.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/origin/5658_18_RKF.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/add_text/5297_19_RKF_text.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_down/5475_22_RKF_con_25-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/origin/5280_18_RKF.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_up/5280_18_RKF_con_25+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_up/5280_18_RKF_zoom_20+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_up/5658_18_RKF_con_25+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/origin/5532_16_RKF.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_down/5634_20_RKF_con_25-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_up/5532_16_RKF_con_25+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_up/5475_22_RKF_con_25+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_down/5280_18_RKF_con_25-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_down/5532_16_RKF_con_25-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_up/5532_16_RKF_zoom_20+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_up/5634_20_RKF_zoom_20+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/origin/5634_20_RKF.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/blur/5658_18_RKF_blur.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/blur/5475_22_RKF_blur.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/blur/5280_18_RKF_blur.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/blur/5532_16_RKF_blur.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_up/5634_20_RKF_con_25+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/zoom_up/5297_19_RKF_zoom_20+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_down/5297_19_RKF_con_25-.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/origin/5297_19_RKF.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/contrast_up/5297_19_RKF_con_25+.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/blur/5297_19_RKF_blur.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/blur/5634_20_RKF_blur.jpg',
                  '/Users/shijack/Desktop/trans_imgs/trans_imgs/crop/5422_19_RKF_crop_1.jpg']

    for i, im in enumerate(imlist):
        print im
        image = mpimg.imread(im)
        plt.title("search output %d" % (i + 1))
        plt.imshow(image)
        plt.show()


def generate_ground_truth():
    pass


if __name__ == "__main__":
    display_imgs(None)
