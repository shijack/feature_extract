# -*- coding: utf-8 -*-
import datetime
import os

import numpy as np

import util.utils as utils
from net_model.extract_cnn_vgg16 import VGG16_MODIFIED

'''

ap = argparse.ArgumentParser()
ap.add_argument("-database", required = True,
	help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Name of index file")
args = vars(ap.parse_args())
'''

args = {'index': 'feature_cnn_v4v2.h5', 'database': 'keyframe/'}  # dct confi sum then absolute

'''
 Extract features and index the images
'''
if __name__ == "__main__":

    db = args["database"]
    path = db
    print "--------------------------------------------------"
    print "         feature extraction starts"
    print "--------------------------------------------------"

    start_time = datetime.datetime.now()
    # model = DenseNETMAX()
    model = VGG16_MODIFIED()
    for dir_imgs in [os.path.join(path, f) for f in os.listdir(path)]:
        img_list = utils.get_imlist(dir_imgs)
        start_time = datetime.datetime.now()
        feats = []
        names = []
        for i, img_path in enumerate(img_list):
            norm_feat = model.extract_feat(img_path)
            # dct_feat = DCT_binaray(img_path)
            # final_feat = np.append(dct_feat, norm_feat)
            img_name = os.path.split(img_path)[1]
            final_feat = np.hstack((norm_feat, np.zeros([288, ], dtype=np.float32)))
            feats.append(final_feat)
            names.append(img_name)
            print "extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list))
        feats = np.array(feats)

        feats_6 = feats.astype('float32')
        np.savetxt(dir_imgs + "/" + dir_imgs.split("/")[-1] + '.bow', feats_6, fmt='%f')
    print "--------------------------------------------------"
    print "      writing feature extraction results ..."
    print "--------------------------------------------------"
    end_time = datetime.datetime.now()
    print ('the total time cnsumed is %d\n', (end_time - start_time))
