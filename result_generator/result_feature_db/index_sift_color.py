# coding=utf-8
import os
import shutil
import time


def get_dirs_child(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def get_all_files_suffix(path, file_suffix='.jpg'):
    all_file = []
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            if name.endswith(file_suffix):
                all_file.append(os.path.join(dirpath, name))
    return all_file


def copyFiles(file_imgs, targetDir):
    list_imgs = []
    with open(file_imgs, 'r') as f:
        list_imgs_tmp = f.readlines()
    for item_img in list_imgs_tmp:
        list_imgs.append(
            item_img.split(' ')[0].replace('/opt/Datasets/Datasets/ccweb_video/dataset_ccweb/trans_imgs',
                                           '/Data/Datasets/ccweb_video/dataset_ccweb/trans_imgs').strip())
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    for eachfile in list_imgs:
        if not os.path.exists(eachfile):
            print "src path not exist:" + eachfile
            print "error!! attation!"
            return -1
        shutil.copy(eachfile, targetDir + os.path.basename(eachfile))
        print eachfile + " copy succeeded!"


cmd = '/usr/local/bin/videofpget_bow_hash /opt/dongsl/keyframe/10732a0e6a0edef9dcbb2155236e46a7ed5047c0/ 1 4 /retrieval/VideoDNA/VideoRetrival/bins/centers128_32sift.bin /retrieval/VideoDNA/VideoRetrival/bins/ITQ_32_dim800.bin /opt/a.bow /opt/dongsl/a.hash'

'/usr/local/bin/videofpget_bow_hash /opt/dongsl/trans_imgs/add_text 1 26069 /retrieval/VideoDNA/VideoRetrival/bins/centers128_32sift.bin /retrieval/VideoDNA/VideoRetrival/bins/ITQ_32_dim800.bin /opt/dongsl/t.bow /opt/dongsl/t.hash'


def feature_generator_sift_color(dir_img):
    dir_child_list = get_dirs_child(dir_img)
    print "--------------------------------------------------"
    print "         feature extraction starts"
    print "--------------------------------------------------"

    start_time = time.time()

    for i, img_path in enumerate(dir_child_list):
        names = []
        img_names = get_all_files_suffix(img_path)
        for j, item_name in enumerate(img_names):
            names.append(item_name)
            newname = os.path.dirname(item_name) + '/%05d' % (j + 1)
            os.rename(item_name, newname + ".jpg")
        img_names = get_all_files_suffix(img_path)
        print len(img_names)
        fp_pick = '/usr/local/bin/videofpget_bow_hash ' + img_path + '/ 1 ' + str(len(
            img_names)) + ' /retrieval/VideoDNA/VideoRetrival/bins/centers128_32sift.bin /retrieval/VideoDNA/VideoRetrival/bins/ITQ_32_dim800.bin ' + os.path.dirname(
            img_path) + '/' + img_path.split('/')[-1] + '.bow ' + os.path.dirname(img_path) + '/' + img_path.split('/')[
                      -1] + '.hash'

        os.system(fp_pick)

        with open(os.path.dirname(img_path) + '/' + img_path.split('/')[-1] + '_img_names.txt', 'w') as name_file:
            name_file.writelines(names)
        print "extracting feature from image No. %d , %d dirs in total" % ((i + 1), len(dir_child_list))
    end_time = time.time()
    print ("final_feature extract time:", (end_time - start_time))

    print "--------------------------------------------------"
    print "      feature extraction ends ..."
    print "--------------------------------------------------"


def feature_generator_query(target_dir):
    '''
    根据图片文件列表，获取线上系统 查询视频帧 .bow .hash .txt 信息
    :param target_dir: endswith /
    :return:
    '''
    copyFiles('./test_2000.txt', target_dir)
    feature_generator_sift_color(dir_img=os.path.abspath(os.path.join(os.path.dirname(target_dir), '../')))


if __name__ == "__main__":
    query_dir_imgs = '/opt/dongsl/tmp2/tmp/'
    feature_generator_query(query_dir_imgs)
