# coding=utf-8
import sys

sys.path.append("..")
from util import utils


def get_gt_1():
    img_list = utils.get_all_files_suffix('/data/datasets/trans_imgs/origin/')
    with open('ground_truth_trans_imgs.dat', 'w') as f:
        for value in img_list:
            pre_value = str('/'.join(value.split('/')[0:-2])) + '/'
            post_value = str(value.split('/')[-1]).split('.')[0]
            f.write(value)
            f.write(' ' + value)
            f.write(
                ' ' + pre_value + 'add_text/' + post_value + '_text.jpg' + ' ' + pre_value + 'blur/' + post_value + '_blur.jpg')
            f.write(
                ' ' + pre_value + 'contrast_down/' + post_value + '_con_25-.jpg' + ' ' + pre_value + 'contrast_up/' + post_value + '_con_25+.jpg')
            f.write(
                ' ' + pre_value + 'crop/' + post_value + '_crop_1.jpg' + ' ' + pre_value + 'flip/' + post_value + '_flip_ho.jpg')
            f.write(
                ' ' + pre_value + 'letterbox/' + post_value + '_letterbox.jpg' + ' ' + pre_value + 'rot_45/' + post_value + '_rot_45+.jpg')
            f.write(
                ' ' + pre_value + 'shift/' + post_value + '_shift.jpg' + ' ' + pre_value + 'zoom_down/' + post_value + '_zoom_20-.jpg')
            f.write(' ' + pre_value + 'zoom_up/' + post_value + '_zoom_20+.jpg' + '\n')
    print 'write done!'


import os


def get_gt_2(file_imgs_query, dir_output):
    list_imgs = []
    with open(file_imgs_query, 'r') as f:
        list_imgs = f.readlines()
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    with open(dir_output + '/ground_truth_trans_imgs_v2.dat', 'w') as f:
        for item_value in list_imgs:
            item_value = item_value.strip()
            if '/origin/' in item_value.split(' ')[0]:
                value = item_value.split(' ')[0]
                pre_value = str('/'.join(value.split('/')[0:-2])) + '/'
                post_value = str(value.split('/')[-1]).split('.')[0]
                f.write(value)
                f.write(' ' + value)
                f.write(
                    ' ' + pre_value + 'add_text/' + post_value + '_text.jpg' + ' ' + pre_value + 'blur/' + post_value + '_blur.jpg')
                f.write(
                    ' ' + pre_value + 'contrast_down/' + post_value + '_con_25-.jpg' + ' ' + pre_value + 'contrast_up/' + post_value + '_con_25+.jpg')
                f.write(
                    ' ' + pre_value + 'crop/' + post_value + '_crop_1.jpg' + ' ' + pre_value + 'flip/' + post_value + '_flip_ho.jpg')
                f.write(
                    ' ' + pre_value + 'letterbox/' + post_value + '_letterbox.jpg' + ' ' + pre_value + 'rot_45/' + post_value + '_rot_45+.jpg')
                f.write(
                    ' ' + pre_value + 'shift/' + post_value + '_shift.jpg' + ' ' + pre_value + 'zoom_down/' + post_value + '_zoom_20-.jpg')
                f.write(' ' + pre_value + 'zoom_up/' + post_value + '_zoom_20+.jpg' + '\n')
            else:
                value = item_value.split(' ')[1]
                value_query = item_value.split(' ')[0]
                f.write(value_query)
                f.write(' ' + value_query)
                pre_value = str('/'.join(value.split('/')[0:-2])) + '/'
                post_value = str(value.split('/')[-1]).split('.')[0]

                list_jpgs = []
                jpg_add_text = ' ' + pre_value + 'add_text/' + post_value + '_text.jpg'
                jpg_blur = ' ' + pre_value + 'blur/' + post_value + '_blur.jpg'
                jpg_con_d = ' ' + pre_value + 'contrast_down/' + post_value + '_con_25-.jpg'
                jpg_con_u = ' ' + pre_value + 'contrast_up/' + post_value + '_con_25+.jpg'
                jpg_crop = ' ' + pre_value + 'crop/' + post_value + '_crop_1.jpg'
                jpg_flip_h = ' ' + pre_value + 'flip/' + post_value + '_flip_ho.jpg'
                jpg_letter = ' ' + pre_value + 'letterbox/' + post_value + '_letterbox.jpg'
                jpg_rot_45 = ' ' + pre_value + 'rot_45/' + post_value + '_rot_45+.jpg'
                jpg_shift = ' ' + pre_value + 'shift/' + post_value + '_shift.jpg'
                jpg_zoom_d = ' ' + pre_value + 'zoom_down/' + post_value + '_zoom_20-.jpg'
                jpg_zoom_u = ' ' + pre_value + 'zoom_up/' + post_value + '_zoom_20+.jpg'

                list_jpgs.append(' ' + value)  # this is the origin
                list_jpgs.append(jpg_add_text)
                list_jpgs.append(jpg_blur)
                list_jpgs.append(jpg_con_d)
                list_jpgs.append(jpg_con_u)
                list_jpgs.append(jpg_crop)
                list_jpgs.append(jpg_flip_h)
                list_jpgs.append(jpg_letter)
                list_jpgs.append(jpg_rot_45)
                list_jpgs.append(jpg_shift)
                list_jpgs.append(jpg_zoom_d)
                list_jpgs.append(jpg_zoom_u)

                for item_jpg in list_jpgs:
                    if value_query in item_jpg:
                        continue
                    else:
                        f.write(item_jpg)
                f.write('\n')
    print 'write done!'


if __name__ == '__main__':
    # get_gt_1()
    file_imgs_query = '/Users/shijack/Desktop/copy_dir/test_2000.txt'
    dir_output = './ground_truth/'
    get_gt_2(file_imgs_query, dir_output)
