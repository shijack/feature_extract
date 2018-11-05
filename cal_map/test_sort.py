# coding=utf-8
'''
ATTENTION: this script useless now
@author shijack
2018年09月04日18:32:38
'''

import os

import numpy as np


def get_rannk_list(rootdir):
    file_list = os.listdir(rootdir)  # 列出该文件夹下所有的目录与文件
    for i in range(0, len(file_list)):
        if "_" not in file_list[i]:  # 不处理带_的文件
            path = os.path.join(rootdir, file_list[i])
            if os.path.isfile(path):
                f = open(path, 'r')
                lines = f.readlines()
                a = []
                for j in range(len(lines)):
                    if 'block' in lines[j] and '%' in lines[j]:
                        item_1 = lines[j].split(',')
                        item_name = item_1[0].split(' ')[-1]
                        item_score = item_1[1].split('%')[0].split(' ')[-1]
                        a.append((item_name, item_score))
                b = sorted(a, reverse=True, key=lambda item: float(item[1]))
                list_name = []
                for item_name in b:
                    if item_name[0] not in list_name:
                        list_name.append(item_name[0])
                list_name = list_name

                output_dir = os.path.dirname(rootdir) + '/' + rootdir.split('/')[-1] + '_rank'
                isExists = os.path.exists(output_dir)
                if not isExists:
                    os.makedirs(output_dir)
                with open(output_dir + '/result_rank_' + path.split('/')[-1], 'w') as f1:
                    lists = [str(line) + "\n" for line in list_name]  # 保存所有结果，包括分数
                    f1.writelines(lists)
                with open(output_dir + '/result_score_' + path.split('/')[-1].split('.')[0] + '.txt', 'w') as f2:
                    lists = [str(line) + "\n" for line in b]  # 保存所有结果，包括分数
                    f2.writelines(lists)
                f2.close()
                f1.close()
                f.close()


def calc_ap(prec, rec):
    mrec = np.array([0, rec, 1])
    mpre = np.array([0, prec, 0])

    for i in range(mrec.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1]);
        # print mpre[i]

    # print mrec[1 : ]
    # print mrec[0 : 2]

    idx1 = np.where(mrec[1:] != mrec[0: 2])
    idx2 = [x + 1 for x in idx1]

    # print mrec.take(idx2)
    # print mrec.take(idx1)

    ap = sum((mrec.take(idx2) - mrec.take(idx1)) * mpre.take(idx2))
    # print "ap = " + str(ap)
    return ap


def get_rank_bowmatch(bowmatch_path='/Users/shijack/Desktop/keyframe'):
    bowmatch_path = bowmatch_path
    for root, dirs, files in os.walk(bowmatch_path):
        for dir in dirs:
            if 'bowmatch' in dir:
                path = os.path.join(root, dir)
                print path
                get_rannk_list(path)


def get_gt(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    gt_set = set()
    for item_line in lines:
        line = item_line.split(',')
        gt_set.add(line[0].split('.')[0])
        gt_set.add(line[1].split('.')[0])
    print gt_set
    for i in gt_set:
        with open('gt_vcdb/' + i + '.txt', 'w') as f:
            lists = [str(line) + "\n" for line in gt_set]
            f.writelines(lists)
        f.close()


def get_groundtruth():
    root_anno = '/Users/shijack/Desktop/keyframe/vcdb-annotation/annotation'
    file_list = os.listdir(root_anno)
    for file_item in file_list:
        get_gt(os.path.join(root_anno, file_item))


def get_mAP(result_rank_dir='/Users/shijack/Desktop/keyframe/bowmatch-cnn-conv4/bowmatch-cnn-conv4-096-30_rank'):
    root_gt = 'gt_vcdb'
    file_list = os.listdir(root_gt)
    gt_dict = {}
    for file_item in file_list:
        f = open(os.path.join(root_gt, file_item), 'r')
        lines = f.readlines()
        gt_dict[file_item] = lines
    root_re = result_rank_dir
    # root_re = '/Users/shijack/Desktop/keyframe/bowmatch-online_rank'
    file_list = os.listdir(root_re)
    re_dict = {}
    for file_item in file_list:
        if 'result_rank_' in file_item:
            f = open(os.path.join(root_re, file_item), 'r')
            lines = f.readlines()
            re_dict[file_item.split('_')[-1]] = lines
    list_ap = []
    list_miss = []
    for item_gr in gt_dict:
        if re_dict.get(item_gr):
            t_num = 0.0
            re_list = re_dict[item_gr]
            gt_list = gt_dict[item_gr]
            re_num = len(re_list)
            gr_num = len(gt_list)
            list_su = list(set(re_list).intersection(set(gt_list)))  # 正检数据
            list_error = list(set(re_list).difference(set(gt_list)))  # 误检数据，在re_list中但不在gt_list中
            list_miss = list(set(gt_list).difference(set(re_list)))  # 漏检数据，在gt_list中但不在re_list中
            t_num = len(list_su)
            precision = 1.0 * t_num / re_num
            recall = 1.0 * t_num / gr_num
            # ap = 0.67786704015
            ap = calc_ap(precision, recall)
            list_ap.append(ap)
            with open(result_rank_dir + '/statis_' + item_gr, 'w') as f:
                f.write('#result_miss_detectoin:\n')
                lists = [str(line) + "\n" for line in list_miss]
                f.writelines(lists)
                f.write('#result_error_detectoin:\n')
                lists = [str(line) + "\n" for line in list_error]
                f.writelines(lists)
                f.write('#result_correct_detectoin:\n')
                lists = [str(line) + "\n" for line in list_su]
                f.writelines(lists)
                # print ap
        else:
            # list_ap.append()
            print "nothing in item_gr: %s" % item_gr

    mAP = sum(list_ap) / len(list_ap)
    print mAP[0]
    return mAP[0]
    # print list_ap


def get_map_list(root_dir='/Users/shijack/Desktop/keyframe/bowmatch-cnn-relu3-blk'):
    root_dir = root_dir
    file_list = os.listdir(root_dir)
    gt_dict = {}
    for rank_item in file_list:
        if '_rank' in rank_item:
            gt_dict[rank_item] = get_mAP(os.path.join(root_dir, rank_item))
    print gt_dict


if __name__ == "__main__":
    # args = parse_args()
    # len(sys.argv)
    # print "args.precision: {}".format(args.precision)
    # print "args.recall: {}".format(args.recall)

    # get_groundtruth()
    get_rank_bowmatch('/Users/shijack/Desktop/keyframe/bowmatch-cnn-fu-v4v2')
    # get_rank_bowmatch('/Users/shijack/Desktop/keyframe/bowmatch-online')
    # get_rank_bowmatch('/Users/shijack/Desktop/keyframe/bowmatch-cnn-conv4')
    get_map_list('/Users/shijack/Desktop/keyframe/bowmatch-cnn-fu-v4v2')
    # get_map_list('/Users/shijack/Desktop/keyframe/bowmatch-cnn-fu-v4v2')
    # get_map_list('/Users/shijack/Desktop/keyframe/bowmatch-online')
