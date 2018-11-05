# coding=utf-8
import os
import sys


def usage():
    print >> sys.stderr, """usage: python holidays_map.py resultfile.dat

Where resultfile.dat is a textfile. Its format is:

result_file = ( result_line newline )*

# each line is a query image with associated results
result_line = query_image_name query_result*

# a query result is a pair: the result's filename is prefixed with its rank (0 based)
query_result = rank result_image_name 

Where:
- all items are separated by whitespaces (space or tab)
- image names are like 12345.jpg (case sensitive)
- the order of queries is not relevant
- if the query image is ranked, it is ignored in the scoring

Copyright INRIA 2008. License: GPL
"""
    sys.exit(1)


def score_ap_from_ranks_1(ranks, nres):
    """ Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset
    """

    # accumulate trapezoids in PR-plot
    ap = 0.0

    # All have an x-size of:
    recall_step = 1.0 / nres

    for ntp, rank in enumerate(ranks):

        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1 = (ntp + 1) / float(rank + 1)

        ap += (precision_1 + precision_0) * recall_step / 2.0

    return ap


def get_groundtruth_trans_imgs(file_result):
    """ Read datafile holidays_images.dat and output a dictionary
    mapping queries to the set of positive results (plus a list of all
    images)"""
    gt = {}
    allnames = set()
    for line in open(file_result, "r"):
        imname = line.split(' ')
        gt_results = set()
        gt[imname[0]] = gt_results
        for i in range(len(imname)):
            allnames.add(imname[i].strip())
            if i != 0:
                gt_results.add(imname[i].strip())

    return (allnames, gt)


def get_result_trans_imgs(file_result):
    """ Read datafile holidays_images.dat and output a dictionary
    mapping queries to the set of positive results (plus a list of all
    images)"""
    gt = {}
    allnames = set()
    for line in open(file_result, "r"):
        fields = line.split()
        imname = fields[0]
        gt_results = set(fields[2::2])
        gt[imname] = gt_results
        allnames.add(imname)

    return (allnames, gt)


def statis_result_trans_imgs(file_result):
    """ 去除结果中的rot_45和shift部分"""
    gt = {}
    allnames = set()
    for line in open(file_result, "r"):
        fields = line.split()
        imname = fields[0]
        gt_results = list(fields[2::2])
        gt_results_left = []
        for item_gt_result in gt_results:
            if not ('shift' in item_gt_result or 'letterbox' in item_gt_result):
                gt_results_left.append(item_gt_result)
        gt[imname] = gt_results_left
        allnames.add(imname)

    return (allnames, gt)


def get_groundtruth():
    """ Read datafile holidays_images.dat and output a dictionary
    mapping queries to the set of positive results (plus a list of all
    images)"""
    gt = {}
    allnames = set()
    for line in open("holidays_images.dat", "r"):
        imname = line.strip()
        allnames.add(imname)
        imno = int(imname[:-len(".jpg")])
        if imno % 100 == 0:
            gt_results = set()
            gt[imname] = gt_results
        else:
            gt_results.add(imname)

    return (allnames, gt)


def print_perfect():
    " make a perfect result file "
    (allnames, gt) = get_groundtruth()
    for qname, results in gt.iteritems():
        print qname,
        for rank, resname in enumerate(results):
            print rank, resname,
        print


def parse_results(fname):
    """ go through the results file and return them in suitable
    structures"""
    for l in open(fname, "r"):
        fields = l.split()
        query_name = fields[0]
        ranks = [int(rank) for rank in fields[1::2]]
        yield (query_name, zip(ranks, fields[2::2]))


#########################################################################
# main program
def cal_map(infilename, valid_small=True):
    # if len(sys.argv) != 2: usage()
    #
    # infilename = sys.argv[1]

    if valid_small:
        (allnames, gt) = get_groundtruth_trans_imgs(
            "/Users/shijack/PycharmProjects/densenet/cal_map/ground_truth/ground_truth_trans_imgs_v2.dat")
    else:
        (allnames, gt) = get_groundtruth_trans_imgs("./ground_truth/ground_truth_trans_imgs.dat")
    # sum of average precisions
    sum_ap = 0.
    # nb of images so far
    n = 0

    # loop over result lines
    for query_name, results in parse_results(infilename):

        if query_name not in gt:
            print "unknown query ", query_name
            sys.exit(1)

        # sort results by increasing rank
        # results.sort()

        # ground truth
        gt_results = gt.pop(query_name)

        # ranks of true positives (not including the query)
        tp_ranks = []

        # apply this shift to ignore null results
        rank_shift = 0

        for rank, returned_name in results:

            if returned_name not in gt_results:
                # print allnames
                # print "image name %s not in Holidays" % returned_name
                continue

            if returned_name == query_name:
                rank_shift = -1
            elif returned_name in gt_results:
                tp_ranks.append(rank + rank_shift)

        sum_ap += score_ap_from_ranks_1(tp_ranks, len(gt_results))
        n += 1

    if gt:
        # some queries left
        print "no result for queries", gt.keys()
        sys.exit(1)

    print "mAP for %s: %.5f" % (infilename, sum_ap / n)


def cal_detail_result(file_result_vae, file_result_sift, dir_output):
    final_list = []

    final_ori_sift = []
    final_list.append(('final_ori_sift', final_ori_sift))
    final_text_sift = []
    final_list.append(('final_text_sift', final_text_sift))
    final_blur_sift = []
    final_list.append(('final_blur_sift', final_blur_sift))
    final_contrast_down_sift = []
    final_list.append(('final_contrast_down_sift', final_contrast_down_sift))
    final_contrast_up_sift = []
    final_list.append(('final_contrast_up_sift', final_contrast_up_sift))
    final_crop_sift = []
    final_list.append(('final_crop_sift', final_crop_sift))
    final_flip_sift = []
    final_list.append(('final_flip_sift', final_flip_sift))
    final_letterbox_sift = []
    final_list.append(('final_letterbox_sift', final_letterbox_sift))
    final_rot_45_sift = []
    final_list.append(('final_rot_45_sift', final_rot_45_sift))
    final_shift_sift = []
    final_list.append(('final_shift_sift', final_shift_sift))
    final_zoom_down_sift = []
    final_list.append(('final_zoom_down_sift', final_zoom_down_sift))
    final_zoom_up_sift = []
    final_list.append(('final_zoom_up_sift', final_zoom_up_sift))

    final_ori_vae = []
    final_list.append(('final_ori_vae', final_ori_vae))
    final_text_vae = []
    final_list.append(('final_text_vae', final_text_vae))
    final_blur_vae = []
    final_list.append(('final_blur_vae', final_blur_vae))
    final_contrast_down_vae = []
    final_list.append(('final_contrast_down_vae', final_contrast_down_vae))
    final_contrast_up_vae = []
    final_list.append(('final_contrast_up_vae', final_contrast_up_vae))
    final_crop_vae = []
    final_list.append(('final_crop_vae', final_crop_vae))
    final_flip_vae = []
    final_list.append(('final_flip_vae', final_flip_vae))
    final_letterbox_vae = []
    final_list.append(('final_letterbox_vae', final_letterbox_vae))
    final_rot_45_vae = []
    final_list.append(('final_rot_45_vae', final_rot_45_vae))
    final_shift_vae = []
    final_list.append(('final_shift_vae', final_shift_vae))
    final_zoom_down_vae = []
    final_list.append(('final_zoom_down_vae', final_zoom_down_vae))
    final_zoom_up_vae = []
    final_list.append(('final_zoom_up_vae', final_zoom_up_vae))

    (all_names_sift, result_sift) = get_result_trans_imgs(file_result_sift)
    (all_names_vae, result_vae) = get_result_trans_imgs(file_result_vae)
    (allnames, gt) = get_groundtruth_trans_imgs()
    for key, value in gt.items():
        value_sift = result_sift.get(key)
        value_vae = result_vae.get(key)
        for item_value in value:
            if 'origin' in item_value:
                if item_value in value_sift:
                    final_ori_sift.append(item_value)
                if item_value in value_vae:
                    final_ori_vae.append(item_value)
                continue
            if 'add_text' in item_value:
                if item_value in value_sift:
                    final_text_sift.append(item_value)
                if item_value in value_vae:
                    final_text_vae.append(item_value)
                continue
            if 'blur' in item_value:
                if item_value in value_sift:
                    final_blur_sift.append(item_value)
                if item_value in value_vae:
                    final_blur_vae.append(item_value)
                continue
            if 'contrast_down' in item_value:
                if item_value in value_sift:
                    final_contrast_down_sift.append(item_value)
                if item_value in value_vae:
                    final_contrast_down_vae.append(item_value)
                continue
            if 'contrast_up' in item_value:
                if item_value in value_sift:
                    final_contrast_up_sift.append(item_value)
                if item_value in value_vae:
                    final_contrast_up_vae.append(item_value)
                continue
            if 'crop' in item_value:
                if item_value in value_sift:
                    final_crop_sift.append(item_value)
                if item_value in value_vae:
                    final_crop_vae.append(item_value)
                continue
            if 'flip' in item_value:
                if item_value in value_sift:
                    final_flip_sift.append(item_value)
                if item_value in value_vae:
                    final_flip_vae.append(item_value)
                continue
            if 'letterbox' in item_value:
                if item_value in value_sift:
                    final_letterbox_sift.append(item_value)
                if item_value in value_vae:
                    final_letterbox_vae.append(item_value)
                continue
            if 'rot_45' in item_value:
                if item_value in value_sift:
                    final_rot_45_sift.append(item_value)
                if item_value in value_vae:
                    final_rot_45_vae.append(item_value)
                continue
            if 'shift' in item_value:
                if item_value in value_sift:
                    final_shift_sift.append(item_value)
                if item_value in value_vae:
                    final_shift_vae.append(item_value)
                continue
            if 'zoom_down' in item_value:
                if item_value in value_sift:
                    final_zoom_down_sift.append(item_value)
                if item_value in value_vae:
                    final_zoom_down_vae.append(item_value)
                continue
            if 'zoom_up' in item_value:
                if item_value in value_sift:
                    final_zoom_up_sift.append(item_value)
                if item_value in value_vae:
                    final_zoom_up_vae.append(item_value)
                continue
    print len(final_ori_sift), len(final_ori_vae)
    if not os.path.exists(os.path.abspath(dir_output)):
        os.makedirs(dir_output)
    for item_list in final_list:
        with open(dir_output + '/' + item_list[0] + '.txt', 'w') as f:
            f.writelines('\n'.join(item_list[1]))


def generate_statis_no_rs(file_dat, file_result):
    (im_names, result_rank) = statis_result_trans_imgs(file_dat)
    if not os.path.exists('./results/statis/'):
        os.makedirs('./results/statis/')
    with open('./results/statis/' + file_result, 'w') as f:
        for key, value in result_rank.items():
            f.write(key)
            for k in range(len(value)):
                f.write(' ' + str(k) + ' ' + value[k].replace('/opt/dongsl', '/data/datasets'))
            f.write('\n')


if __name__ == "__main__":
    # cal_detail_result('./results/result_vae.dat','./results/result_sift_color.dat','./results/split_20_vae')
    # generate_statis_no_rs('./results/result_sift_color.dat', 'statis_no_sl_sift_color.dat')
    # generate_statis_no_rs('./results/result_vae.dat', 'statis_no_sl_vae.dat')
    file_result = '/Users/shijack/PycharmProjects/densenet/result_generator/results/result_sift_color_query_ccweb_v2.dat'
    cal_map(infilename=file_result)
