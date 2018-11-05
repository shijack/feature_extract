# coding=utf-8

def parse_result_1(file_result_matr, file_names, output_gt_file, output_result_file):
    with open(file_names, 'r') as f:
        tmp_list_names = f.readlines()
    dict_names = {}
    dict_category = {}
    for i in range(len(tmp_list_names)):
        tmp_line = tmp_list_names[i].rstrip().split(' ')
        dict_names[i] = tmp_line[0]
        if not dict_category.has_key(tmp_line[1]):
            dict_category[tmp_line[1]] = [i]
        else:
            tmp_item_result = dict_category[tmp_line[1]]
            tmp_item_result.append(i)
    print len(dict_names)
    print len(dict_category)

    with open(output_gt_file, 'w') as f:
        for tmp_k in dict_category:
            tmp_result_line_nums = dict_category[tmp_k]
            tmp_result_line = []
            for item_tmp_line in tmp_result_line_nums:
                tmp_result_line.append(dict_names[int(item_tmp_line)])
            for item_tmp_k in dict_category[tmp_k]:
                f.write(dict_names[int(item_tmp_k)])
                f.write(' ')
                f.write(' '.join(tmp_result_line))
                f.write('\n')
    with open(file_result_matr, 'r') as f:
        tmp_result = f.readlines()
    with open(output_result_file, 'w') as f:
        for tmp_item_i in range(len(tmp_result)):
            print ('workon the ' + str(tmp_item_i) + ' item total is ' + str(len(tmp_result)))
            tmp_line_result = tmp_result[tmp_item_i].rstrip().split(' ')
            f.write(dict_names[int(tmp_line_result[0])])
            # tmp_line_result = tmp_line_result[:30]
            for tmp_i in range(len(tmp_line_result)):
                f.write(' ')
                f.write(str(tmp_i))
                f.write(' ')
                f.write(dict_names[int(tmp_line_result[tmp_i])])
            f.write('\n')
    print 'ok'

    print len(dict_category)


if __name__ == '__main__':
    parse_result_1(file_result_matr='/Users/shijack/Desktop/copy_dir/result_tmp/B.txt',
                   file_names='/Users/shijack/Desktop/copy_dir/result_tmp/test_imglist.txt', output_gt_file='./gt.dat',
                   output_result_file='./result_all.dat')
