# coding=utf-8
'''
提取dct特征：首先将图像进行分割，然后统计每个patch前四个子带的AC和DC系数，然后产生级联特征。
'''
import cv2
import numpy as np


# split(): split image into patches, and save them
# return: null
def split(img, size=(64, 64), ratio=0.125, n=8, ):
    '''
    分割图像
    :param img: 图像路径
    :param size:
    :param ratio: patch_length/image_length
    :param n: number of patches per line
    :return:
    '''
    assert size[0] == size[1], "size error"
    img_list = []
    img = cv2.resize(cv2.imread(img), size)
    height = img.shape[0]
    width = img.shape[1]
    # cv2.imshow(imgPath, img)
    pHeight = int(ratio * height)
    pHeightInterval = (height - pHeight) / (n - 1)

    pWidth = int(ratio * width)
    pWidthInterval = (width - pWidth) / (n - 1)

    cnt = 1
    for j in range(n):
        for i in range(n):
            x = pWidthInterval * i
            y = pHeightInterval * j

            patch = img[y:y + pHeight, x:x + pWidth, :]
            img_list.append(patch)
            # cv2.imwrite(dstPath + '_%d' % cnt + '.jpg', patch);
            cnt += 1
    return img_list


def dct(image):
    '''
    dct变换
    :param image: image matrix
    :return:
    '''
    img_YUV = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)[:, :, 0]
    img1 = img_YUV.astype('float')  # 将uint8转化为float类型
    img_dct = cv2.dct(img1)
    return img_dct


def printZMatrix(matrix):
    '''
    该段代码只能遍历所有像素点。
    :param matrix:
    :return:
    '''
    result, count = [], 0
    if len(matrix) == 0:
        return result
    m, n = len(matrix), len(matrix[0])
    i, j = 0, 0
    while count < m * n:
        up = True
        while up == True and i >= 0 and j < n:
            result.append(matrix[i][j])
            i -= 1
            j += 1
            count += 1
        # 可以往右走
        if j <= n - 1:
            i += 1
        # 不能往右,就只能往下走
        else:
            i += 2
            j -= 1
        up = False
        while up == False and i < m and j >= 0:
            result.append(matrix[i][j])
            i += 1
            j -= 1
            count += 1
        # 可以往下走
        if i <= m - 1:
            j += 1
        # 不能往下走，就只能往右
        else:
            j += 2
            i -= 1
    return result


def get_dct_feature(img):
    '''
    获取图像dct特征
    :param img: img path
    :return:
    '''
    dct_feature = []
    for img in split(img):
        zmatrix = printZMatrix(dct(img))[:10]
        sub_dct = []
        '''
        sub_dct.append(abs(zmatrix[0]))
        sub_dct.append(abs(zmatrix[1])+zmatrix[2])
        sub_dct.append(abs(zmatrix[3]+zmatrix[4]+zmatrix[5]))
        sub_dct.append(abs(zmatrix[6]+zmatrix[7]+zmatrix[8]+zmatrix[9]))
        '''
        sub_dct.append(zmatrix[0])
        sub_dct.append(zmatrix[1] + zmatrix[2])
        sub_dct.append(zmatrix[3] + zmatrix[4] + zmatrix[5])
        sub_dct.append(zmatrix[6] + zmatrix[7] + zmatrix[8] + zmatrix[9])
        dct_feature.append(sub_dct)
    dct_feature_matrix = np.array(get_code_linear(dct_feature))
    return dct_feature_matrix


def get_code_linear(lists):
    '''
    比较patch相邻sub_dct元素大小
    '''
    j = len(lists) - 1
    hash_list = []
    m, n = 0, 1
    for i in range(j):
        sub_m = lists[m]
        sub_n = lists[n]
        if sub_m[0] > sub_n[0]:
            hash_list.append(1)
        else:
            hash_list.append(0)
        if sub_m[1] > sub_n[1]:
            hash_list.append(1)
        else:
            hash_list.append(0)
        if sub_m[2] > sub_n[2]:
            hash_list.append(1)
        else:
            hash_list.append(0)
        if sub_m[3] > sub_n[3]:
            hash_list.append(1)
        else:
            hash_list.append(0)
        m += 1
        n += 1
    sub_m = lists[j]
    sub_n = lists[0]
    if sub_m[0] > sub_n[0]:
        hash_list.append(1)
    else:
        hash_list.append(0)
    if sub_m[1] > sub_n[1]:
        hash_list.append(1)
    else:
        hash_list.append(0)
    if sub_m[2] > sub_n[2]:
        hash_list.append(1)
    else:
        hash_list.append(0)
    if sub_m[3] > sub_n[3]:
        hash_list.append(1)
    else:
        hash_list.append(0)
    return hash_list


if __name__ == "__main__":
    print get_dct_feature('./all_souls_000000.jpg')
