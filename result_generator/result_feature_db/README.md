
#1.线上sift(800维)拷贝检测系统mAP计算流程：
##1. 准备文件：待检测的图片文件列表，如: test_2000.txt；
##2. 到10.16.51.204服务器中，执行python index_sift_color.py，得到提取到三个文件：.bow格式的特征、.hash格式的哈希特征、_img_names.txt为待检测图像文件名；
**注意：1. 设置好query_dir_imgs路径：注意：确保目标目录的上级目录仅包含这一个文件夹，比如'/opt/dongsl/tmp2/tmp/'为目标目录，那么tmp2文件夹必须只包含tmp这个文件夹，不能有其他文件或者其他文件夹；
        2. 如果图片文件列表文件中的图片不在204服务器，需要执行index_sift_color.py文件中的copyFiles（）函数，将列表中的图片复制到指定文件夹中，那么在204服务器上执行python index_sift_color.py时，需要注释掉feature_generator_query(target_dir)函数中的copyFiles('./test_2000.txt', target_dir)。**
##3. 将bow格式的特征转化为h5格式：执行python format_bow2h5f.py，得到.h5文件，该文件中包含特征和其对应的图像路径；
##4. 检索：执行python result_generator.py，得到检索结果.dat文件；
##5. 生成groundtruth文件：调用python ground_truth_generator.py 得到gt文件；
##6. 计算mAP值：调用python holidays_map.py得到mAP结果。


##7.举例：
+ 7.1 登录10.16.71.204服务器，并准备好test_2000.txt文件。
+ 7.2 设置query_dir_imgs = '/opt/dongsl/tmp2/tmp/'。其中tmp2目录下只包含tmp文件夹。
+ 7.3 执行python index_sift_color.py，在/opt/dongsl/tmp2 目录下得到tmp.bow tmp.hash tmp_img_names.txt三个文件。
+ 7.4 在format_bow2h5f.py脚本中设置 dir_bow = '/opt/dongsl/tmp2/tmp/' ， file_feature_output = '/opt/dongsl/tmp2/tmp/feature_sift_color_query_ccweb_v2.h5'，file_feature_output为h5文件的路径，执行python format_bow2h5f.py
+ 7.5 检索：在result_generator.py脚本中设置h5_file_query='/opt/dongsl/tmp2/tmp/feature_sift_color_query_ccweb_v2.h5'，h5_file_db='/opt/dongsl/tmp2/tmp/feature_sift_color_query_ccweb_v2.h5'，file_result='/opt/dongsl/tmp2/tmp/result_sift_color_query_ccweb_v2.dat'，执行python result_generator.py，得到检索结果.dat文件。
+ 7.6 在ground_truth_generator.py脚本中设置file_imgs_query，dir_output，调用python ground_truth_generator.py 得到gt文件
+ 7.7 在holidays_map.py中设置检索结果文件,设置gt文件（**修改gt文件中每张图像的路径，确保与检索结果文件中图片路径一致**），调用python holidays_map.py得到mAP结果。


#2.检索加速库faiss——cpu版源码安装步骤：
**result_generator.py脚本中使用了faiss检索加速库，可以从源码编译,所依据的教程为官方教程：https://github.com/facebookresearch/faiss/blob/master/INSTALL.md**
##1.下载源码：git clone https://github.com/facebookresearch/faiss.git,切换目录到faiss/
##1.编译 C++ Faiss：执行./configure && make && make install
##2.编译python接口：执行 make py
##3.使用：在result_generator.py脚本中加入 sys.path.append('/shihuijie/software/faiss/python/')  import faiss
##4.愉快的使用：示例demo可以查看result_generator.py


#3.相似矩阵B.txt和图像名称列表 mAP计算流程：
##1. 设置相似矩阵文件路径和图片列表文件路径，执行python test.py
##2. 在holidays_map.py中设置检索结果文件，调用python holidays_map.py得到mAP结果。