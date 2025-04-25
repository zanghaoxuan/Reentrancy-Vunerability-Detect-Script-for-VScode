#尽量把所有样本的版本都设置一样的，不然编译结果不一样容易报错

#提取与函数调用和变量赋值相关的行位置特征,并将这些特征标准化为固定长度的向量
#提取的是包含call关键字的行位置和包含-=或=运算符的行位置。而position_feature是提取特定函数的位置信息
from solidity_parser import parser
import os
import numpy as np
import pprint
from sklearn.preprocessing import MinMaxScaler



def extract_two_position_feature(code,vectore_length):
    # 初始化计数器
    equal_count = 0
    minus_equal_count = 0
    # 按行拆分代码
    lines = code.split('\n')
    # 遍历每一行代码
    src=[]
    for i, line in enumerate(lines):
      if 'call' in line:
        src.append(i)

    for i, line in enumerate(lines):
      if '-=' in line or '=' in line:
        src.append(i)

    merged_list = src + [0] * (vectore_length - len(src))
    #下面我把数组的范围变成了-1-1
    min_val = np.min(merged_list)
    max_val = np.max(merged_list)
    scaled_array = -1 + 2 * (merged_list - min_val) / (max_val - min_val)
    return [scaled_array]  




# # with open('dataset/code/simple_dao.sol') as f:
# #     code=f.read()
# # extract_2position_feature('dataset/code/simple_dao.sol',code,'withdraw')





