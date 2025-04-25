#从文件中解析和清理代码片段，并使用这些片段训练一个向量化模型。
#将代码片段转换为向量表示。
#整合多个代码片段进行向量化训练。
from feature.vectorize_fragment import *
import pandas
import numpy as np
np.random.seed(42)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONHASHSEED'] = str(100)
from solidity_parser import parser
from clean_fragment import *
#从给定文件中解析代码片段，清理它们，并训练一个向量化模型
def train_vector(filename, vector_length=60):   #数据集中的所有片段都要合起来训练,
    fragments = []
    count = 0
    vectorizer = FragmentVectorizer(vector_length)
    for fragment, val in parser.parse_file(filename):
        fragment = clean_fragment(fragment)   
        count += 1
        print("Collecting fragments...", count, end="\r")
        #将清理后的片段添加到向量化器中，并将其存储在列表中
        vectorizer.add_fragment(fragment)
        row = {"fragment": fragment, "val": val}
        fragments.append(row)

    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print()
    print("Training model...", end="\r")
    vectorizer.train_model()

#从给定代码片段中提取特征向量
def extract_text_feature(code):   
    
    code_lines = [line.strip() for line in code.split('\n') if line.strip()]                                    
    vectorizer = FragmentVectorizer(60)
    fragment = clean_fragment(code_lines)
    vectorizer.load_model()
    vector = vectorizer.vectorize(fragment)
    return vector   

#从多个代码片段中进行向量化训练
def my_train_vector(combine_snipples, vector_length=60):   #数据集中的所有片段都要合起来训练,
    count = 0
    vectorizer = FragmentVectorizer(vector_length)
    for i in range(len(combine_snipples)):
        fragment = clean_fragment(combine_snipples[i])   #我后加的,它原来没有加
        count += 1
        vectorizer.add_fragment(fragment)
    vectorizer.train_model()

# extract_text_feature(code)

