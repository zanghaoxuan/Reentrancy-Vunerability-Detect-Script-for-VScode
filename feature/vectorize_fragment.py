#将Solidity代码片段向量化。标记（tokenize）、训练FastText模型生成词嵌入（word embeddings）、并将代码片段转换为向量表示
import re
import warnings
import numpy as np
np.random.seed(42)
from gensim.models import FastText
import os
os.environ['PYTHONHASHSEED'] = str(100)

warnings.filterwarnings("ignore")

#定义了三个优先级的运算
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';', 
    '{', '}'
}



class FragmentVectorizer:
    def __init__(self, vector_length):
        #片段列表，向量长度，前向后向切片计数器
        self.fragments = []
        self.vector_length = 60
        self.forward_slices = 0
        self.backward_slices = 0

    """
    Takes a line of solidity code (string) as input
    Tokenizes solidity code (breaks down into identifier, variables, keywords, operators)
    Returns a list of tokens, preserving order in which they appear
    """

    @staticmethod
    #将一行代码转换为标记列表
    def tokenize(line):
        tmp, w = [], []
        i = 0
        while i < len(line):
            # 将当前构建的单词添加到 tmp，并将空格本身也添加到 tmp
            if line[i] == ' ':
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            # 将当前构建的单词添加到 tmp，并将三字符运算符添加到 tmp，然后跳过这三个字符
            elif line[i:i + 3] in operators3:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 3])
                w = []
                i += 3
            #将当前构建的单词添加到 tmp，并将两字符运算符添加到 tmp，然后跳过这两个字符
            elif line[i:i + 2] in operators2:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 2])
                w = []
                i += 2
            #将当前构建的单词添加到 tmp，并将单字符运算符添加到 tmp
            elif line[i] in operators1:
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            #默认处理，将字符添加到当前构建的单词中
            else:
                w.append(line[i])
                i += 1
        # 过滤掉空字符串和空格字符，返回标记列表
        res = list(filter(lambda c: c != '', tmp))
        return list(filter(lambda c: c != ' ', res))

    """
    Tokenize entire fragment
    Tokenize each line and concatenate to one long list
    """

    #对整个代码片段进行标记，并检测代码片段中是否包含函数定义
    @staticmethod
    def tokenize_fragment(fragment):
        tokenized = []
        function_regex = re.compile('function(\d)+')
        backwards_slice = False
        for line in fragment:
            tokens = FragmentVectorizer.tokenize(line)
            tokenized += tokens
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            else:
                backwards_slice = False
        return tokenized, backwards_slice 

    """
    Add input fragment to model
    Tokenize fragment and buffer it to list
    """
    #将标记化的代码片段添加到模型中，并更新前向和后向切片的计数器
    def add_fragment(self, fragment):
        tokenized_fragment, backwards_slice = FragmentVectorizer.tokenize_fragment(fragment)
        self.fragments.append(tokenized_fragment)
        if backwards_slice:
            self.backward_slices += 1
        else:
            self.forward_slices += 1

    #将代码片段转换为向量表示。根据前向或后向切片选择不同的向量填充方式
    def vectorize(self, fragment):
        tokenized_fragment, backwards_slice = FragmentVectorizer.tokenize_fragment(fragment) 
        vectors = np.zeros(shape=(100, self.vector_length))
        if backwards_slice:
            for i in range(min(len(tokenized_fragment), 100)):
                if tokenized_fragment[len(tokenized_fragment) - 1 - i] in self.embeddings:
                    vectors[100 - 1 - i] = self.embeddings[tokenized_fragment[len(tokenized_fragment) - 1 - i]]
        else:
            for i in range(min(len(tokenized_fragment), 100)):
                if tokenized_fragment[i] in self.embeddings:
                    vectors[i] = self.embeddings[tokenized_fragment[i]]
        return vectors

    #加载预训练的FastText模型，获取词嵌入
    def load_model(self):
        model = FastText.load("vocabulary.model") 
        self.embeddings = model.wv 
        del model

    def train_model(self):
        # Set min_count to 1 to prevent out-of-vocabulary errors
        model = FastText(self.fragments, min_count=1, sg=1,vector_size =self.vector_length, window=5,alpha=0.025,batch_words=50)  # sg=0: CBOW; sg=1: Skip-Gram
        self.embeddings = model.wv
        model.save("vocabulary.model")
        del model
        del self.fragments
