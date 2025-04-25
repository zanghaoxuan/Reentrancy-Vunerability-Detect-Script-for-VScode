import tensorflow as tf
tf.random.set_seed(100)
import os
os.environ['PYTHONHASHSEED'] = str(100)
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)
from keras.layers import Concatenate,Reshape,Flatten,concatenate,Input,Dropout,Dense,Bidirectional,ReLU,LSTM,Flatten,Dense,Conv1D,MaxPooling1D, GlobalMaxPooling1D,BatchNormalization
from keras.models import Sequential,Model
from keras.optimizers import Adamax
import keras
from keras.utils import to_categorical



#用于创建成对的数据样本，目的是将类别不同的数据对组合在一起，并标记为 1
def make_pairs(x, y):

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    pairs=[]
    labels=[]
    
    # for idx1 in range(len(x)):
    #     x1 = x[idx1]
    #     label1 = y[idx1]

    #     # 与同一类别的样本组合
    #     for idx2 in digit_indices[label1]:
    #         if idx2 != idx1:
    #             x2 = x[idx2]
    #             pairs.append([x1, x2])
    #             labels.append(0)

    #     # 与其他类别的样本组合
    #     for label2 in range(num_classes):
    #         if label2 != label1:
    #             idx2 = random.choice(digit_indices[label2])
    #             x2 = x[idx2]
    #             pairs.append([x1, x2])
    #             labels.append(1)


    # 类别0中的两两组合
    # for i in range(len(digit_indices[0])):
    #     for j in range(i + 1, len(digit_indices[0])):
    #         idx1 = digit_indices[0][i]
    #         idx2 = digit_indices[0][j]
    #         x1 = x[idx1]
    #         x2 = x[idx2]
    #         pairs.append([x1, x2])
    #         labels.append(0)
    print(len(labels)) 
    #类别0和类别1的组合
    for idx1 in digit_indices[0]:
        for idx2 in digit_indices[1]:
            x1 = x[idx1]
            x2 = x[idx2]
            pairs.append([x1, x2])
            labels.append(1)

    print(len(labels))        
    # # 类别1中的两两组合
    # for i in range(5): #len(digit_indices[1])
    #     for j in range(i + 1, len(digit_indices[1])):
    #         idx1 = digit_indices[1][i]
    #         idx2 = digit_indices[1][j]
    #         x1 = x[idx1]
    #         x2 = x[idx2]
    #         pairs.append([x1, x2])
    #         labels.append(0)
    # print(len(labels))

    return np.array(pairs), np.array(labels)

    # for idx1 in range(len(x)):
    #   x1=x[idx1]
    #   label1=y[idx1]
    #   idx2=random.choice(digit_indices[label1])
    #   x2=x[idx2]
    #   pairs +=[[x1,x2]]
    #   labels+=[0]
    #   label2 = random.randint(0, num_classes - 1)
    #   while label2 == label1:
    #       label2 = random.randint(0, num_classes - 1)

    #   idx2 = random.choice(digit_indices[label2])
    #   x2 = x[idx2]
    #   pairs += [[x1, x2]]
    #   labels += [1]
    # return np.array(pairs), np.array(labels)

from keras import backend as K
#用于计算欧几里得距离和对比损失，用于 Siamese 网络的训练
def euclidean_distance(inputs):
    x, y = inputs
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(sum_square)
def loss(margin=1):

    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - (y_pred), 0))
        return K.mean((1 - K.cast(y_true, dtype='float32')) * square_pred + (K.cast(y_true, dtype='float32')) * margin_square)

    return contrastive_loss



#构建和训练一个结合 CNN 和 LSTM 的网络，用于处理时间序列数据
class cnn_lstm:
    #类的初始化方法中，利用SMOTE对数据进行重采样，从而增强数据集
    def __init__(self, x_data, y_label,batch_size=64, lr=0.001, epochs=20): 
        x_data = np.tile(x_data, (1, 1, 1))
        y_label = np.tile(y_label, 1)

        # initial_length = len(x_data)
        # original_x_data = x_data.copy()
        # original_y_label = y_label.copy()

        # # 循环5次
        # for i in range(2):
        #     # 重新排列顺序
        #     num_samples = initial_length
        #     indices = np.arange(num_samples)
        #     np.random.shuffle(indices)
        #     original_x_data = original_x_data[indices]
        #     original_y_label = original_y_label[indices]
            
        #     # 扩大x_data和y_label
        #     if i != 0:
        #         x_data=np.concatenate((x_data, original_x_data[:-i]))
        #         y_label=np.concatenate((y_label, original_y_label[:-i]))
        
        
        print(x_data.shape)
        num_samples, seq_length, vector_dim = x_data.shape
        x_data_reshaped = x_data.reshape(num_samples, seq_length * vector_dim)
        
        x_train,x_test,y_train,y_test=train_test_split(x_data,y_label,test_size=0.3,random_state=42)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self):
      input_data = Input(shape=(100, 60))
      x = Conv1D(128, 7, activation='relu')(input_data)
      x = MaxPooling1D(2)(x)
      x = Conv1D(64,3 , activation='relu')(x)
      x = GlobalMaxPooling1D()(x)
      # Reshape to make it compatible with LSTM input
      x = Reshape((-1, 64))(x)

      y = LSTM(128,return_sequences = True)(x)
      y = LSTM(128)(y)
      y = Dense(32, activation='relu')(y)
      y = Dense(16, activation='relu')(y)
      z = Dense(1, activation='sigmoid')(y)

      # Compiling the model
      model = Model(inputs=input_data, outputs=z)
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

      # Plotting the model
      keras.utils.plot_model(model, 'tmp.png', show_shapes=True)

      # Saving the model
      self.model = model
    
    def train(self):

      history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size,
                      epochs=self.epochs,
                      shuffle=True)
      self.model.save_weights("save/model.pkl")
      self.model.save_weights("save/model.pkl")


    def test(self):
        self.model.load_weights("save/model.pkl")
        values = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy: ", values[1])

        predictions = self.model.predict(self.x_test, batch_size=self.batch_size)
        predictions_binary = (predictions > 0.5).astype(int)
        print(predictions_binary.flatten())
        
        tn, fp, fn, tp = confusion_matrix(self.y_test,predictions_binary).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall: ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))  
    
    

    
#构建和训练一个 CNN 模型，用于处理文本数据
#提取语义特征，使用文本卷积神经网络（textCNN）
class textcnn:
    def __init__(self, x, y,batch_size=256, lr=0.01, epochs=20):
        x = np.tile(x, (100, 1, 1))
        y = np.tile(y, 100)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.9,random_state=42)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        self.class_weight={i:w for i,w in enumerate(self.class_weight)}

        model = Sequential()
        model.add(Conv1D(128, 5, activation='relu', input_shape=(100, 40)))
        model.add(BatchNormalization())
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(BatchNormalization())
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
    """
    Training model
    """

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size,
                       epochs=self.epochs,
                       class_weight=self.class_weight,shuffle=False)
        self.model.save_weights("save/model.pkl")

    """
    Testing model
    """
    def test(self):
        self.model.load_weights("save/model.pkl")
        values = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy: ", values[1])
        
        predictions = self.model.predict(self.x_test, batch_size=self.batch_size)
        predictions = (predictions[:, 1]).round()
        predictions = predictions.astype(int)
 
        tn, fp, fn, tp = confusion_matrix(self.y_test,predictions).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall: ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))  


#构建和训练一个 LSTM 模型，用于处理文本数据
class textlstm:
    def __init__(self, x, y,batch_size=64, lr=0.01, epochs=10):
        print(x.shape)
        x = np.tile(x, (1, 1, 1))
        print(x.shape)
        y = np.tile(y,1)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        self.class_weight={i:w for i,w in enumerate(self.class_weight)}

        model = Sequential()
        model.add(LSTM(128,input_shape=(100, 60)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Training model
    """

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size,
                       epochs=self.epochs,
                       class_weight=self.class_weight,shuffle=False)
        self.model.save_weights("./save/model.pkl")

    """
    Testing model
    """
    def test(self):
        self.model.load_weights("./save/model.pkl")
        values = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy: ", values[1])
        
        predictions = self.model.predict(self.x_test, batch_size=self.batch_size)
        predictions = (predictions[:, 1]).round()
        predictions = predictions.astype(int)
 
        tn, fp, fn, tp = confusion_matrix(self.y_test,predictions).ravel()
        print(tn,fp,fn,tp)
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall: ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))