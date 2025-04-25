#实现了几种机器学习模型，用于分类任务
#每个模型类都包含训练和测试的方法，用于评估模型性能。这些模型可以通过不同的评估指标（如准确率、召回率、精确率和 F1 分数）来比较它们的表现
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib
import os
import json
from tensorflow.keras.models import load_model

#决策树
class DT:
    def __init__(self, x, y):
        # x = np.tile(x, (2, 1, 1))
        # y = np.tile(y, 2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        self.x_train = np.reshape(x_train, (x_train.shape[0], -1))  # 将输入数据展平
        self.x_test = np.reshape(x_test, (x_test.shape[0], -1))
        self.y_train = y_train
        self.y_test = y_test
        #初始化决策树分类器模型，设置最大深度为 30。
        self.model = DecisionTreeClassifier(max_depth=30)

    """
    Training model
    """

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    """
    Testing model
    """

    def test(self):
        values = self.model.score(self.x_test, self.y_test)
        print("Accuracy: ", values)

        predictions = self.model.predict(self.x_test)
        print(predictions)
        tn, fp, fn, tp = confusion_matrix(self.y_test,predictions).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = recall_score(self.y_test, predictions)
        print('Recall: ', recall)
        precision = precision_score(self.y_test, predictions)
        print('Precision: ', precision)
        print('F1 score: ', f1_score(self.y_test, predictions))

#支持向量机SVM
class SVM:
    def __init__(self,x,y):
        # x = np.tile(x, (5, 1, 1))
        # y = np.tile(y, 5)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        self.x_train = np.reshape(x_train, (x_train.shape[0], -1))  # 将输入数据展平
        self.x_test = np.reshape(x_test, (x_test.shape[0], -1))
        self.y_train = y_train
        self.y_test = y_test
        self.model = SVC(C=1000,kernel='poly')
    """
    Training model
    """
    def train(self):
        self.model.fit(self.x_train, self.y_train)
    """
    Testing model
    """
    def test(self):
        values = self.model.score(self.x_test, self.y_test)
        print("Accuracy: ", values)
        predictions = self.model.predict(self.x_test)
        print(predictions)
        tn, fp, fn, tp = confusion_matrix(self.y_test,predictions).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = recall_score(self.y_test, predictions)
        print('Recall: ', recall)
        precision = precision_score(self.y_test, predictions)
        print('Precision: ', precision)
        print('F1 score: ', f1_score(self.y_test, predictions))

#梯度增强回归树
class gbrt:
    def __init__(self, x, y):
        # x = np.tile(x, (2, 1, 1))
        # y = np.tile(y, 2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        self.x_train = np.reshape(x_train, (x_train.shape[0], -1))  # 将输入数据展平
        self.x_test = np.reshape(x_test, (x_test.shape[0], -1))
        self.y_train = y_train
        self.y_test = y_test
        self.model = GradientBoostingRegressor()

    """
    Training model
    """

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    """
    Testing model
    """

    def test(self):
        values = self.model.score(self.x_test, self.y_test)
        print("Accuracy: ", values)
        predictions = self.model.predict(self.x_test)
        print(predictions)
        tn, fp, fn, tp = confusion_matrix(self.y_test,predictions).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = recall_score(self.y_test, predictions)
        print('Recall: ', recall)
        precision = precision_score(self.y_test, predictions)
        print('Precision: ', precision)
        print('F1 score: ', f1_score(self.y_test, predictions))


from models.x import *
class RDvote:
    def __init__(self, modelX, modelY_1, modelY_2):
        self.modelX = modelX
        self.modelY_1 = modelY_1
        self.modelY_2 = modelY_2
    def save(self, folder_path):
        """
        保存RDvote的子模型：
        modelX保存为.h5
        modelY_1和modelY_2保存为.pkl
        """
        os.makedirs(folder_path, exist_ok=True)

        # 保存modelX
        self.modelX.model.save(os.path.join(folder_path, 'modelX.h5'))
        
        # 保存modelY_1
        joblib.dump(self.modelY_1.model, os.path.join(folder_path, 'modelY_1.pkl'))

        # 保存modelY_2
        joblib.dump(self.modelY_2.model, os.path.join(folder_path, 'modelY_2.pkl'))

        # 保存配置信息
        config = {
            "modelX": "modelX.h5",
            "modelY_1": "modelY_1.pkl",
            "modelY_2": "modelY_2.pkl"
        }
        with open(os.path.join(folder_path, 'config.json'), 'w') as f:
            json.dump(config, f)

        print(f"RDvote and all sub-models saved successfully to '{folder_path}'.")

    def load(self, folder_path):
        """
        加载RDvote的子模型：
        modelX从.h5加载
        modelY_1和modelY_2从.pkl加载
        """
        with open(os.path.join(folder_path, 'config.json'), 'r') as f:
            config = json.load(f)

        # 加载modelX
        self.modelX.model = load_model(os.path.join(folder_path, config["modelX"]))

        # 加载modelY_1
        self.modelY_1.model = joblib.load(os.path.join(folder_path, config["modelY_1"]))

        # 加载modelY_2
        self.modelY_2.model = joblib.load(os.path.join(folder_path, config["modelY_2"]))

        print(f"RDvote and all sub-models loaded successfully from '{folder_path}'.")
        
    def train(self):
        # 训练modelX
        self.modelX.train()

        # 训练modelY_1
        self.modelY_1.train()

        # 训练modelY_2
        self.modelY_2.train()
        
    def test(self):
        # 从ModelX获取预测结果
        self.modelX.test()
        predictions_x = (self.modelX.model.predict(self.modelX.x_test) > 0.5).astype(int)
        predictions_x = predictions_x.flatten()
        # 从ModelY_1获取预测结果
        self.modelY_1.test()
        predictions_y_1 = (self.modelY_1.model.predict(self.modelY_1.x_test) > 0.5).astype(int)
        # 从ModelY_2获取预测结果
        self.modelY_2.test()
        predictions_y_2 = (self.modelY_2.model.predict(self.modelY_2.x_test) > 0.5).astype(int)

        # 合并预测结果（多数投票）
        final_predictions = []  # 假设这是您的原始预测数组
        # 确保两个数组长度相同
        assert len(predictions_x) == len(predictions_y_1) == len(predictions_y_2), "predictions_x and predictions_y must have the same length"

        # 遍历predictions_x和predictions_y的对应元素，并进行多数投票
        for pred_x, pred_y_1, pred_y_2 in zip(predictions_x, predictions_y_1, predictions_y_2):
            # 如果两个预测之和大于等于2，则最终预测为1，否则为0
            final_pred = 1 if (pred_x + pred_y_1 + pred_y_2) >= 2 else 0
            final_predictions.append(final_pred)
        
        
        values = precision_score(final_predictions, self.modelY_1.y_test)
        print("Vote_Accuracy: ", values)
        tn, fp, fn, tp = confusion_matrix(self.modelY_1.y_test,final_predictions).ravel()
        print('Vote_False positive rate(FP): ', fp / (fp + tn))
        print('Vote_False negative rate(FN): ', fn / (fn + tp))
        recall = recall_score(self.modelY_1.y_test, final_predictions)
        print('Vote_Recall: ', recall)
        precision = precision_score(self.modelY_1.y_test, final_predictions)
        print('Vote_Precision: ', precision)
        print('Vote_F1 score: ', f1_score(self.modelY_1.y_test,final_predictions))
        
    def predict(self, modelX_input, modelY_input):

        predictions_x = (self.modelX.predict(modelX_input) > 0.5).astype(int)
        predictions_x = predictions_x.flatten()

        modelY_input = modelY_input.reshape(modelY_input.shape[0], -1)
        predictions_y_1 = (self.modelY_1.predict(modelY_input) > 0.5).astype(int)
        predictions_y_2 = (self.modelY_2.predict(modelY_input) > 0.5).astype(int)
        
        # voting
        final_predictions = []
        assert len(predictions_x) == len(predictions_y_1) == len(predictions_y_2)
        for pred_x, pred_y_1, pred_y_2 in zip(predictions_x, predictions_y_1, predictions_y_2):
            final_pred = 1 if (pred_x + pred_y_1 + pred_y_2) >= 2 else 0
            final_predictions.append(final_pred)
        print(final_predictions)
        return final_predictions 
      

    