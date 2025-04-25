
# -*- coding: utf-8 -*-
# @File : net.py
# @Author: Runist
# @Time : 2020/7/6 16:52
# @Software: PyCharm
# @Brief: 实现模型分类的网络，MAML与网络结构无关，重点在训练过程
# MAML是一种用于训练模型在少量数据上快速适应新任务的元学习算法
from tensorflow.keras import layers, models, losses
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

#训练和评估模型，使用MAML算法进行元学习
class MAML:
    def __init__(self, input_shape, num_classes):

        self.input_shape = (105,100)
        self.num_classes = 2
        self.meta_model = self.get_maml_model()

    #构建和编译一个简单的神经网络模型
    
    #使用 Keras 的 Sequential 模型。
    #添加一个 Flatten 层将输入展平。
    #添加一个 Dense 层，使用 ReLU 激活函数。
    #添加一个输出层，使用 softmax 激活函数。
    #编译模型，使用 Adam 优化器和 sparse categorical crossentropy 损失函数，并设置准确率为评估指标
    def get_maml_model(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
    #实现了 MAML 的训练过程
    def train_on_batch(self, train_data, inner_optimizer, inner_step, outer_optimizer=None):

        batch_acc = []
        batch_loss = []
        task_weights = []

        # 用meta_weights保存一开始的权重，并将其设置为inner step模型的权重
        meta_weights = self.meta_model.get_weights()

        meta_support_image, meta_support_label, meta_query_image, meta_query_label = next(train_data)
        for support_image, support_label in zip(meta_support_image, meta_support_label):

            # 每个task都需要载入最原始的weights进行更新
            self.meta_model.set_weights(meta_weights)
            for _ in range(inner_step):
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_image, training=True)
                    loss = losses.sparse_categorical_crossentropy(support_label, logits)
                    loss = tf.reduce_mean(loss)

                    acc = tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int32) == support_label, tf.float32)
                    acc = tf.reduce_mean(acc)

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            # 每次经过inner loop更新过后的weights都需要保存一次，保证这个weights后面outer loop训练的是同一个task
            task_weights.append(self.meta_model.get_weights())

        with tf.GradientTape() as tape:
            for i, (query_image, query_label) in enumerate(zip(meta_query_image, meta_query_label)):

                # 载入每个task weights进行前向传播
                self.meta_model.set_weights(task_weights[i])

                logits = self.meta_model(query_image, training=True)
                loss = losses.sparse_categorical_crossentropy(query_label, logits)
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)

                acc = tf.cast(tf.argmax(logits, axis=-1) == query_label, tf.float32)
                acc = tf.reduce_mean(acc)
                batch_acc.append(acc)

            mean_acc = tf.reduce_mean(batch_acc)
            mean_loss = tf.reduce_mean(batch_loss)

        # 无论是否更新，都需要载入最开始的权重进行更新，防止val阶段改变了原本的权重
        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        return mean_loss, mean_acc
