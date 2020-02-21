import cv2
import matplotlib.pyplot as plt
import os
import random
import torch
import numpy as np
from PIL import Image
import tensorflow as tf

import keras 
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.models import Model, Sequential


data_dir = "/project/face/train"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)

def k_fold(X, y, splits=10, shuffle=True):
    if len(X) != len(y):
        raise ValueError("The number of X should be the same as y")
    if shuffle:
        z = list(zip(X, y))
        random.shuffle(z)
        X,y = zip(*z)
    num = len(X)
    fold_sizes = np.full(splits, num//splits, dtype=int) 
    fold_sizes[:num % splits] += 1 
    current = 0
    for fold_size in fold_sizes:
        X_train, y_train = X[0:current] + X[current + fold_size:], \
                           y[0:current] + y[current + fold_size:]
        X_test, y_test = X[current : current + fold_size], \
                         y[current : current + fold_size]
        yield X_train, y_train, X_test, y_test


class GauCNN():
    def __init__(self, p_input_size = (160,160,3), \
                 p_output_size = 4, p_epoches = 100):
        self._net = Sequential()
        self.input_size = p_input_size
        # self.histories = []
        self.accuracies = []
        self.X = []
        self.y = []
        self.epoches = p_epoches
        self.output_size = p_output_size
        self.k_splits = 10
        self.k = 0
    # 重置所有变量
    def reset(self):
        del self._net
        K.clear_session()
        self._net = Sequential()
        self.build()
    # 读取数据
    def load_dataset(self):
        data_x = [] 
        data_y = []
        dir_label = {}
        child_dirs = []
        name_dict = {}

        # 遍历根目录获取图像文件夹路径以及对应的label
        name_list = list(os.listdir(data_dir))
        for i, name in enumerate(name_list):
            name_dict[name] = i

        for dir in os.listdir(data_dir):
            # 获取文件夹（明星）名称 eg. dir_file_path = './face/train/zhoudongyu'
            dir_file_path = os.path.join(data_dir,dir)
            # 构造图像文件夹路径字典 eg. dir_label['./face/train/zhoudongyu'] = 0
            dir_label[dir_file_path] = name_dict[dir]  
            
        # 遍历所有文件夹，读取图像,（给测试集添加噪声）
        for dir, label in dir_label.items():
        # 遍历单一文件夹，获取所有图像路径
            child_dirs = os.listdir(dir)
            
            # 逐一读取图像
            for child_dir in child_dirs:
                # 读取图像
                img = cv2.imread(dir+'/'+child_dir, \
                                cv2.IMREAD_UNCHANGED)
                # 归一图像大小
                img = cv2.resize(img, (160, 160), \
                                interpolation = cv2.INTER_AREA) 
                # 将张量的值归一到（0，1）内
                img = img/255
                if img.shape != (160, 160, 3):
                    pass 
                else:
                    # 将图像添加到数据列中
                    data_x.append(img)
                    # 将label添加到数据列中
                    data_y.append(label)      
            
        self.X = data_x.copy()
        self.y = data_y.copy()
    # 构建网络
    def build(self):
        self._net.add(Conv2D(filters=96, kernel_size=11, input_shape=self.input_size))
        self._net.add(MaxPooling2D(pool_size=3, strides=2))
        self._net.add(Conv2D(filters=256, kernel_size=5, padding='same'))
        self._net.add(MaxPooling2D(pool_size=3, strides=2))
        self._net.add(Conv2D(filters=384, kernel_size=3, padding='same'))
        # self._net.add(Conv2D(filters=384, kernel_size=3, padding='same'))
        self._net.add(Conv2D(filters=256, kernel_size=3, padding='same'))
        self._net.add(MaxPooling2D(pool_size=3, strides=2))
        self._net.add(Flatten())
        self._net.add(Dense(4096))
        self._net.add(Dropout(0.5))
        self._net.add(Dense(1024))
        self._net.add(Dropout(0.5))
        self._net.add(Dense(64))
        self._net.add(Dropout(0.5))
        self._net.add(Dense(4))
        self._net.summary()

    def netsummary(self):
        self._net.summary()
    # 进行一次训练
    def train_one_fold(self, p_X, p_y, p_X_test, p_y_test):
        l_X = np.asarray(p_X)
        l_y = np.eye(self.output_size)[np.array(p_y)]
        l_history = None 

        self._net.compile(optimizer='Adadelta', loss='categorical_crossentropy', \
                          metrics=['accuracy'])
        l_history =self._net.fit(l_X, l_y, batch_size=16, \
                                    epochs=self.epoches,validation_split=0.1)   
        l_acc = self.test(p_X_test, p_y_test)
        self.accuracies.append(l_acc)                      
        self.save_history(l_history, l_acc)
    # 打印记录
    def save_history(self, p_history, p_acc):
        # 绘制训练 & 验证的准确率值
        plt.subplot(2,1,1)
        plt.plot(p_history.history['accuracy'])
        plt.plot(p_history.history['val_accuracy'])
        plt.hlines(p_acc,0,self.epoches)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val', 'Test'], loc='upper left')

        # 绘制训练 & 验证的损失值
        plt.subplot(2,1,2)
        plt.plot(p_history.history['loss'])
        plt.plot(p_history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.savefig('./Saved/'+str(self.k)+'\'s_fold.png')
        plt.close()
    
    # 打印网络结构
    def summary(self):
        self._net.summary()
    # 将数据集拆成ｋ份进行交叉验证
    def k_fold(self, X, y, shuffle=True):
        if shuffle:
            z = list(zip(X, y))
            random.shuffle(z)
            X,y = zip(*z)
        num = len(X)
        fold_sizes = np.full(self.k_splits, num//self.k_splits, dtype=int) 
        fold_sizes[:num % self.k_splits] += 1 
        current = 0
        for fold_size in fold_sizes:
            X_train, y_train = X[0:current] + X[current + fold_size:], \
                            y[0:current] + y[current + fold_size:]
            X_test, y_test = X[current : current + fold_size], \
                            y[current : current + fold_size]
        yield X_train, y_train, X_test, y_test
    # 训练
    def train(self):
        self.k = 1
        self.load_dataset()
        for X_train, y_train, X_test, y_test in k_fold(self.X, self.y):
            self.reset()
            self.train_one_fold(X_train, y_train, X_test, y_test)
            self.k += 1
        acc_sum = np.array(self.accuracies)
        print("Average accuracy: ", str(acc_sum.sum()/len(self.accuracies)))
    
    def test(self, p_test_X, p_test_Y, p_model_name='NAS_CNN'):
        l_test_X = np.asarray(p_test_X)
        l_test_y = np.eye(self.output_size)[np.array(p_test_Y)]
        # 获取预测值
        preds_Y = self._net.predict(l_test_X) # type(preds_Y) = list
        # 将预测值转为张量
        preds_Y = torch.from_numpy(preds_Y)
        # 获取可能性最大的label并构建list
        _, outputs = torch.max(preds_Y, 1)
        
        # 将test_Y（原为独热编码）转成label的形式（eg.[0,1,2...])
        l_test_y = torch.from_numpy(l_test_y)
        _, labels = torch.max(l_test_y, 1)
        
        # 计算准确率
        correctness = (outputs == labels).sum()
        acc = int(correctness) / len(labels)
        print("%s Accuracy: %.3f" % (p_model_name, acc))
        return acc
    
if __name__ == '__main__':
    a = GauCNN(p_epoches=100)
    a.train()