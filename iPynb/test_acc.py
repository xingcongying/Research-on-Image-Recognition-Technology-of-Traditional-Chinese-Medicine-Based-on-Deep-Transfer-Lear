# coding=utf-8

# 导入模块
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import warnings
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras import backend
from keras import layers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, PReLU
from keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

import numpy as np
import json

warnings.filterwarnings("ignore")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 参数
img_width = 300
img_height = 200  # 图片的长宽
train_data_dir = "../TCM/train/"  # 训练集的文件位置
test_data_dir = "../TCM/val/"  # 测试集的文件位置

epochs = 60 # epoch数目
batch_size = 8  # 批次大小
classes = 80 # 样本分类数目
model_name = "idadp_test_1.h5" # model的名称

#卷积层Conv Block
def conv_block(x, growth_rate, name):
    bn_axis = 3
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + "_0_bn")(x)  # 批量标准化层
    x1 = layers.Activation("relu", name=name + "_0_relu")(x1)  # 激活
    x1 = layers.Conv2D(4 * growth_rate,
                       1,
                       use_bias=False,
                       name=name + "_1_conv")(x1)  # 2D卷积核

    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + "_1_bn")(x1)
    x1 = layers.Activation("relu", name=name + "_1_relu")(x1)
    x1 = layers.Conv2D(growth_rate,
                       3,
                       padding='same',
                       use_bias=False,
                       name=name + "_2_conv")(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + "_concate")([x, x1])
    return x

#Dense Block
#卷积层的叠加, 每一个卷积层都与其之上的同一个DenseBlock中的卷积层有相互链接
def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + "_block" + str(i + 1))
    return x

#Transition Block
#连接每个Dense Block
def transition_block(x, reduction, name):
    bn_axis = 3
    x = layers.BatchNormalization(axis=bn_axis,
                                  epsilon=1.001e-5,
                                  name=name + "_0_bn")(x)
    x = layers.Activation('relu', name=name + "_relu")(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction),
                      1,
                      use_bias=False,
                      name=name + "_conv")(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + "_pool")(x)
    return x

# DenseNet的实现
def dense_net(blocks, input_shape=(600, 400, 3), classes=6):
    img_input = Input(shape=input_shape)

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name="conv1/conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1/bn")(x)
    x = layers.Activation("relu", name="conv1/relu")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)
    
    x = dense_block(x, blocks[0], name="conv2")
    x = transition_block(x, 0.5, name="pool2")
    x = dense_block(x, blocks[1], name="conv3")
    x = transition_block(x, 0.5, name="pool3")
    x = dense_block(x, blocks[2], name="conv4")
    x = transition_block(x, 0.5, name="pool4")
    x = dense_block(x, blocks[3], name="conv5")
    
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation="softmax", name="fc6")(x)
    
    inputs = img_input
    model = Model(inputs, x, name="densenet")
    
    return model

model = dense_net(blocks=[6, 12, 48, 32], input_shape=(300, 200, 3), classes=80)
model.summary() # 输出网络详细信息


# 测试模型
model.load_weights("./idadp_test_1.h5") # 读取模型

import os
train_dir = "../TCM/train/"
test_dir = "../TCM/test/"
val_dir = "../TCM/val/"
print(os.listdir(test_dir))

def get_result(file_dir):
    class_index = [1,10,11,12,13,14,15,16,17,18,19,
                   2,20,21,22,23,24,25,26,27,28,29,
                   3,30,31,32,33,34,35,36,37,38,39,
                   4,40,41,42,43,44,45,46,47,48,49,
                   5,50,51,52,53,54,55,56,57,58,59,
                   6,60,61,62,63,64,65,66,67,68,69,
                   7,70,71,72,73,74,75,76,77,78,79,
                   8,80,
                   9]
    images = os.listdir(file_dir)
    result = []
    for image_name in images:
        image_path = file_dir + image_name  # 单个测试图片位置
        image_file = image.load_img(image_path,
                                target_size=(300, 200))
        x = image.img_to_array(image_file) / 255.0
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        temp_json = dict()
        temp_json["image_id"] = image_name
        temp_json["category"] = class_index[int(np.argmax(prediction, axis=1))]
        result.append(temp_json)
    return result

def json_output(result, file_name):
    json2 = json.dumps(result)
    f = open(file_name,'w',encoding='utf-8')
    f.write(json2)
    f.close()
    

def get_test_acc(file_dir, category):
    class_index = [1,10,11,12,13,14,15,16,17,18,19,
                   2,20,21,22,23,24,25,26,27,28,29,
                   3,30,31,32,33,34,35,36,37,38,39,
                   4,40,41,42,43,44,45,46,47,48,49,
                   5,50,51,52,53,54,55,56,57,58,59,
                   6,60,61,62,63,64,65,66,67,68,69,
                   7,70,71,72,73,74,75,76,77,78,79,
                   8,80,
                   9]
    images = os.listdir(file_dir)
    result = []
    count = 0
    correct = 0
    for image_name in images:
        path = file_dir + image_name
        file = image.load_img(path, target_size=(300, 200))
        x = image.img_to_array(file) / 255.0
        x = np.expand_dims(x, axis=0)
        prediction = class_index[int(np.argmax(model.predict(x), axis=1))]
        count = count + 1
        
        if prediction is category:
            correct = correct + 1
        else:
            print(category,prediction)
#     print(correct, "/", count)
    acc = correct / count
    return acc

def test1(file_dir, classes):
    sum = 0
    f = open("test_acc1214.txt", "w+")
    for i in range(1, classes + 1):
        temp_acc = get_test_acc(file_dir + str(i) + "/", i)
        print("category", i, ":", temp_acc)
        print("category", i, ":", temp_acc, file=f)
        sum = sum + temp_acc
    print("test_acc:", sum / classes, file=f)

def test2(file_dir, classes):
    for i in range(1, classes + 1):
        get_result(file_dir + str(i) + "/")

test1(test_dir, 80)