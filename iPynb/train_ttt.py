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
import os
# 使用第一张与第三张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 参数
img_width = 300
img_height = 200  # 图片的长宽
train_data_dir = "../TCM/train/"  # 训练集的文件位置
test_data_dir = "../TCM/val/"  # 测试集的文件位置
train_num = 17917  # 训练集的图片数量
test_num = 3566  # 测试集的图片数量

epochs = 60 # epoch数目
batch_size = 32  # 批次大小
classes = 80 # 样本分类数目
model_name = "idadp_test_3.h5" # model的名称

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
output1 = open("./modelStruc.txt", "w+")
print(model.summary(), file=output1)
model.summary() # 输出网络详细信息

# 数据预处理
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   rotation_range=20,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 数据读取
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                   target_size=(300, 200),
                                                   batch_size=batch_size,
                                                   class_mode="categorical")
test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                 target_size=(300, 200),
                                                 batch_size=batch_size,
                                                 class_mode="categorical")

# model预编译
# Optimizer使用Adam，初始学习速率为0.0005
model.compile(loss="categorical_crossentropy",
             optimizer=Adam(lr=0.0005),
             metrics=["accuracy"])

# Callbacks
# 当val_acc停止提升3个epoch时，降低学习速率
lr_reducing = ReduceLROnPlateau(
    monitor='val_acc', patience=3, verbose=1, mode="max", factor=0.5, min_lr=0.000001)
# 自动保存model
checkpoint = ModelCheckpoint(
    model_name, monitor='val_acc', save_best_only=True)

# 训练模型
#  history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_num // batch_size,
    # epochs=epochs,
    # validation_data=test_generator,
    # validation_steps=test_num // batch_size,
    # callbacks=[lr_reducing, checkpoint])

# 保存训练参数
# loss_history = history.history
# print("history:", loss_history)
# output = open("./train_history1108.txt", "w+")
# print(loss_history, file=output)

from keras import backend as K
K.clear_session()