# coding=utf-8

import datetime			#用于计时

import numpy as np
import mxnet as mx    		#mxnet
from mxnet import autograd,gluon, init, nd, image    #导入自动梯度，gluon前端，图像等模块
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn   #导入模型相关模块
from mxnet.gluon import utils as gutils 

import matplotlib.pyplot as plt	#绘图工具导入
def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for label, output in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int),
                          output.asnumpy()):
            op_argsort = np.argsort(op)[::-1]    #输出排序后的index，最大概率的值对应的index
            lb_int = int(lb)    #标签对应的整数
            ap += 1.0 / (1+list(op_argsort).index(lb_int))    #精度计算 正确的个数
            cnt += 1
    return ((ap, cnt))

def try_gpu(): # 本函数已保存在 gluonbook 包中⽅便以后使⽤。
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx
def transform_train(data, label):
    im = data.astype('float32') / 255		#归并到0~1之间
    #图像增强的函数组定义，并利用ImageNet的预训练均值、方差归一化输入图像
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    rand_crop=True, rand_mirror=True,
                                    mean = np.array([0.485, 0.456, 0.406]),
                                    std = np.array([0.229, 0.224, 0.225]))			
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))    #改变
    return (im, nd.array([label]).asscalar())

# 验证集图片增广，没有随机裁剪和翻转
def transform_val(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    mean = np.array([0.485, 0.456, 0.406]),
                                    std = np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))    #改变格式为 channel width height
    return (im, nd.array([label]).asscalar())

def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
# 在验证集上预测并评估

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
def evaluate(net, data_iter,ctx):
    loss, acc = 0., 0.
    steps = len(data_iter)
    for X,y in data_iter:
        y = y.as_in_context(ctx)
            #计算预训练模型输出的特征
        out_features = net.features(X.as_in_context(ctx))
        outputs = net.output_new(out_features)    #final output
        acc += accuracy(outputs, y)
    loss += nd.mean(softmax_cross_entropy(outputs, y)).asscalar()
    return loss/steps, acc/steps



train_set = gdata.vision.ImageFolderDataset('IDADP/train/',flag=1)
valid_set = gdata.vision.ImageFolderDataset('IDADP/test/',flag=1)
train_valid_set=gdata.vision.ImageFolderDataset('IDADP-PRCV2019-training/datian_photos/',flag=1)

batch_size =64  #32--2821M  could be 64

train_iter = gdata.DataLoader(train_set.transform(transform_train),
                              batch_size, shuffle=True, last_batch='keep', num_workers=0)
valid_iter = gdata.DataLoader(valid_set.transform(transform_val),
                              batch_size, shuffle=True, last_batch='keep', num_workers=0)
train_valid_iter = gdata.DataLoader(valid_set.transform(transform_val),
                              batch_size, shuffle=True, last_batch='keep', num_workers=0)

def get_net(ctx):
    resnet = model_zoo.vision.resnet152_v2(pretrained=True)  #ctx  使用resnet_50作为基本网络抽取特征
    resnet.output_new = nn.HybridSequential(prefix='')     #output is the origin  得到特征，新定义一个输出
    #add two fcn for finetune
    resnet.output_new.add(nn.Dense(256,activation = 'relu'))   #在模型基础上，定义最后两个全连接层
    resnet.output_new.add(nn.Dense(6))
    #initialize
    resnet.output_new.initialize(init.Xavier(),ctx=ctx)  #for fintune
    resnet.collect_params().reset_ctx(ctx)           #for whole net
    return resnet    

#for loss
loss = gloss.SoftmaxCrossEntropyLoss()    #分类损失交叉熵

def get_loss(data,net,ctx):
    l=0.0
    n=0#loss
    for X,y in data:
        y = y.as_in_context(ctx)
        #计算预训练模型输出的特征
        out_features = net.features(X.as_in_context(ctx))
        outputs = net.output_new(out_features)    #final output
        l += loss(outputs,y).mean().asscalar()    #loss for the process
        n += y.size
    return l/n

def train(net,train_iter,valid_iter,num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd', 
                           {'learning_rate':lr, 'momentum':0.9, 'wd': wd})
    plot_loss = []  #plot loss
    tic = datetime.datetime.now()
    print('Traing is begining, please waiting......')
    for epoch in range(num_epochs):
        train_l = 0.0    #存储训练loss
        train_acc=0.0
            #every period step update lr
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)    #every steps updata lr
        #print("There are %d data could train network"%len(train_iter))
        for X,y in train_iter:      #X~32(batch)*1024(iter)= 32768

            
            y = y.astype('float32').as_in_context(ctx)
            #feature
            out_features = net.features(X.as_in_context(ctx))    #预训练直接前传得到特征，未来这一步可以一次性做
            #partly training fineturning
            with autograd.record():
                #features to output, just use features as input
                
                outputs = net.output_new(out_features)    #这里只bp最后两层，只训练最后新定义的部分
                l = loss(outputs, y).sum()
            l.backward()
            
            #for next batch
            trainer.step(batch_size)
            train_l += l.mean().asscalar()
            train_acc += accuracy(outputs, y)
       
        
        toc = datetime.datetime.now()
        h, remainder = divmod((toc - tic).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_s = "time %02d:%02d:%02d" % (h, m, s)
        #validata
        
        if valid_iter is not None:   #验证数据，验证训练效果
  
            valid_loss = get_loss(valid_iter, net, ctx)
            val_loss,val_acc=evaluate(net,valid_iter,ctx)
            epoch_s = ("epoch %d, train loss is %f, valid loss is %f ,train acc is %.3f ,val acc is %.3f"
                       %(epoch+1, train_l/len(train_iter),valid_loss,train_acc/len(train_iter),val_acc))
        else:
            epoch_s = ("epoch %d, train loss is %f ,train acc is %.3f"
                       %(epoch+1, train_l/len(train_iter),train_acc/len(train_iter)))
        tic = toc
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))
        #plot loss
        plot_loss.append(train_l/len(train_iter))
        plt.plot(plot_loss)    #将损失优化结果保存到图里
        plt.savefig("./training_loss.png")

ctx = try_gpu();
num_epochs = 20;
lr = 0.01;wd = 1e-4;lr_period = 10;lr_decay = 0.99;
net = get_net(ctx)    #将网络和数据定义到gpu上
net.hybridize()
train(net,train_iter,valid_iter,num_epochs, lr, wd, ctx, lr_period, lr_decay)    #训练
net.output_new.collect_params().save('./output_new_2_1000.params')       #训练结束后保存参数

