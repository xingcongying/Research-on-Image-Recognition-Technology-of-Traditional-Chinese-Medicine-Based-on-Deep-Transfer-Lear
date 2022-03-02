from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os 
import random 
import time
import json
import torch
from collections import OrderedDict
import torchvision
import numpy as np 
import pandas as pd 
import warnings
from datetime import datetime
from torch import nn,optim
from config import config 
from collections import OrderedDict
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
from IPython import embed
#1. set random.seed and cudnn performance
#test_data /home/hyq-user/TCM/test/(1......80)

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
import torch
torch.cuda.set_device(1) 
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

pic_sum = 0
pic_cor = 0
#3. test model on public dataset and save the probability matrix
def test(test_loader,model,folds,name):
	global pic_sum
	global pic_cor
	sum_ = 0
	cor = 0
	model.cuda()
	model.eval()
	tta = False
	for i,(input,filepath) in enumerate(tqdm(test_loader)):
        #3.2 change everything to cuda and get only basename
		filepath = [os.path.basename(x) for x in filepath]
		with torch.no_grad():
			image_var = Variable(input).cuda()
            #3.3.output
            #print(filepath)
            #print(input,input.shape)
            
            #temp = OrderedDict()

			if tta == False:
				pic_sum = pic_sum+1
				sum_ = sum_ + 1
				y_pred = model(image_var)
				smax = nn.Softmax(1)
				smax_out = smax(y_pred)
				label = np.argmax(smax_out.cpu().data.numpy())+1
				true_label = int(name)
			#	current_label = int((str(label)))
				if(true_label == int((str(label)))):
					pic_cor = pic_cor + 1
					cor = cor + 1


                #temp["imaged_id"] = filepath[0]
                #temp["disease_class"] = int((str(label)))
                #result.append(temp)
                #test_labels.write(filepath[0]+","+str(label)+"\n")

    #test_labels.write(json.dumps(result))
    #test_labels.close()
	percent = cor/sum_
	message = "类别:"+name + "准确率 :" + str(percent)
	print(message)
	with open("./test_results/%s.txt"%config.model_name,"a") as f:
		print(message,file = f)
		f.close() 

#4. more details to build main function    
def main():
	fold = 0
	global pic_sum
	global pic_cor
    #4.1 mkdirs
	if not os.path.exists(config.submit):
		os.mkdir(config.submit)
	if not os.path.exists(config.weights):
		os.mkdir(config.weights)
	if not os.path.exists(config.best_models):
		os.mkdir(config.best_models)
	if not os.path.exists(config.logs):
		os.mkdir(config.logs)
	if not os.path.exists(config.weights + config.model_name + os.sep +str(fold) + os.sep):
		os.makedirs(config.weights + config.model_name + os.sep +str(fold) + os.sep)
	if not os.path.exists(config.best_models + config.model_name + os.sep +str(fold) + os.sep):
		os.makedirs(config.best_models + config.model_name + os.sep +str(fold) + os.sep)       

	model = get_net()
	model.cuda()
	best_model = torch.load("checkpoints/best_model/%s/0/model_best.pth.tar"%config.model_name)
	model.load_state_dict(best_model["state_dict"])
	
	for i in range(1,81):
		test_files = get_files(config.test_data+str(i)+'/',"test")

		test_dataloader = DataLoader(ChaojieDataset(test_files,test=True),batch_size=1,shuffle=False,pin_memory=False)

		test(test_dataloader,model,fold,str(i))
	with open("./test_results/%s.txt"%config.model_name,"a") as f:
		print(pic_cor/pic_sum,file =f)
		f.close()
	print(pic_cor/pic_sum)

if __name__ =="__main__":
	main()





















