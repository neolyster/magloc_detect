import matplotlib.pyplot as plt
from time import time
import datetime
import numpy as np
import os
import logging
import torch.utils
import torch.utils.data
import torch.utils.data.sampler
from resnet import ResNet
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset,random_split,ConcatDataset
import torch
from datasets import spectral_dataloader
from training import run_epoch, get_predictions
from torch import optim
from numpy import *
import csv
import os
from txt_dataset2 import txt_dataset
import math
from torch.autograd import Variable
from torch import nn


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    filepath1 = './data/0418/0418data0'
    filepath2 = './data/0418/0418data1'
    dirpath0_0 = './data/0418/0418data0/第0类/'
    dirpath0_1= './data/0418/0418data0/第1类/'
    dirpath0_2 = './data/0418/0418data1/第2类/'
    dirpath0_3 = './data/0418/0418data0/第3类/'
    dirpath0_4 = './data/0418/0418data1/第4类/'
    dirpath0_5 = './data/0418/0418data0/第5类/'
    dirpath0_6 = './data/0418/0418data1/第6类/'
    dirpath0_7 = './data/0418/0418data0/第7类/'
    dirpath0_8 = './data/0418/0418data1/第8类/'
    dirpath0_9 = './data/0418/0418data0/第9类/'
    dirpath0_10 = './data/0418/0418data1/第10类/'
    dirpath0_11 = './data/0418/0418data0/第11类/'
    dirpath0_12 = './data/0418/0418data1/第12类/'
    path_lists1 = [dirpath0_1,dirpath0_3,dirpath0_5,dirpath0_7,dirpath0_9,dirpath0_11]
    path_lists2 = [dirpath0_2,dirpath0_4,dirpath0_6,dirpath0_8,dirpath0_10,dirpath0_12]
    MyModel = torch.load('./models_save/0426/m-60')
    input = np.zeros((1,3,100))
    for filepath in path_lists1:
        for root, dirs, files in os.walk(filepath):
            len_files = len(files)
            random_int = random.randint(1, len_files)
            read_path = os.path.join(root, files[random_int])
            ori_path = os.path.join(filepath1, files[random_int])
            data = np.loadtxt(read_path, delimiter=' ')
            data_ori = np.loadtxt(ori_path, delimiter=' ')
            
            input[0,0,:] = data[:, 0]
            input[0,1,:] = data[:, 1]
            input[0,2,:] = data[:, 2]
            input = torch.from_numpy(input)           
            input = input.float()
            input = input.cuda()
            outputs = MyModel(input)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            