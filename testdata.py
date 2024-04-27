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
from test_dataset import test_dataset
import math
from torch.autograd import Variable

if __name__ == '__main__':
    model = torch.load('./models_save/0426/m-60')
    model.eval()
    dataset = test_dataset('/home/xq/myClassfier/data/0426/0426USB0')
    save_name = 'USB0.txt'
    logging.basicConfig(filename='test.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    data_loader = DataLoader(dataset, batch_size=64,shuffle=False)
    for batch in data_loader:
        # batch =Variable(batch)
        batch = batch.permute(0, 2, 1)
        batch = batch.type(torch.cuda.FloatTensor)
        inputs = torch.tensor(batch)  # 将数据转换为PyTorch张量
        outputs = model(inputs)  # 将数据输入到模型中
        ddd, predicted = torch.max(outputs.data, 1)
        numpy_array = predicted.cpu().numpy()
        # np.savetxt('usb1.txt', numpy_array, fmt='%f\n')
        with open(save_name, 'a') as f:
            np.savetxt(f, numpy_array, fmt='%f\n')
        # 然后使用savetxt保存到txt文件
        # np.savetxt('tensor.txt', numpy_array, fmt='%f')
        # torch.save(predicted,'test.txt')
        print(outputs)



