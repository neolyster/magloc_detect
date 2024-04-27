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


def read_folder_data(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)

            for root_2, dirs_2, files_2 in os.walk(dir_path):
                for file in files_2:
                    print(os.path.join(root_2, file))

NAME_LIST = []

# 训练
if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    NAME_LIST.clear()
    batch_size = 1024
    num_classes = 13
    filepath1 = './data/0418/0418data0'
    filepath2 = './data/0418/0418data1'
    logging.basicConfig(filename='0427.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # 路径设置
    dirpathA = './data/0418/0418data0/第0类/'
    dirpathB = './data/0418/0418data0/第1类/'
    dirpathC = './data/0418/0418data0/第2类/'
    dirpathD = './data/0418/0418data0/第3类/'
    dirpathE = './data/0418/0418data0/第4类/'
    dirpathF = './data/0418/0418data0/第5类/'
    model_save_route = './models_save/0426/'
    model_route = ''
    if not os.path.exists(model_save_route):
        os.mkdir(model_save_route)
    else:
        print("文件已存在")

    # label_path = dirpath + '/label.txt'
    dl_tr0 = txt_dataset(filepath1, num_classes, no_label=True, label=0)
    dl_tr1 = txt_dataset(filepath2, num_classes, no_label=True, label=0)

    counts0 = dl_tr0.get_counts()
    counts1 = dl_tr1.get_counts()
    class_counts = counts0 + counts1
    counts0_int = int(sum(counts0))
    counts1_int = int(sum(counts1))
    # 计算每个类别的权重
    print(sum(class_counts))
    # class_weights = torch.tensor(class_counts,dtype=torch.float)
    index_class = 0
    class_weights = np.zeros([num_classes])
    for item in class_counts:
        if (item !=0):
            class_weights[index_class] = 1/item
        else:
            class_weights[index_class] = 0
        index_class = index_class +1
    # dl_trA = txt_dataset(dirpathA, num_classes, no_label=True, label=0)
    # dl_trB = txt_dataset(dirpathB, num_classes, no_label=True, label=1)
    # dl_trC = txt_dataset(dirpathC, num_classes, no_label=True, label=1)
    # dl_trE = txt_dataset(dirpathD, num_classes, no_label=True, label=1)
    # dl_trD = txt_dataset(dirpathE, num_classes, no_label=True, label=2)
    # dl_trF = txt_dataset(dirpathF, num_classes, no_label=True, label=3)
    # train_A = int(0.005*len(dl_trA))
    # test_A = len(dl_trA) - train_A
    # train_A,test_A = random_split(dl_trA,[train_A,test_A])
    # test_valA = txt_dataset(dirpathA, 3, no_label=True, label=0)
    # test_valB = txt_dataset(dirpathB, 3, no_label=True, label=1)
    # test_valC = txt_dataset(dirpathC, 3, no_label=True, label=1)
    # all_val = ConcatDataset([train_A, dl_trB,dl_trC,dl_trD,dl_trE,dl_trF])
    all_val = ConcatDataset([dl_tr0, dl_tr1])
    train_size = int(0.8 * len(all_val))
    test_size = int(0.1 * len(all_val))
    val_size = int(0.1 * len(all_val))
    val_size = len(all_val) - train_size - test_size
    train_dataset, val_dataset,test_dateset = random_split(all_val, [train_size, val_size,test_size])
    dl_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print('验证集长度:', len(dl_val))
    dl_test = DataLoader(test_dateset, batch_size=batch_size, shuffle=False)
    # 生成权重器参数
    train_count = len(train_dataset)
    weights_nums  = np.zeros([train_count])
    print('训练集长度:', len(train_dataset))
    for i ,data in enumerate(train_dataset):
        inputs, label = data
        weights_nums[i] = class_weights[label]

    
    
    
    # 创建权重的随机抽样器
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.tensor(weights_nums), num_samples=train_count,replacement= True)
    dl_tr = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)
    x,y = next(iter(dl_tr))
    print(y)
    print(len(dl_tr))

    print('训练集长度:',len(dl_tr))



    # dl_tr = spectral_dataloader(X, y, idxs=idx_train, batch_size=batch_size, shuffle=False)
    # dl_val = spectral_dataloader(X, y, idxs=idx_val, batch_size=batch_size, shuffle=False)
    # dl_test = spectral_dataloader(X, y, idxs=idx_test, batch_size=batch_size, shuffle=False)
    # ==== data

    # ==== nn
    layers = 4
    hidden_size = 60
    block_size = 2
    hidden_sizes = [hidden_size] * layers
    num_blocks = [block_size] * layers
    input_dim = 100
    in_channels = 64
    n_classes = num_classes
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(0)
    cnn = ResNet(hidden_sizes, num_blocks, input_dim=input_dim,
                 in_channels=in_channels, n_classes=n_classes)
    
    cuda = torch.cuda.is_available()
    if cuda: cnn.cuda()    # cnn.load_state_dict(torch.load(

    
    epochs = 100 # Change this number to ~30 for full training
    Train = 1
    learn_rate = 0.002
    optimizer = optim.Adam(cnn.parameters(), lr=learn_rate, betas=(0.5, 0.999))
    best_val = 0
    no_improvement = 0
    max_no_improvement = 5

    # print("VAl SAMPLES ...")
    # for i in idx_val:
    #     print(NAME_LIST[i])

    for epoch in range(epochs):
        print(' Epoch {}: '.format(epoch))
        logging.debug(' Epoch {}: '.format(epoch))
        # =====train======
        if Train == 1:
            print("开始训练:...")
            logging.debug("开始训练:...")
            print(' lr=', optimizer.param_groups[0]['lr'])
            acc_tr, loss_tr = run_epoch(epoch, cnn, dl_tr, cuda, num_classes=n_classes,training=True, optimizer=optimizer)
            print('训练集准确率 acc: {:0.5f}'.format(acc_tr), '  Train loss: {:0.5f}'.format(loss_tr))
            str_tmp = '训练集准确率 acc: {:0.5f}'.format(acc_tr) + '  Train loss: {:0.8f}'.format(loss_tr)
            logging.debug(str_tmp)
            acc_val, loss_val = run_epoch(epoch, cnn, dl_val, cuda, num_classes=n_classes, training=False, optimizer=optimizer)
            print('测试集准确率 acc  : {:0.2f}'.format(acc_val), '   Val loss: {:0.5f}'.format(loss_val))
            str_tmp = '测试集准确率 acc: {:0.5f}'.format(acc_val) + '  Train loss: {:0.8f}'.format(loss_val)
            logging.debug(str_tmp)
            if epoch%5 ==0:
                acc_val, loss_val = run_epoch(epoch, cnn, dl_test, cuda, num_classes=n_classes,training=False, optimizer=optimizer)
                print('验证集准确率 acc  : {:0.2f}'.format(acc_val), ' Test loss: {:0.5f}'.format(loss_val))
                str_tmp = '验证集准确率 acc: {:0.5f}'.format(acc_val) + '  Train loss: {:0.8f}'.format(loss_val)
                logging.debug(str_tmp)
            if epoch > 10 and learn_rate > 0.0005:
                learn_rate = learn_rate * math.exp(-epoch/(epochs*5))
                optimizer.param_groups[0]['lr'] = learn_rate

            if acc_val > best_val or epoch == 0: # Check performance for early stopping
                best_val = acc_val
                no_improvement = 0
            else:
                no_improvement += 1

            if not os.path.exists(model_save_route):
                os.makedirs(model_save_route)

            if no_improvement >= max_no_improvement:

                # print('Finished after {} epochs!'.format(epoch))
                torch.save(cnn, model_save_route+"m-"+str(epoch))
                # break

            torch.save(cnn, model_save_route+"m-"+str(epoch))



        elif Train == 0:

            print("开始测试:...")
            acc_test, loss_test = run_epoch(epoch, cnn, dl_test, cuda, training=False, optimizer=optimizer)
            print('  盲样准确率 acc  : {:0.2f}'.format(acc_test), '  Test loss: {:0.2f}'.format(loss_test))
            print('  本批次准确率  : {:0.2f}'.format(acc_test))

            ### ====数据表示=================
            count_array =  np.zeros((4, 4))
            y_hat = get_predictions(cnn, dl_test, cuda)
            # for i in range(0, len(y_hat)):
            #     print( '结果: ',y_hat[i], ' 文件:',NAME_LIST[idx_test[i]])
            #     true_y = int(y[idx_test[i]])
            #     pre_y = int(y_hat[i])
            #     count_array[true_y][pre_y] = count_array[true_y][pre_y] + 1

            print("模型判定PreAns 0 1 2 3")
            print(count_array)
            for i in range(0, 4):
                total = 0
                for j in range(0,4):
                    total += count_array[i][j]
                if total != 0:
                    print(count_array[i][i]/total)
                else:
                    print("no Sam")

            break



