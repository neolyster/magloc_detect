import numpy as np
import os
import datetime
import math
import matplotlib.pyplot as plt
class data_cut():
    def __init__(self,file_path,data_col=1,num_classes = 2, interval=10, cut=100):
        self.file_path = file_path
        self.interval = interval
        self.cut = cut
        self.data_col = data_col
        current_directory = os.path.dirname(file_path)
        dirname = os.path.basename(self.file_path)
        dirname = os.path.splitext(dirname)[0]
        self.dir_names = []
        self.dir_name = current_directory + '/' + dirname 
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        else:
            print("文件夹已存在")

        self.dir_nameA = current_directory + '/'+dirname+'/A'
        self.dir_nameC = current_directory + '/'+dirname+'/C'
        self.num_classes = num_classes
  
        # print(self.dir_nameA)
        # if not os.path.exists(self.dir_name):
        #     os.mkdir(self.dir_name)
        # else:
        #     print("文件已存在")
        # if not os.path.exists(self.dir_nameA):
        #     os.mkdir(self.dir_nameA)
        # else:
        #     print("文件已存在")
        # self.dir_nameB = current_directory + '/' + dirname + '/B'
        # print(self.dir_nameB)
        # if not os.path.exists(self.dir_nameB):
        #     os.mkdir(self.dir_nameB)
        # else:
        #     print("文件已存在")
        # if not os.path.exists(self.dir_nameC):
        #     os.mkdir(self.dir_nameC)
        # else:
        #     print("文件已存在")
        # 创建以当前时间为文件名的txt文件
    def cut_data(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        # 初始化列表
        lists = [[] for _ in range(len(lines[0].split())-1)]
        col = len(lines[0].split())
        # 遍历每一行，将每个元素添加到对应的列表中
        for line in lines:
            elements = line.strip().split()
            for i, element in enumerate(elements):
                if(i>=1):
                    lists[i-1].append(float(element))
        rows = len(lines)

        iters_double = (rows - self.cut)/self.interval
        iters = math.floor(iters_double)
        labels = []

        list_label = [0,0,0,0]
        # label_name = self.dir_name + '/label.txt'
        # with open(label_name, 'w') as file:
            # pass
        label = 0
        for i in range(0, iters):
            max_index = 0
            label = 0
            start_point = i * self.interval
            file_name = self.dir_name + '/' + str(i) + '.txt'
            # 数据分块

                
                
            # if( i == 8000):
            #     print(list_label)
            #     plt.plot(filnum)

            # max_index = list_label.index(max_value)

            # if (max_index != 0):
            #     max_index = 1
            file_name = self.dir_name + '/' + str(i) + '.txt'
            labels.append(max_index)
            file_name = self.dir_name + '/' + str(i) + '.txt'
            with open(file_name, 'w') as file:
                for j in range(start_point, start_point + self.cut):
                    list_x = lists[self.data_col-1][start_point:start_point+self.cut]
                    list_y = lists[self.data_col][start_point:start_point + self.cut]
                    list_z = lists[self.data_col+1][start_point:start_point + self.cut]
                    max_x = max(list_x)
                    min_x = min(list_x)
                    max_y = max(list_y)
                    min_y = min(list_y)
                    max_z = max(list_z)
                    min_z = min(list_z)
                    for k in range(0, 3 ):
                        tmp_num = lists[k][j]
                        if (k == 0):
                            max_data = max_x
                            min_data = min_x
                        if (k == 1):
                            max_data = max_y
                            min_data = min_y
                        if (k == 2):
                            max_data = max_z
                            min_data = min_z
                        filnum = (float(tmp_num) - float(min_data))/(float(max_data) - float(min_data))
                        # filnum = float(tmp_num)
                        if (k == 2):
                            file.write(format(filnum, ".8f") + '\n')
                        else:
                            file.write(format(filnum, ".8f") + ' ')






if __name__ == '__main__':
    ddCut = data_cut('./data/0426/0426USB1.txt', data_col=1,num_classes=13,interval=10, cut=100)
    ddCut.cut_data()

    # file_path = r"./datacut.py"
    # file_name = os.path.basename(file_path)
    # file_name_without_extension = os.path.splitext(file_name)[0]
    # current_directory = os.path.dirname(file_path)
    # print(current_directory)
    # print(file_name_without_extension)
    # print(file_name)


