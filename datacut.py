import numpy as np
import os
import datetime
import math
import matplotlib.pyplot as plt
class data_cut():
    def __init__(self,file_path,data_col=1,num_classes = 2, interval=10, cut=100, label_col=5):
        self.file_path = file_path
        self.interval = interval
        self.label_col = label_col
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
        for i in range(0,num_classes):
            tmp_dirname = self.dir_name+'/第' + str(i) +'类'
            self.dir_names.append(tmp_dirname)
            if not os.path.exists(tmp_dirname):
                os.mkdir(tmp_dirname)
            else:
                print("文件已存在")

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
        lists = [[] for _ in range(len(lines[0].split()))]
        col = len(lines[0].split())
        # 遍历每一行，将每个元素添加到对应的列表中
        for line in lines:
            elements = line.strip().split()
            for i, element in enumerate(elements):
                lists[i].append(float(element))
        rows = len(lines)

        iters_double = (rows - self.cut)/self.interval
        iters = math.floor(iters_double)
        labels = []

        list_label = [0,0,0,0]
        label_name = self.dir_name + '/label.txt'
        with open(label_name, 'w') as file:
            pass
        label = 0
        for i in range(0, iters):
            max_index = 0
            list_label = [0 for _ in range(self.num_classes)]
            txt_label = [0 for _ in range(self.cut)]
            label = 0
            start_point = i * self.interval
            file_name = self.dir_name + '/' + str(i) + '.txt'
            # 数据分块
            index_txt = 0
            for j in range(start_point,start_point+self.cut):
                index = int(lists[self.label_col][j])
                list_label[index] = list_label[index] + 1
                txt_label[index_txt] = index
                index_txt = index_txt +1 

            max_index =  txt_label[35]
            for z in range(35,65):
                new_tmp_label =  txt_label[z]
                if(new_tmp_label != max_index):
                    max_index = 0 
                
                
                
            # if( i == 8000):
            #     print(list_label)
            #     plt.plot(filnum)

            # max_index = list_label.index(max_value)

            # if (max_index != 0):
            #     max_index = 1
            file_name = self.dir_name + '/' + str(i) + '.txt'
            with open(label_name, 'a') as file:
                file.write(file_name + ' ' + str(max_index) + '\n')
            labels.append(max_index)
            file_name = self.dir_name + '/' + str(i) + '.txt'
            with open(file_name, 'w') as file:
                for j in range(start_point, start_point + self.cut):
                    list_x = lists[self.data_col][start_point:start_point+self.cut]
                    list_y = lists[self.data_col+1][start_point:start_point + self.cut]
                    list_z = lists[self.data_col+2][start_point:start_point + self.cut]
                    max_x = max(list_x)
                    min_x = min(list_x)
                    max_y = max(list_y)
                    min_y = min(list_y)
                    max_z = max(list_z)
                    min_z = min(list_z)
                    for k in range(self.data_col, col - 1):
                        tmp_num = lists[k][j]
                        if (k == self.data_col):
                            max_data = max_x
                            min_data = min_x
                        if (k == self.data_col+1):
                            max_data = max_y
                            min_data = min_y
                        if (k == self.data_col+2):
                            max_data = max_z
                            min_data = min_z
                        filnum = (float(tmp_num) - float(min_data))/(float(max_data) - float(min_data))
                        filnum = float(tmp_num)
                        if (k == col - 2):
                            file.write(format(filnum, ".4f") + '\n')
                        else:
                            file.write(format(filnum, ".4f") + ' ')


            file_name = self.dir_names[max_index] + '/' + str(i) + '.txt'

            with open(file_name, 'w') as file:
                for j in range(start_point, start_point + self.cut):
                    list_x = lists[self.data_col][start_point:start_point + self.cut]
                    list_y = lists[self.data_col + 1][start_point:start_point + self.cut]
                    list_z = lists[self.data_col + 2][start_point:start_point + self.cut]
                    max_x = max(list_x)
                    min_x = min(list_x)
                    max_y = max(list_y)
                    min_y = min(list_y)
                    max_z = max(list_z)
                    min_z = min(list_z)
                    for k in range(self.data_col, col - 1):
                        tmp_num = lists[k][j]
                        if (k == self.data_col):
                            max_data = max_x
                            min_data = min_x
                        if (k == self.data_col + 1):
                            max_data = max_y
                            min_data = min_y
                        if (k == self.data_col + 2):
                            max_data = max_z
                            min_data = min_z
                        filnum = (float(tmp_num) - float(min_data)) / (float(max_data) - float(min_data))
                        if (k == col - 2):
                            file.write(format(filnum, ".4f") + '\n')
                        else:
                            file.write(format(filnum, ".4f") + ' ')




if __name__ == '__main__':
    ddCut = data_cut('./data/0418/0418data0.txt', data_col=0,num_classes=13,interval=10, cut=100, label_col=3)
    ddCut.cut_data()

    # file_path = r"./datacut.py"
    # file_name = os.path.basename(file_path)
    # file_name_without_extension = os.path.splitext(file_name)[0]
    # current_directory = os.path.dirname(file_path)
    # print(current_directory)
    # print(file_name_without_extension)
    # print(file_name)


