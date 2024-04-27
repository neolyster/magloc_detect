from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
import torch

class txt_dataset(Dataset):

    def __init__(self,filepath,num_classes,no_label = False,label=0):
        self.labels = []
        self.file_paths = []
        self.num_classes = num_classes
        if (no_label):
                for root, dirs, files in os.walk(filepath):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        for root_2, dirs_2, files_2 in os.walk(dir_path):
                            for file in files_2:
                                tmp_label = self.extract_numbers(dir)[0]
                                if(tmp_label == 9):
                                    tmp_label = 2
                                if(tmp_label == 11):
                                    tmp_label = 3

                                self.file_paths.append(os.path.join(root_2,file))
                                self.labels.append(tmp_label)
                                # print(os.path.join(root_2, file))

        else:
            with open(filepath, 'r', encoding='utf-8') as file:

                print('label txt located!')
                for line in file:
                    file_path_tmp,label_tmp = line.strip().split(' ')
                    self.file_paths.append(file_path_tmp)
                    self.labels.append(int(label_tmp))
    def read_folder_data(self,folder_path):
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)

                for root_2, dirs_2, files_2 in os.walk(dir_path):
                    for file in files_2:
                        print(os.path.join(root_2, file))
        # print(self.file_path)
    def extract_numbers(self,s):
        return [int(num) for num in re.findall(r'\d+', s)]
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        filepath = self.file_paths[index]
        mag_data = np.loadtxt(filepath,delimiter=' ')
        col = mag_data.shape[1]
        mag_torch_tensor = torch.from_numpy(mag_data)
        label = self.labels[index]
        label_ont_hot = self.get_one_hot(label)
        return mag_torch_tensor, label
    def get_labels(self):
        return self.labels
    def get_one_hot(self,label):
        label_one_hot = np.zeros([self.num_classes])
        label_int = int(label)
        # print(label_int)
        label_one_hot[label_int] = 1
        return label_one_hot
    def get_counts(self):
        label_one_hot = np.zeros([self.num_classes])
        for label in self.labels:
            num = label_one_hot[label]
            label_one_hot[label] = num + 1
        return label_one_hot

if __name__ == '__main__':
    test = txt_dataset('./data/0418/0418data0',num_classes=13,no_label = True)
    len = len(test)
    print(len)
    data,label = test.__getitem__(len-10000)
    print(test.get_counts())
