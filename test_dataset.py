from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
import torch

class test_dataset(Dataset):

    def __init__(self,filepath,label=0):
        self.labels = []
        self.file_paths = []
        for root,dirs,files in os.walk(filepath):
            for file in sorted(files):
                if file.endswith('.txt'):
                    self.file_paths.append(os.path.join(root,file))
        print('here')

    def read_folder_data(self,folder_path):
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)

                for root_2, dirs_2, files_2 in os.walk(dir_path):
                    for file in files_2:
                        print(os.path.join(root_2, file))
        # print(self.file_path)
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        filepath = self.file_paths[index]
        mag_data = np.loadtxt(filepath,delimiter=' ')
        col = mag_data.shape[1]
        mag_torch_tensor = torch.from_numpy(mag_data)

        return mag_torch_tensor
    def get_labels(self):
        return self.labels
    def get_one_hot(self,label):
        label_one_hot = np.zeros([self.num_classes])
        label_int = int(label)
        # print(label_int)
        label_one_hot[label_int] = 1
        return label_one_hot

if __name__ == '__main__':
    test = test_dataset('/home/xq/myClassfier/data/0426/0426USB0')
    len = len(test)
    print(len)
    data = test.__getitem__(len-1)
    print(data)