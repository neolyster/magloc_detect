from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# torchvision 是个图形库，服务于pytorch
import torch
import numpy as np

# Dataset 是pytorch 的一个抽象类，继承之
class SpectralDataset(Dataset):
    """
    Builds a dataset of spectral data. Use idxs to specify which samples to use
    for dataset - this allows for random splitting into training, validation,
    and test sets. Instead of passing in filenames for X and y, we can also
    pass in numpy arrays directly.
    """
                        # X数据,y标签,idxs 样本编号序列
    # X_fn = 10*1000*1
    def __init__(self, X_fn, y_fn, idxs=None, transform=None):
        if type(X_fn) == str:
            self.X = np.load(X_fn) # 读取npy文件  X = 10*1000*1 list
        else:
            self.X = X_fn
        if type(y_fn) == str:
            self.y = np.load(y_fn) # 读取npy文件
        else:
            self.y = y_fn
        if idxs is None: idxs = np.arange(len(self.y)) # idxs 样本编号 1...n
        self.idxs = idxs
        self.transform = transform # 数据处理函数流

    def __len__(self):
        return len(self.idxs)

    # 读取 idx 第几个（并非样本库中真实序号）
    def __getitem__(self, idx):
        i = self.idxs[idx] # 取出第几个数据   # self.idxs真实样本编号
        x, y = self.X[i], self.y[i] # x = 1000*1
        x = np.expand_dims(x, axis=0) # x= 1*1000
        if self.transform:
            x = self.transform(x)
        return (x, y)


### TRANSFORMS ###


class GetInterval(object):
    """
    Gets an interval of each spectrum.
    """
    def __init__(self, min_idx, max_idx):
        self.min_idx = min_idx
        self.max_idx = max_idx

    def __call__(self, x):
        x = x[:,self.min_idx:self.max_idx]
        return x

#
class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, x): # x=1*1000
        x = torch.from_numpy(x).float() # 把数组 array 换成张量 sensor ，且二者共享内存（修改一个另一个也会改变）
        return x


### TRANSFORMS ###


def spectral_dataloader(X_fn, y_fn, idxs=None, batch_size=10, shuffle=True,
    num_workers=4, min_idx=None, max_idx=None, sampler=None):
    """
    Returns a DataLoader with spectral data.
    """
    transform_list = []
    if min_idx is not None and max_idx is not None: # 截取
        transform_list.append(GetInterval(min_idx, max_idx)) # 截取
    transform_list.append(ToFloatTensor()) # 最后 数据 转换成 Tensor类型 float类型
    transform = transforms.Compose(transform_list) # Compose 用于串联 transform_list 中的多个操作
    dataset = SpectralDataset(X_fn, y_fn, idxs=idxs, transform=transform) # 构建一个数据集对象 idxs是数据长度
    # Dataloader是个抽象类，属于 pytorch 用于数据集合采样训练等
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, sampler=sampler)
    return dataloader



def spectral_dataloaders(X_fn, y_fn, n_train=None, p_train=0.8, p_val=0.1,
    n_test=None, batch_size=10, shuffle=True, num_workers=4, min_idx=None,
    max_idx=None):
    """
    Returns train, val, and test DataLoaders by splitting the dataset randomly.
    Can also take X_fn and y_fn as numpy arrays.
    """
    if type(y_fn) == str:
        idxs = np.arange(len(np.load(y_fn)))
    else:
        idxs = np.arange(len(y_fn))
    np.random.shuffle(idxs) # 返回一个随机序列 类似于洗牌
    if n_train is None: n_train = int(p_train * len(idxs)) # 一部分作训练集
    n_val = int(p_val * n_train) # 训练集中一部分作为验证集
    val_idxs, train_idxs = idxs[:n_val], idxs[n_val:n_train]
    if n_test is None: test_idxs = idxs[n_train:] # 剩下的取部分是测试集
    else: test_idxs = idxs[n_train:n_train+n_test]

    # 分别对三个集合的数据 处理，得到三个 loader
    trainloader = spectral_dataloader(X_fn, y_fn, train_idxs,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        min_idx=min_idx, max_idx=max_idx)
    valloader = spectral_dataloader(X_fn, y_fn, val_idxs,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        min_idx=min_idx, max_idx=max_idx)
    testloader = spectral_dataloader(X_fn, y_fn, test_idxs,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        min_idx=min_idx, max_idx=max_idx)

    return (trainloader, valloader, testloader)