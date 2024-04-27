import matplotlib.pyplot as plt
import os
import numpy as np
import random
if __name__ == '__main__':
    
    dirpath = './data/0426/0426USB1'
    filepath = './USB1.txt'
    label = np.loadtxt(filepath)
    filepaths = []
    for root,dirs,files in os.walk(dirpath):
        for file in files:
            if file.endswith('.txt'):
                filepaths.append(os.path.join(root,file))  
     
    
    rows = np.nonzero(label)
    for i in range(0,len(rows[0])):
        num = rows[0][i]
        read_path = filepaths[num]
        data = np.loadtxt(read_path, delimiter=' ')
        A = data[:, 0]
        B = data[:, 1]
        C = data[:, 2]
        plt.plot(A, label='X')
        plt.plot(B, label='Y')
        plt.plot(C, label='Z')
        plt.legend()
        plt.title(f"第几类: {label[num]}")
        plt.show()
    
    print('here')

            
            
    
        