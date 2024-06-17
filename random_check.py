import matplotlib.pyplot as plt
import os
import numpy as np
import random
if __name__ == '__main__':
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
    path_lists = [dirpath0_0,dirpath0_1,dirpath0_2,dirpath0_3,dirpath0_4,dirpath0_5,dirpath0_6,dirpath0_7,dirpath0_8,dirpath0_9,dirpath0_10,dirpath0_11,dirpath0_12]
    path_lists_tmp = path_lists[4:5]
    count = 0
    for i in range(0,20):
        for filepath in path_lists_tmp:
            for root, dirs, files in os.walk(filepath):
                
                len_files = len(files)
                random_int = random.randint(1, len_files)
                read_path = os.path.join(root, files[random_int])
                data = np.loadtxt(read_path, delimiter=' ')
                A = data[:, 0]
                B = data[:, 1]
                C = data[:, 2]
                plt.plot(A, label='X')
                plt.plot(B, label='Y')
                plt.plot(C, label='Z')
                plt.legend()
                plt.title(f"第几类: {count}")
            plt.show()
        count = count + 1
            
        
# 提取三列数据

            
            
    
        