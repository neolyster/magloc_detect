import os
import re
import numpy as np
import torch
# def extract_numbers(s):
#     return [int(num) for num in re.findall(r'\d+', s)]
# def read_folder_recursively(folder_path):
#     for root, dirs, files in os.walk(folder_path):
#         for dir in dirs:
#             dir_path = os.path.join(root, dir)
#             for root_2, dirs_2, files_2 in os.walk(dir_path):
#                 for file in files_2:
#                     print(extract_numbers(dir)[0])
#                     print(os.path.join(root_2, file))

# print(torch.cuda.is_available())
# dd = np.zeros([12])
# folder_path = "./data/0418/0418data0"
# read_folder_recursively(folder_path)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
shifted_matrix = np.roll(matrix, -1, axis=1)

print(shifted_matrix)
shifted_matrix[0,2] = 10
print(shifted_matrix)