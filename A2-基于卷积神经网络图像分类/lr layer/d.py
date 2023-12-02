import matplotlib.pyplot as plt
import os

# 定义文件名后缀
file_suffix = '.txt'
arrays_dict = {}  # 用于存储数组的字典

# 获取当前目录下所有以指定后缀结尾的文件
files = [f for f in os.listdir('.') if f.endswith(file_suffix)]

# 读取每个文件的内容并将数字存入数组
for file in files:
    with open(file, 'r') as f:
        content = f.read()
        # 假设文件内容是以逗号分隔的数字
        numbers = [float(num) for num in content.split(',')]
        # 将数组与文件名关联存入字典
        arrays_dict[file] = numbers

# 打印结果
for file, array in arrays_dict.items():
    print(f'Array in {file}: {array}')



plt.plot(range(1, 11), arrays_dict['9-16-0.01-eval-acc.txt'], color='b', linewidth=0.5)
plt.plot(range(1, 11), arrays_dict['16-16-0.01-eval-acc.txt'], color='g', linewidth=0.5)
plt.plot(range(1, 11), arrays_dict['16-25-0.01-eval-acc.txt'], color='y', linewidth=0.5)
plt.plot(range(1, 11), arrays_dict['9-16-0.05-eval-acc.txt'], color='r', linestyle='--', linewidth=0.5)
plt.plot(range(1, 11), arrays_dict['16-16-0.05-eval-acc.txt'], color='purple', linestyle='--', linewidth=0.5)
plt.plot(range(1, 11), arrays_dict['16-25-0.05-eval-acc.txt'], color='pink', linestyle='--', linewidth=0.5)
plt.plot(range(1, 11), arrays_dict['9-16-0.005-eval-acc.txt'], color='black', marker='o', markersize=0.5, linewidth=0.5)
plt.plot(range(1, 11), arrays_dict['16-16-0.005-eval-acc.txt'], color='orange', marker='o', markersize=0.5, linewidth=0.5)
plt.plot(range(1, 11), arrays_dict['16-25-0.005-eval-acc.txt'], color='brown', marker='o', markersize=0.5, linewidth=0.5)
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.savefig('t.png')
plt.clf()
