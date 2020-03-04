# -*- coding: utf-8 -*-

"""
backend
    图形展示层，后端
    决定图形输出位置和方式
artist
    内部实现创建图形
    一张图中所有元素都可视为artist
    面向对象的API，更灵活
scripting
    API，用来创建图形
    接口, pyplot API
"""

"""
Figure
    - Axes  可理解为“子图”
        -- title
        -- x-label
        -- y-label
        -- axis
        -- legend
        -- tick, ticklabel 刻度，刻度标签
        -- spine 可理解为边框
        -- grid
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 非交互模式
plt.ioff()
mpl.is_interactive()
plt.plot([1.5, 3.0])
plt.title("Non Interactive Mode")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

#%% 从外部文件获取数据
# 默认输入数据应是list, numpy array, pandas dataframe格式的
path1 = "./示例代码/Chapter01/test.txt"
raw_data = np.loadtxt(path1, delimiter = ',')
data = raw_data.reshape(5, 2)
x = data[:, 0]
y = data[:, 1]

path2 = "./示例代码/Chapter01/test.csv"
# unpack如果为True，将分列读取
x2, y2 = np.loadtxt(path2, unpack = True, usecols = (0, 1), delimiter = ',')

path3 = "./示例代码/Chapter01/test.xlsx"
df = pd.read_excel(path3, "sheet", header = None)
data = np.array(df)
x3, y3 = data[:, 0], data[:, 1]

plt.plot(x, y)
plt.show()

#%% 改变或重置绘图环境设置
"""
matplotlibrc文件用来存储绘图参数设置
"""
print(mpl.matplotlib_fname())  # 文件位置
print(mpl.rcParams)

# 批量修改某一组的
mpl.rc('lines', linewidth = 4, linestyle = '-', marker = '*')

# 逐个修改
mpl.rcParams['lines.markersize'] = 20
mpl.rcParams['font.size'] = '15.0'

plt.plot(x, y)
plt.show()

# 恢复
mpl.rcdefaults()
plt.plot(x, y)
plt.show()
