# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%% 线图
# 描绘两个连续变量的关系
# 绘制Google股价
path = './示例代码/Chapter02/GOOG.csv'
stock = pd.read_csv(path, header = None, delimiter = ',')
stock.columns = ['date', 'price']
stock['date'] = pd.to_datetime(stock['date'], format = '%d-%m-%Y')
indexed_stock = stock.set_index('date')
ts = indexed_stock['price']

plt.plot(ts)
plt.xticks(rotation = 20)
plt.show()

#%% 条形图
# 针对类别变量
month_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
month_str = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', \
            'Oct', 'Nov', 'Dec']
units_sold = [500, 600, 750, 900, 1100, 1050, 1000, 950, 800, 700, 550, 450]

# 主要是以后多图布局需要
# 使用面向对象的方法，先产生一个Axes对象，再对此对象进行操作，实现绘图
fig, ax = plt.subplots()
plt.xticks(ticks = month_num, labels = month_str, rotation = 20)
plot = ax.bar(month_num, units_sold)

# 添加数据标签
for rect in plot:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, 1.002 * height, \
            '%d' % int(height), ha = 'center', va = 'bottom')

plt.show()

"""
# 横着的
month_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
month_str = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', \
            'Oct', 'Nov', 'Dec']
units_sold = [500, 600, 750, 900, 1100, 1050, 1000, 950, 800, 700, 550, 450]

fig, ax = plt.subplots()
plt.yticks(ticks = month_num, labels = month_str, rotation = 20)
plot = plt.barh(month_num, units_sold)

plt.show()
"""

#%% 点图
# 观察两个变量是否有相关性
plt.figure(figsize = (10, 6))

age_weight = pd.read_excel('./示例代码/Chapter02/scatter_ex.xlsx', 'age_weight')
x = age_weight['age']
y = age_weight['weight']

plt.scatter(x, y)
# 对于大数据集， plot()速度更快
plt.xlabel('Age')
plt.ylabel('Weight')
plt.show()

# 观察簇
iris = pd.read_csv('./示例代码/Chapter02/iris_dataset.csv', delimiter = ',')

# 将类别映射到0，1，2
iris['species'] = iris['species'].map({'setosa': 0, 'versicolor': 1, \
    'virginica': 2})

plt.scatter(iris.petal_length, iris.petal_width, c = iris.species)  
# c参数要求是数值
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.show()

#%% 气泡图
plt.scatter(iris.petal_length, iris.petal_width, \
            s = 50 * iris.petal_length * iris.petal_width, c = iris.species, \
            alpha = 0.3)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.show()

#%% 堆叠图
x =np.array([1, 2, 3, 4, 5, 6], dtype = np.int32)
Apr = [5, 7, 6, 8, 7, 9]
May = [0, 4, 3, 7, 8, 9]
June = [6, 7, 4, 5, 6, 8]
labels = ["April", "May", "June"]

fig, ax = plt.subplots()
ax.stackplot(x, Apr, May, June, labels = labels)
ax.legend(loc = 2)

plt.xlabel("defect reason code")
plt.ylabel("number of defects")
plt.title("Product Defects - Q1 FY2019")

plt.show()

#%% 饼图
labels = ["SciFi", "Drama", "Thriller", "Comedy", "Action", "Romance"]
sizes = [5, 15, 10, 20, 40, 10]  # 百分制
explode = (0, 0, 0, 0, 0.1, 0)  # 突出显示某一部分
plt.pie(sizes, labels = labels, explode = explode, autopct = "%1.1f%%", \
        startangle = 90)  
# 后一个%用来格式化输出自己本身
# 逆时针绘制
plt.show()

#%% 表


#%% 极坐标图


#%% 直方图
# 连续变量的分布
# y轴可以是频数，也可以是频率
grp_exp = np.array([12,  15,  13,  20,  19,  20,  11,  19,  11,  12,  19,  13,\
                    12,  10,  6,  19,  3,  1,  1,  0,  4,  4,  6,  5,  3,  7,\
                    12,  7,  9,  8,  12,  11,  11,  18,  19,  18,  19,  3,  6,\
                    5,  6,  9,  11,  10,  14,  14,  16,  17,  17,  19,  0,  2,\
                    0,  3,  1,  4,  6,  6,  8,  7,  7,  6,  7,  11,  11,  10,\
                    11,  10,  13,  13,  15,  18,  20,  19,  1,  10,  8,  16,\
                    19,  19,  17,  16,  11,  1,  10,  13,  15,  3,  8,  6,  9,\
                    10,  15,  19,  2,  4,  5,  6,  9,  11,  10,  9,  10,  9,\
                    15,  16,  18,  13])
# 等分21组
nbins = 21
n, bins, patches = plt.hist(grp_exp, bins = nbins)
# n 每组的数目
# bins 每组的开始数
# patches 每一组作为对象的list

plt.xlabel("Experience in years")
plt.ylabel("Frequency")
plt.title("Distribution of Experience in a Lateral Training Program")
plt.show()

"""
# y轴显示概率
n, bins, patches = plt.hist(grp_exp, bins = nbins, density = 1)
# 绘制正态曲线
mu = grp_exp.mean()
sigma = grp_exp.std()
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
plt.plot(bins, y, '--')
plt.show()
"""

#%% 箱线图





#%% 小提琴图


#%% heatmap


#%% Hinton图


#%% 轮廓图

