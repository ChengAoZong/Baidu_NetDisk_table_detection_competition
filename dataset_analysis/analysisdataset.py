# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import os
import cv2
import numpy as np

from matplotlib import pyplot as plt

def xyxy2xywh(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2 * dw
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return (x, y, w, h)  # 返回的都是标准化后的值

# 统计图片分辨率的分布
# 统计anchor的分布
# 统计一张图中有多少个目标的分布
# 统计各个种类的分布

resolution_set = dict()
anchor_set = dict()
numbers_per_image = dict()
class_numbers = dict() # 本比赛中只有1类 所以不做统计

"""
if __name__ == '__main__':
    #filename = "/home/zca/Downloads/train/annos.txt"
    filename = "D:\\competion\\detection\\train\\annos.txt"
    txt_out_path = "D:\\competion\\detection\\train\\labels"
    with open(filename, 'r') as f:
        json_data = json.load(f)
    #json data 存放的是最外层对像，其内有很多字典
    for item in json_data.keys():# item 表示每张图片

        res = []
        classes = 0
        img_path = "D:\competion\\detection\\train\\images\\"+item
        img = cv2.imread(img_path,1)
        h,w,chanels = img.shape
        key_resolution = (h,w)
        if key_resolution not in resolution_set.keys():
            resolution_set[key_resolution] = 1
        else:
            resolution_set[key_resolution] += 1

        numbers_of_object = 0
        for table in json_data[item]:# table表示每个框
            numbers_of_object += 1
            x1 = table['box'][0]
            y1 = table['box'][1]
            x2 = table['box'][2]
            y2 = table['box'][3]
            xywh = xyxy2xywh((w,h),[x1,y1,x2,y2])
            anchor = (xywh[2],xywh[3])
            if anchor not in anchor_set.keys():
                anchor_set[anchor] = 1
            else:
                anchor_set[anchor] += 1

        if numbers_of_object not in numbers_per_image.keys():
            numbers_per_image[numbers_of_object] = 1
        else:
            numbers_per_image[numbers_of_object] += 1

with open('dataset_analysis/resolution_distribute.txt', 'w') as f:
    f.write(str(resolution_set))

with open('dataset_analysis/anchor_distribute.txt', 'w') as f:
    f.write(str(anchor_set))

with open('dataset_analysis/numbers_per_image.txt', 'w') as f:
    f.write(str(numbers_per_image))
"""


with open('dataset_analysis/resolution_distribute.txt', 'r') as f:
    a = f.read()
resolution_set = eval(a)


with open('dataset_analysis/anchor_distribute.txt', 'r') as f:
    a = f.read()
anchor_set = eval(a)


with open('dataset_analysis/numbers_per_image.txt', 'r') as f:
    a = f.read()
numbers_per_image = eval(a)


# 2.创建一张figure
fig = plt.figure(1)
# 3. 设置颜色 color 值【可选参数，即可填可不填】，方式有几种
# colors = np.random.rand(n) # 随机产生10个0~1之间的颜色值，或者
colors = ['r', 'g', 'y', 'b', 'r', 'c', 'g', 'b', 'k', 'm']  # 可设置随机数取
# 4. 设置点的面积大小 area 值 【可选参数】

#area = 20 * np.arange(1, n + 1)
n = len(resolution_set)
area = [resolution_set[item] for item in resolution_set.keys()]
x = [item[0] for item in resolution_set.keys()]
y = [item[1] for item in resolution_set.keys()]

# 5. 设置点的边界线宽度 【可选参数】
# widths = np.arange(n)  # 0-9的数字
widths = 1
# 6. 正式绘制散点图：scatter
plt.scatter(x, y, s=area, c=colors[1], linewidths=widths, alpha=0.5, marker='o')
# 7. 设置轴标签：xlabel、ylabel
# 设置X轴标签
plt.xlabel('image width')
# 设置Y轴标签
plt.ylabel('image height')
# 8. 设置图标题：title
plt.title('resolution distribute')
# 9. 设置轴的上下限显示值：xlim、ylim
# 设置横轴的上下限值
# plt.xlim(-0.5, 2.5)
# 设置纵轴的上下限值
# plt.ylim(-0.5, 2.5)
# 10. 设置轴的刻度值：xticks、yticks
# 设置横轴精准刻度
plt.xticks(np.arange(np.min(x) - (np.max(x)-np.min(x))/5, np.max(x) + (np.max(x)-np.min(x))/5, step=(np.max(x)-np.min(x))/5))
# 设置纵轴精准刻度
plt.yticks(np.arange(np.min(y) - (np.max(y)-np.min(y))/5, np.max(y) + (np.max(y)-np.min(y))/5, step=(np.max(y)-np.min(y))/5))
# 也可按照xlim和ylim来设置
# 设置横轴精准刻度
# plt.xticks(np.arange(-0.5, 2.5, step=0.5))
# 设置纵轴精准刻度
# plt.yticks(np.arange(-0.5, 2.5, step=0.5))

# 11. 在图中某些点上（位置）显示标签：annotate
# plt.annotate("(" + str(round(x[2], 2)) + ", " + str(round(y[2], 2)) + ")", xy=(x[2], y[2]), fontsize=10, xycoords='data')# 或者
# plt.annotate("({0},{1})".format(round(x[2], 2), round(y[2], 2)), xy=(x[2], y[2]), fontsize=10, xycoords='data')
# xycoords='data' 以data值为基准
# 设置字体大小为 10
# 12. 在图中某些位置显示文本：text
#plt.text(round(x[6], 2), round(y[6], 2), "good point", fontdict={'size': 10, 'color': 'red'})  # fontdict设置文本字体
# Add text to the axes.
# 13. 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 14. 设置legend，【注意，'绘图测试’：一定要是可迭代格式，例如元组或者列表，要不然只会显示第一个字符，也就是legend会显示不全】
# plt.legend(['resolution'], loc=2, fontsize=10)
# plt.legend(['绘图测试'], loc='upper left', markerscale = 0.5, fontsize = 10) #这个也可
# markerscale：The relative size of legend markers compared with the originally drawn ones.
# 15. 保存图片 savefig

plt.savefig('dataset_analysis/resolution_distribute.png', dpi=200, bbox_inches='tight', transparent=False)

# dpi: The resolution in dots per inch，设置分辨率，用于改变清晰度
# If *True*, the axes patches will all be transparent
# 16. 显示图片 show
plt.show()



fig = plt.figure(1)
n = len(anchor_set)
area = [anchor_set[item] for item in anchor_set.keys()]
x = [item[0] for item in anchor_set.keys()]
y = [item[1] for item in anchor_set.keys()]
widths = 1
plt.scatter(x, y, s=area, c=colors[1], linewidths=widths, alpha=0.5, marker='o')
plt.xlabel('anchor width')
plt.ylabel('anchor height')
plt.title('anchor distribute')
plt.xticks(np.arange(np.min(x) - (np.max(x)-np.min(x))/5, np.max(x) + (np.max(x)-np.min(x))/5, step=(np.max(x)-np.min(x))/5))
plt.yticks(np.arange(np.min(y) - (np.max(y)-np.min(y))/5, np.max(y) + (np.max(y)-np.min(y))/5, step=(np.max(y)-np.min(y))/5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.legend(['anchor'], loc=2, fontsize=10)
plt.savefig('dataset_analysis/anchor_distribute.png', dpi=200, bbox_inches='tight', transparent=False)
plt.show()


fig = plt.figure(1)
n = len(numbers_per_image)
area = [numbers_per_image[item] for item in numbers_per_image.keys()]
x = [item for item in numbers_per_image.keys()]
order = np.argsort(x)
x = [x[item] for item in order]
y = [numbers_per_image[item] for item in numbers_per_image.keys()]
y = [y[item] for item in order]
plt.bar(x, y, width=0.8, bottom=None, align='center')
plt.xlabel('objecet number')
plt.ylabel('images number')
plt.title('number of object per image')
plt.xticks(np.arange(0, 10, step=1))
plt.yticks(np.arange(0, 10000, step=1000))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.legend(['number of per image'], loc=2, fontsize=10)
plt.savefig('dataset_analysis/number_of_per_image.png', dpi=200, bbox_inches='tight', transparent=False)
plt.show()