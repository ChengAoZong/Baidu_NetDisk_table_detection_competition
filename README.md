# 百度网盘AI大赛——表格检测

### 项目说明

该比赛需要检测图片中的表格以及预测表格的四个角点，表格检测的IOU需要>=0.9，角点的误差需要小于20个像素，本项目使用的方案是一个二阶段的方法，第一阶段使用修改检测头的yolov5完成表格的检测以及关键点的初步预测，第二阶段在第一阶段预测的关键点周围裁剪图片，然后使用热力图的方式进行第二阶段的关键点预测。



### install

```python
git clone https://github.com/ChengAoZong/Baidu_NetDisk_table_detection_competition.git

cd Baidu_NetDisk_table_detection_competition

pip install -r requirements
```



### USE

* heatmap: 用于第二阶段关键点预测的网络
* submit: 比赛提交的推理代码
* dataset_analysis: 对数据集进行可视化分析代码
* divide_dataset:用于对数据集的划分
* preprocess:用于对数据的前处理
* getscore:用于本地测试得分
* train.py训练脚本

在train.py中配置好参数，例如训练的epoch、device、batchsize以及网络模型等参数，并且选择好配置文件，在配置文件中需要确定数据集的路径以及检测的类别等信息（参考yolov5训练自定义数据集），即可使用：

```python
python train.py
```

进行训练第一阶段的yolov5表格检测及关键点回归模型。

（提交checkpoint的训练参数：img_size:1280 epoch:300 数据增强：旋转数据增强 batchsize：8）

在release-v1.0中提供了checkpoint的pytorch模型以及转化得到的paddle模型。

当自己训练得到pytorch模型后可以使用下面的方式转化为paddle的模型：

```
python export.py --grid
x2paddle --framework=onnx --model=runs\train\expxx\weights\last.onnx --save_dir=runs\train\expxx\weights
```

首先将pytorch模型转化为onnx模型，再将onnx模型转化为paddle模型。



### 训练过程中所做的尝试与改进

* 训练尺寸：一开始使用的尺寸为640，提交得分大概只有0.1，可能是原图比较大，当缩小训练尺寸后原图的细节信息变得不可见，所以效果不是很好，因此增大训练的图像尺寸有利于提高精度，但是增大图片尺寸会导致训练和推理变慢，为了平衡速度和精度，最终选择的训练尺寸为1280.

* 损失系数:yolov5的损失原本由三部分组成，再加上关键点的损失总共由四部分组成，由于关键点的nms是根据框的IOU来做的，因此若想检测准确关键点则需要检测准确目标框，因此需要平衡好这两部分的损失，一开始box和关键点的损失系数都为0.05，但是这样模型很难收敛，后续选择的box损失系数为0.05，关键点的系数为0.005
* 数据增强：一开始使用了随机缩放、平移、马赛克、水平垂直翻转等数据增强的方式，由于数据集中的图片大多数只有一张图片，基本上没有小目标，因此去除了马赛克和随机缩放以及平移的数据增强，由于水平和垂直翻转后图案和文字变为镜像，不太好说明几个关键点的顺序，因此去除了水平和垂直翻转的数据增强，由于训练集中有各个角度的表格，因此增加了旋转的数据增强。

