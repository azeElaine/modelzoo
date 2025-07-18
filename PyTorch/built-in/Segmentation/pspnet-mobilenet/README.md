## PSPnet：Pyramid Scene Parsing Network语义分割模型在Pytorch当中的实现
---
PSPNet（Pyramid Scene Parsing Network）是一种用于语义分割的深度学习模型。它通过引入金字塔池化模块（PPM），能够从不同的尺度提取图像的上下文信息，从而提高了图像分割的精度。

快速开始 使用本模型执行训练的主要流程如下：
基础环境安装：介绍训练前需要完成的基础环境检查和安装。 获取数据集：介绍如何获取训练所需的数据集。 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。 启动训练：介绍如何运行训练。 
2.1 基础环境安装 请参考基础环境安装章节，完成训练前的基础环境检查和安装。
2.2 准备数据集 
2.2.1 获取数据集
/mnt/nvme1/dnn/houjx/project/segformer-pytorch-master/VOCdevkit
2.3 构建Docker环境 使用Dockerfile，创建运行模型训练所需的Docker环境。
例如：
docker pull jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0   jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.0b0-torch_sdaa2.1.0b0

docker run -itd --name={name} --net=host -v /mnt/nvme1/dnn/houjx:/mnt/nvme1/dnn/houjx -v /mnt/:/mnt -v /hpe_share/:/hpe_share -p 22 -p 8080 -p 8888 --device=/dev/tcaicard0 --device=/dev/tcaicard1 --device=/dev/tcaicard2 --device=/dev/tcaicard3 --device=/dev/tcaicard4 --device=/dev/tcaicard5 --device=/dev/tcaicard6 --device=/dev/tcaicard7 --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 300g jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0 /bin/bash

docker exec -it {name} bash

### 目录
1. [仓库更新 Top News](#仓库更新)
2. [相关仓库 Related code](#相关仓库)
3. [性能情况 Performance](#性能情况)
4. [所需环境 Environment](#所需环境)
5. [文件下载 Download](#文件下载)
6. [训练步骤 How2train](#训练步骤)
7. [预测步骤 How2predict](#预测步骤)
8. [评估步骤 miou](#评估步骤)
9. [参考资料 Reference](#Reference)

## Top News
**`2022-04`**:**支持多GPU训练。**  

**`2022-03`**:**进行大幅度更新、支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/pspnet-pytorch/tree/bilibili

**`2020-08`**:**创建仓库、支持多backbone、支持数据miou评估、标注数据处理、大量注释等。**

## 相关仓库
| 模型 | 路径 |
| :----- | :----- |
Unet | https://github.com/bubbliiiing/unet-pytorch  
PSPnet | https://github.com/bubbliiiing/pspnet-pytorch
deeplabv3+ | https://github.com/bubbliiiing/deeplabv3-plus-pytorch
hrnet | https://github.com/bubbliiiing/hrnet-pytorch

### 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| VOC12+SBD | [pspnet_mobilenetv2.pth](https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/pspnet_mobilenetv2.pth) | VOC-Val12 | 473x473| 68.59 | 
| VOC12+SBD | [pspnet_resnet50.pth](https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/pspnet_resnet50.pth) | VOC-Val12 | 473x473| 81.44 | 

### 所需环境
torch==1.2.0  

### 文件下载
训练所需的pspnet_mobilenetv2.pth和pspnet_resnet50.pth可在百度网盘中下载。    
链接: https://pan.baidu.com/s/1Ecz-l6lFcf6HmeX_pLCXZw 提取码: wps9    

VOC拓展数据集的百度网盘如下：  
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng 提取码: 44mk   

### 训练步骤
#### a、训练voc数据集
1、将我提供的voc数据集放入VOCdevkit中（无需运行voc_annotation.py）。  
2、在train.py中设置对应参数，默认参数已经对应voc数据集所需要的参数了，所以只要修改backbone和model_path即可。  
3、运行train.py进行训练。  

#### b、训练自己的数据集
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4、在训练前利用voc_annotation.py文件生成对应的txt。    
5、在train.py文件夹下面，选择自己要使用的主干模型和下采样因子。本文提供的主干模型有mobilenet和resnet50。下采样因子可以在8和16中选择。需要注意的是，预训练模型需要和主干模型相对应。   
6、注意修改train.py的num_classes为分类个数+1。    
7、运行train.py即可开始训练。  

### 预测步骤
#### a、使用预训练权重
1. 下载完库后解压，如果想用backbone为mobilenet的进行预测，直接运行predict.py就可以了；如果想要利用backbone为resnet50的进行预测，在百度网盘下载pspnet_resnet50.pth，放入model_data，修改pspnet.py的backbone和model_path之后再运行predict.py，输入。  
```python
img/street.jpg
```  
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。    
#### b、使用自己训练的权重
1. 按照训练步骤训练。    
2. 在pspnet.py文件里面，在如下部分修改model_path和backbone使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，backbone是所使用的主干特征提取网络**。    
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
    #-------------------------------------------------------------------#
    "model_path"        : 'model_data/pspnet_mobilenetv2.pth',
    #----------------------------------------#
    #   所需要区分的类的个数+1
    #----------------------------------------#
    "num_classes"       : 21,
    #----------------------------------------#
    #   所使用的的主干网络：mobilenet、resnet50
    #----------------------------------------#
    "backbone"          : "mobilenet",
    #----------------------------------------#
    #   输入图片的大小
    #----------------------------------------#
    "input_shape"       : [473, 473],
    #----------------------------------------#
    #   下采样的倍数，一般可选的为8和16
    #   与训练时设置的一样即可
    #----------------------------------------#
    "downsample_factor" : 16,
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"             : True,
    #--------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py，输入    
```python
img/street.jpg
```   
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。    

### 评估步骤
1、设置get_miou.py里面的num_classes为预测的类的数量加1。  
2、设置get_miou.py里面的name_classes为需要去区分的类别。  
3、运行get_miou.py即可获得miou大小。  

### Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus
