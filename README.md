## 写在前面
天气晴朗万物可爱，希望通过这篇文章对大家学习GAN有所帮助。话不多说，我们开始吧！

## 先来看看效果吧
这是pycharm跑出来的效果，看起来挺不错的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210713175706274.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)
这是做的一个app
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210713180500245.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70)

> APP界面参考了微信小程序的AI卡通秀

## 大家最想要的

本项目CSDN博客
[https://blog.csdn.net/iiiiiiimp/article/details/118701276](https://blog.csdn.net/iiiiiiimp/article/details/118701276)


如何运行？
 1. 在百度云上下载训练好的模型
 链接：[https://pan.baidu.com/s/1TLkQCcuxR9KUAKeBo5Y_rw](https://pan.baidu.com/s/1TLkQCcuxR9KUAKeBo5Y_rw) 
 提取码：iimp 
 2. 将下载好的模型放在save_model文件夹之下
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210713202019839.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)

 3. 将你要转换的人脸图像如nini.png放入dataset/img中，将要融合的背景图像如yourname2.jpeg放入dataset/back_ground中
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210713205924634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)

 4. 参数设置
 找到**mian.py**文件中的parse_opt()方法，直接改default里面的数值就好
 **--img-name**填写你放在dataset/img中人脸图片的名字如nini.png
 **--background-name**填写你放在dataset/back_ground中背景图片的名字如yourname2.jpeg
 **--fusion-method**为融合方式，有pre_fusion（前景融合）和back_fusion（背景融合）两种，默认为pre_fusion（前景融合）
 **--text-content**为你要在图片上写的字如'nini'，默认啥也不写
 **--text-scale**为图片上写的字的大小，默认为70
 **--text-location**为图片上写的字的位置，默认为(220,30)
 **--shear-rate**为人脸剪切的比例大小，数值越大，剪切的就越大，默认为0.8
 **--segment-model**为选择人脸分割所使用的模型，有U2net和FCN两种，各有优劣，默认为U2net
 **--migration-method**为选择卡通图像风格迁移模型，有Photo2cartoon、U-GAT-IT、Pix2pix三种，强烈建议使用Photo2cartoon，效果最好。
 
 	**一图胜千言**(参数控制的地方)
  
 	![在这里插入图片描述](https://img-blog.csdnimg.cn/20210714091342451.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)
 
 5. 运行main.py文件
最后结果在dataset/pre_fuse_output（前景融合）或dataset/back_fuse_output（背景融合）,dataset其余文件是保存中间结果。
 
## 项目原理简介

简单介绍一下项目的运行流程，主要分为图像预处理和卡通图像风格迁移两部分

> 主要参考了[https://github.com/minivision-ai/photo2cartoon](https://github.com/minivision-ai/photo2cartoon)

 **1. 图像预处理**
 流程图如下：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210713214029473.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)

（1）人脸关键点检测，获得人脸的68个关键点坐标。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210714092317427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)

> 检测方法[https://github.com/1adrianb/face-alignment](https://github.com/1adrianb/face-alignment)

 （2）人脸校正，通过68个人脸关键点的第37、46两个点（即眼角的两个点）的坐标结合仿射变换，将倾斜的人脸转正。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210714093605119.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)

> 仿射变换这篇文章讲的不错[https://blog.csdn.net/liuweiyuxiang/article/details/82799999](https://blog.csdn.net/liuweiyuxiang/article/details/82799999)

（3）人脸截取，根据68个关键点中最左边，最右边，最上边，最下边的四个点的坐标位置按一定比例框出一个正方形来截取出人脸。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210714094014448.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)
（4）人脸分割，使用FCN或U2net语义分割模型将人脸截取出来。然后用原图与截取后的图像相乘就能去除掉背景。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210714094752860.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)

> FCN模型[https://github.com/minivision-ai/photo2cartoon](https://github.com/minivision-ai/photo2cartoon)
> FCN论文[https://arxiv.org/abs/1411.4038](https://arxiv.org/abs/1411.4038)
> U2net模型[https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
> U2net论文[https://arxiv.org/pdf/2005.09007.pdf](https://arxiv.org/pdf/2005.09007.pdf)

 **2. 人像卡通化**

（1）将去除背景后的人像送入卡通风格迁移模型进行风格迁移，我分别使用了Photo2Cartoon、U-GAT-IT、Pix2pix三模型。经测试Photo2Cartoon是效果最好的。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210714095905105.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)

> Photo2Catoon模型[https://github.com/minivision-ai/photo2cartoon](https://github.com/minivision-ai/photo2cartoon)
> U-GAT-IT模型[https://github.com/znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch)
> U-GAT-IT论文[https://arxiv.org/abs/1907.10830](https://arxiv.org/abs/1907.10830) 
> 飞桨U-GAT-IT论文复现也讲的很好 [https://aistudio.baidu.com/aistudio/education/group/info/1340](https://aistudio.baidu.com/aistudio/education/group/info/1340)
> Pix2pix模型[https://phillipi.github.io/pix2pix/](https://phillipi.github.io/pix2pix/)
> Pix2pix论文[https://arxiv.org/abs/1611.07004](https://arxiv.org/abs/1611.07004)
> 飞桨PaddleGAN郝强老师讲得也很好人也帅气[https://aistudio.baidu.com/aistudio/education/group/info/16651](https://aistudio.baidu.com/aistudio/education/group/info/16651)

模型训练用的数据集
链接：[https://pan.baidu.com/s/1NmLHCBo0pLfdKhUPCjR84Q](https://pan.baidu.com/s/1NmLHCBo0pLfdKhUPCjR84Q) 
提取码：iimp 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210714102531796.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)

> 蟹蟹飞桨提供的数据[https://aistudio.baidu.com/aistudio/education/group/info/16651](https://aistudio.baidu.com/aistudio/education/group/info/16651)

（2）将卡通图像与其他图像融合，看起来更阔爱(●'◡'●)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210714101559341.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2lpaWlpaWltcA==,size_16,color_FFFFFF,t_70#pic_center)

> 背景融合参考了[https://github.com/leijue222/portrait-matting-unet-flask](https://github.com/leijue222/portrait-matting-unet-flask)

## (2021/11/16更)本项目C++和Pytorch的OnnxRuntime使用方法
见博主的另一篇博客
[https://blog.csdn.net/iiiiiiimp/article/details/120621682](https://blog.csdn.net/iiiiiiimp/article/details/120621682)

## (2022/1/3更)APP的代码
写在这里
[https://blog.csdn.net/iiiiiiimp/article/details/122384622](https://blog.csdn.net/iiiiiiimp/article/details/122384622)

 <img src="https://img-blog.csdnimg.cn/ebe30ae3e19740018c6e8a58638c0d93.gif" width="200"> <img src="https://img-blog.csdnimg.cn/d095c6d03d2c4951ad039b7fd90714b3.gif" width="200"> <img src="https://img-blog.csdnimg.cn/cd66cf9f32ba4297a3ad241a859387bc.gif" width="200">

## 写在后面
博主今年大四毕业单身狗，做这个项目的原因是想送给自己喜欢的一位姑娘一副她本人的漫画图像，顺带完成一下毕业设计。**若有写的不好的地方还望多多包含**~今年天临3年，毕业依旧很难，嘤嘤嘤。希望对大家学习GAN有所帮助！

<img width="160" alt="zyy" src="https://user-images.githubusercontent.com/47819608/147376046-d067ac63-23dd-4bb8-a167-3c14d7b9ca57.png">

