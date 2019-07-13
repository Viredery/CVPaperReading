## 两阶段目标检测

### 一、基础框架

两阶段网络的总体流程分为五步，通过Backbone提取特征、RPN网络提取RoI、RoI池化、RCNN网络进行分类和回归以及后处理得到检测结果。

同时，由于目标在原图中的尺度差异较大，单一分辨率的特征无法很好地预测较大和较小尺度的目标，因此引入多尺度建模方法来解决不同尺度目标的预测问题。

1. 两阶段检测的总体流程

   * **[R-CNN]** Rich feature hierarchies for accurate object detection and semantic segmentation **[CVPR' 14]**

   * **[Fast R-CNN]** Fast R-CNN **[ICCV' 15]**

   * **[Faster R-CNN, RPN]** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks **[NIPS' 15]**


2. 多尺度建模方法

   * **[FPN]** Feature Pyramid Networks for Object Detection **[CVPR' 17]**   
      目前最流行的方法。输出多个不同分辨率的特征图。分辨率大的特征图检测小目标；分辨率小的特征图检测大目标

   * **[SNIP]** An Analysis of Scale Invariance in Object Detection – SNIP **[CVPR' 18]**   
      输入为图像金字塔，对于图像金字塔中的每一个图像，仅选择固定尺度范围内的目标进行训练（反向传播）。也就是说，原图像放大后仅检测小目标，原图像缩小后仅检测大目标。由于每个目标在训练时都会有几个不同的尺寸，那么总有一个尺寸是在指定的尺寸范围内。

   * **[TridentNet]** Scale-Aware Trident Networks for Object Detection **[CVPR' 19]**    
      输出多个不同感受野的特征图。以ResNet50为例，将Stage5卷积移至RCNN，将Stage4分成三个分支，不同分支中的3x3卷积的dilated分别设置为1，2和3，不同分支共享权重。同时，对于不同的分支，选择特定尺度范围内的目标进行训练（反向传播）。也就是说小目标和大目标分别在不同的分支中检测，最后三个分支的预测结果通过NMS结合起来。


### 二、骨架网络设计和改进

1. 整体结构设计

    常见的Backbone为ResNet、ResNeXt、Inception-ResNet，针对轻量化设计要求，也可以使用MobileNet等网络。这些网络在分类任务中被提出，被运用在了各种视觉任务中。同时，也有些网络针对位置敏感问题设计的网络。DetNet网络中将最后一阶段的stride从2减小到1并将dilated设置为2也是常见的针对位置敏感问题的优化方法。CVPR19中也有NAS去搜索目标检测中最优骨架网络结构的工作。    
    Hourglass和HRNet网络设计的出发点都是低分辨率的语义信息和高分辨率的位置信息的融合。     

    * **[Hourglass]** Stacked Hourglass Networks for Human Pose Estimation **[CVPR' 16]**    
       Hourglass网络类似于UNet，上采样阶段和下采样阶段对称，对应的相同分辨率大小的特征图之间的skip connection（UNet）替换为残差块（Hourglass）。CenterNet(Keypoint Triplets)的骨架网络为Hourglass。    

    * **[HRNet]** Deep High-Resolution Representation Learning for Human Pose Estimation **[CVPR' 19]**    
       该模型是通过在高分辨率特征图主网络逐渐并行加入低分辨率特征图子网络，不同网络实现多尺度融合与特征提取实现的。    

2. 模块设计

    * **[DCN]** Deformable Convolutional Networks **[ICCV' 17]**    
      **[DCNv2]** Deformable ConvNets v2: More Deformable, Better Results **[CVPR' 19]**    
       提出可变形卷积，将其替换残差块中的3x3卷积可以稳定提分。注意，一个可变形卷积实际上进行了两次卷积运算。    

    * **[NL]** Non-local Neural Networks **[CVPR' 18]**     
       在高阶语义层（如Stage4）中引入Non-local层（类似Self-Attention）。由于Non-local计算量较大，**CCNet**提出局部平均，在不下降性能的情况下减少计算量。**GCNet**将简化的Non-local和SE Block结合起来。    

    * **[WS]** Weight Standardization    
      **[GN]** Group Normalization **[ECCV' 18]**    
       基于ResNet50或者ResNet101的目标检测网络的训练中，一般一个GPU中图片的batch size为1或者2，使用BN会损害训练效果，因此一般将BN层的权重固定住。这里将卷积层的权重做标准化，BN替换成GN，适合目标检测、实例分割等batch size较小的任务的训练。另一种做法，是在多GPU训练的情况下，将BN替换成Sync BN。    


### 三、多尺度建模改进

1. 基于FPN的改进  

    * **[PANet]** Path Aggregation Network for Instance Segmentation **[CVPR' 18]**     
       FPN的信息流是自上而下，PANet在FPN的基础上再通过自下而上的路径增强，在较底层用准确的定位信号增强了整个特征分层，从而缩短了较底层和最高层特征之间的信息路径。    
    同时，在RoI Pooling阶段，不是使用一层的feature map提特征，而是结合了所有层的信息。   
    
    * **[Libra R-CNN]** Libra R-CNN: Towards Balanced Learning for Object Detection **[CVPR' 19]**        
       也是将FPN中不同层的信息进行融合，主要分为四步，rescale到同一尺度，融合（相加），计算Non-local，最后将得到的融合信息加强到不同层的feature map中   
 
    * **[NAS-FPN]** NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection **[CVPR' 19]**      
       用NAS寻找最优的FPN结构   

2. 基于SNIP的改进

    * **[SNIPER]** SNIPER: Efficient Multi-Scale Training **[NIPS' 18]**   
       SNIPER通过生成Patch的方式，加快SNIP的训练速度   

    * **[AutoFocus]** AutoFocus: Efficient Multi-Scale Inference   
       AutoFocus加快SNIP的推断速度   
    
### 四、RPN改进

* **[GA-RPN]** Region Proposal by Guided Anchoring **[CVPR' 19]**      
   大多数解决方案，都是根据不同的数据集，去设计每个位置的预选框（Anchor）的尺度和长宽比（即调参，调整先验假设）。GA-RPN是训练过程中指导预选框的生成。对每层特征图的每一个位置，通过训练去预测这个位置对应的预选框的高和宽。

### 五、RoI池化方式

* **[Mask R-CNN]** Mask R-CNN **[ICCV' 17]**   
   提出RoIAlign，将最大池化过程替换成了双线性插值。此外加入分割损失后，对回归任务有效果上的提升。   

* **[IoUNet]** Acquisition of Localization Confidence for Accurate Object Detection **[ECCV' 18]**   
   提出PrRoIPooling，将RoIAlign上选择一个坐标点进行双线性插值，改为积分的方式，对全局进行插值操作。   

### 六、RCNN改进

1. 针对分类和检测的根本矛盾问题

   图像分类要求图像具有平移不变性，而目标检测则要求图像具有位置敏感性。
   
   * **[R-FCN]** R-FCN: Object Detection via Region-based Fully Convolutional Networks **[NIPS' 16]**
     **[R-FCN++]** R-FCN++: Towards Accurate Region-Based Fully Convolutional Networks for Object Detection
      R-FCN中抛弃了全连接层，而是使用卷积层结合位置敏感RoI池化层，使得RoI-aware的操作只有一层。

   * Rethinking Classification and Localization in R-CNN
  
   * Grid RCNN
     Grid RCNN Plus
     
2. 关联不同RoI间的信息

    * Relation Network

3. 提高分类检测的能力

    * Cascade RCNN


### 七、损失函数及后处理

* GIoU Loss
* KL Loss(Softer NMS)

* Soft-NMS
* IoU-Net
 （MS R-CNN）

### 八、采样方式

* Prime Sample Attention in Object Detection 
* Libra R-CNN
   
### 九、训练方式

* MegDet
* OHEM
* Rethinking ImageNet Pre-training
* RePr
* Bag of Tricks for Image Classification with Convolutional Neural Networks
* Bag of Freebies for Training Object Detection Neural Networks
* Augmentation for small object detection

### 十、其他目标检测

1. 领域迁移

   * Domain Adaptive Faster R-CNN for Object Detection in the Wild 
   * Few-shot Adaptive Faster R-CNN
   * Strong-Weak Distribution Alignment 

2. 蒸馏

   * Quantization Mimic: Towards Very Tiny CNN for Object Detection
   
3. 图网络
