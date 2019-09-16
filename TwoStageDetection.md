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

    常见的Backbone为ResNet、ResNeXt、Inception-ResNet，针对轻量化设计要求，也可以使用MobileNet等网络。这些网络在分类任务中被提出，被运用在了各种视觉任务中。同时，也有些网络针对位置敏感问题设计的网络。DetNet网络中将最后一阶段的stride从2减小到1并将dilated设置为2也是常见的针对位置敏感问题的优化方法。同时也有NAS去搜索目标检测中最优骨架网络结构 **[DetNas]** 的工作。    
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

    FPN的改进有两个方向。一是研究如何更好的生成金字塔特征，如**PANet**和**NAS-FPN**；二是如果更好地出金字塔特征中提取RoI特征，如**PANet**和**Libra R-CNN**，一阶段目标检测网络**FSAF**种也提出了一种有趣的方法。    

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

1. 网络结构改进

   * **[GA-RPN]** Region Proposal by Guided Anchoring **[CVPR' 19]**      
      大多数解决方案，都是根据不同的数据集，去设计每个位置的预选框（Anchor）的尺度和长宽比（即调参，调整先验假设）。GA-RPN是训练过程中指导预选框的生成。对每层特征图的每一个位置，通过训练去预测这个位置对应的预选框的高和宽(**Bounded IoU Loss**)。


2. 采样方式改进

   * **[Libra R-CNN]** Libra R-CNN: Towards Balanced Learning for Object Detection **[CVPR' 19]**    
      常规过程中，一个短边800像素的图片，会产生20万左右的Anchor，绝大多数都是负样本，随机进行采样，会产生很多简单背景作为负样本，无法很好地把握目标和非目标之间的辨识度。Libra R-CNN提出了一种基于IoU的采样方式。

3. Anchor机制

   * 19年出了很多Anchor-Free的论文，其实两阶段网络也可能完全舍弃Anchor机制。比如将RPN部分换成**FCOS**的头(Head)，RCNN部分保持不变，或者换成**Grid RCNN**等。这里没提改进，是因为目标我们没法证明其中一种方法优于另一种。

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

   * **[Double-Head RCNN]** Double-Head RCNN: Rethinking Classification and Localization for Object Detection    
      做了一堆实验，结论是，box回归前用卷积，class分类前用全连接效果最好。这里卷积使用的是残差块和Non-local块。    
      
   * **[Grid RCNN]** Grid RCNN    
     **[Grid RCNN Plus]** Grid RCNN Plus: Faster and Better    
      由于RCNN阶段GT和RoI的交并比大于0.5为正，因此对RoIAlign后的结果上采样四倍，可以完全覆盖原GT，然后预测目标的九个点（边框上八个点，目标中间一个点，类似热点图）。即完全把回归损失变为分类损失。    
     
2. 关联不同RoI间的信息

   * **[Relation Network]** Relation Networks for Object Detection     
      简而言之就是将前面提到的Non-local放在了RCNN中，然后将NMS后处理也改为用Relation来做。    

3. 提高分类和检测能力

   * **[Cascade RCNN]** Cascade R-CNN: Delving into High Quality Object Detection    
      一个标准的通用做法，就是不断地去串联同一个模块，不断地去精修结果，**Stacked Hourglass Network** 也是这么做的    
       
4. 预测的分类得分无法指导后处理的问题
 
   这个问题也是19年前后大家开始关注的一个问题。RCNN系列的训练过程中，认为交并比大于0.5就算正例，因此，模型本身并不能使IoU越大的框的得分越高。**Mask Score R-CNN**中也指出，分类得分并不能指导每个Mask预测情况的好坏。
         	
   * **[IoUNet]** Acquisition of Localization Confidence for Accurate Object Detection **[ECCV' 18]**    
      增加一个分支去预测回归框和GT的IoU，用这个IoU值去指导NMS


### 七、损失函数及后处理

* Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics    
  Auxiliary Tasks in Multi-task Learning    
   损失函数权重的自动学习。对于多任务学习存在多个损失函数的情况，不同损失函数的权重是个比较重要的超参数。手动调参的计算代码巨大。本文提出了将该权重作为一个参数，在模型训练的过程中去自动地更新、学习。   
    
* **[GIoU Loss]** Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression **[CVPR' 19]**    
   出发点是常用的Smooth L1损失不是基于IoU优化。而IoU损失有两个问题，一是在GT和预测的框的IoU为0时没有梯度，二是IoU损失对于不同重叠的方式的惩罚力度是一样的。因此提出了GIoU损失，更加严格地进行了约束。

* **[KL Loss(Softer NMS)]** Bounding Box Regression with Uncertainty for Accurate Object Detection **[CVPR' 19]**    
   对框的四条边的不确定性进行建模。出发点和上面的损失函数权重的自学习差不多。引入KL loss来评估 ground truth 和 预测的 bounding box 分布之间的"差距"。学到的“差距”（或叫“不确定性”）的变性，在后处理的时候作为Box Voting的权重，即Softer NMS    
  
* **[Soft-NMS]** Improving Object DetectionWith One Line of Code **[ICCV' 17]**    
   对于两个距离很近的目标，在NMS过程中可能会因为IoU过大而被舍弃。Soft-NMS不会舍弃预测出来的框，而是将得分次高的框的得分进行抑制。

* Prime Sample Attention in Object Detection    
   这篇论文的出发点和上面几篇不一样。大多数情况下，一个共识是，大量的简单样本对模型的参数更新方向的帮助不大（一个例子，比如Triplet Loss中，随机选三元组可能会训不起来，而在同一个batch里做难样本挖掘来生成三元组却有效果）。而这篇论文，在做回归损失的时候，降低了难样本的权重，提高简单样本的权重。论文给出的动机是，在NMS阶段，更好质量的框会保留下来，其他的框被丢弃的，那么，影响最后的指标的是训练过程中的简单样本。刚好与难样本挖掘反其道而行之       
   


### 八、训练方式

* Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour    
   **[MegNet]** MegDet: A Large Mini-Batch Object Detector    
   这两篇论文介绍了在大batch size的情况下，如何训练模型。    
   对于目标检测任务，由于输入图片的尺度较大，导致batch size一般只能设置为1或者2，在这种情况下无法计算BN的准确统计量。这是因为batch较小时，BN计算的统计量存在一定的偏向性(bias)，正负样本的比例也存在着抖动和严重的不平衡。因此需要Cross-GPU BN，即SyncBN，来增加每一次更新所需要统计的图像数量。（这里，batch size和BN size不一定相等）     
   外此对于学习率，linear scaling rule和warmup结合的策略加快训练。linear scaling rule策略：一般两阶段阶段实验中常用的设置是batch size为16，学习率为0.02，那么这个策略就是batch size增大多少倍，学习率就增大同样的倍数。在网络训练初期，较大的学习率导致梯度变化剧烈，因此引入warmup策略      
   
* **[OHEM]** Training Region-based Object Detectors with Online Hard Example Mining **[CVPR' 16]**    
   各种问题下的常用Tricks
   
* Rethinking ImageNet Pre-training   
   指出不使用预训练模型，在较长的训练时间下，可以得到不劣于使用预训练分类权重的模型   
   
* **[RePr]** RePr: Improved Training of Convolutional Filters **[CVPR' 19]**       
   一种学习率Schedule方法

* Bag of Tricks for Image Classification with Convolutional Neural Networks    
  Bag of Freebies for Training Object Detection Neural Networks    
   分类和检测的Tricks
  
* Learning Data Augmentation Strategies for Object Detection    
   目前两阶段最常见的数据增强为随机水平翻转。其他还有随便裁剪，颜色扰动，Mixup等，本文通过AutoML方法自动学习有效的数据增强策略
  

### 九、其他目标检测

1. 领域迁移

   * Domain Adaptive Faster R-CNN for Object Detection in the Wild **[CVPR' 18]**   
      目标检测的领域迁移问题，主要考虑两方面，图像迁移P(I)和目标迁移P(B,I)，文章分别训练两个判别器，使其无法区分两个域的图像和目标，除此之外，还添加了一致性正则化损失项，保证两个层面的域分类结果一致         

   * Strong-Weak Distribution Alignment for Adaptive Object Detection **[CVPR' 19]**    
      这篇文章没有考虑目标迁移，而是在图像迁移中，考虑了高分辨率局部信息和低分辨率全局信息的领域问题。对高分辨率局部信息，使用交叉熵损失进行强对齐；对低分辨率全局信息使用focal loss进行弱对齐       
   * Few-shot Adaptive Faster R-CNN       

2. 蒸馏

   * Quantization Mimic: Towards Very Tiny CNN for Object Detection   
      直接对backbone feature map进行蒸馏效果不好。因此文章对RoIPooling的输出进行蒸馏。此外，将feature map量化，减少了函数的输出空间，降低了学习的难度      
   
3. 图网络学习
