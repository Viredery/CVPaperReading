## 一阶段目标检测

目前来说，一阶段和阶段检测网络的一个主要的区别标准在于是否使用了RoI Pooling。由于没有refinement的过程，设计一阶段网络的关注点放在了  

* 正负样本不平衡问题（两阶段检测中RPN过滤掉了大部分简单负样本）     
* 目标的回归精度问题（两阶段检测中不断修正回归框）    
* feature map和anchor之间的misalignment问题（两阶段检测中使用RoI Pooling方法）    


### 一、YOLO系列    


* **[YOLO v1]** You Only Look Once: Unified, Real-Time Object Detection **[CVPR' 16]**    
   YOLO设计的初衷就是实现一个实时的端到端检测网络。输入图片448x448，网络输出特征图7x7。每个位置输出一个(1 + 4) x 2 + C       
   这里，每一个位置会输出两个区域框，但其对应类别只有一个。因此在训练时，如果该位置上确实存在目标，那么只选择与ground truth的IoU最大的那个区域框来负责预测该目标，而另一个边界框认为不存在目标     
   由于只计算49x2个区域框中是否存在目标的置信度，因此不存在严重的不平衡问题。YOLO中的不平衡处理方式是调整不同损失函数的权重。对于存在目标的区域框的置信度损失，权重为1；而不存在目标的区域框的置信度损失权重为0.5。另外，回归损失的权重为5，回归 normalized x y w h    

* **[YOLO v2]** YOLO9000: Better, Faster, Stronger **[CVPR' 17]**          
   YOLO存在的问题：召回低，回归准确率低。YOLO v2引入Anchor机制，通过对GT聚类来选定Anchor的长宽，网络预测13 x 13 x 5(num_anchor)个区域框（YOLO v2输出的特征图大小为 13 x 13），提高召回     
   （这里的聚类分析的方法，非常适合业务场景中解决Anchor的设定问题）      
   引入Anchor后，一方面候选区域框大幅增加，其实就给训练引入了样本不平衡的问题。但这里候选框不到1k个，影响不大。另一方面，引入了feature map和anchor之间的misalignment问题，YOLO v2设计了回归值和其损失函数，将每个位置的Anchor的中心点，限制在该位置范围(32 x 32 pixels)内    
   使用Trick：Darknet-19在分类任务中微调；高分辨率特征融合；多尺度训练等    
   训练过程：YOLO v2的训练过程，包括损失函数的设计，非常复杂。YOLO v2里面叫box priors而不是叫Anchor（下面都统一称为Anchor），是因为YOLO v2中的回归，依旧是基于位置回归而不是基于Anchor回归，网络训练初期会增加一个损失，使每个位置上回归的区域框去拟合预设框Anchor。此外，YOLO系列中，每一个GT只会分配给一个Anchor，这与R-CNN、SSD等不同    
   YOLO9000设计了一个不同任务下的数据集联合训练的方式。

* **[YOLO v3]** YOLOv3: An Incremental Improvement     
    Darknet-53，加入残差结构。三个尺度的feature map，每个尺度计算3个Anchor，提高小目标的检测率    


### 二、SSD系列    

SSD系列其实有很多文章，但我只看过最早的SSD。相比于YOLO系列，SSD在网络设计和训练方式上更加接近R-CNN系列     

* **[SSD]** SSD: Single Shot MultiBox Detector **[ECCV' 16]**      
   Backbone使用的是加入dilated conv的VGG16。采用多尺度特征图结合Anchor机制用于检测。Anchor机制和R-CNN基本一致，不同尺度上设计不同数量和尺度的Anchor    

   对于样本不平衡问题，SSD在训练过程中类似RPN，使用采样的方式来保证训练过程中的正负样本比例，不过这里的采样方式是OHEM    

   数据增强：color jitter；random crop；random expand   
   
   特点：SSD使用随机裁剪，使得小目标被放大，每个像素对应的Anchor得到更充分地训练。但另一方面，检测小目标所用的特征图来自于浅层信息（高分辨率信息），缺少低分辨率的语义信息，因此小目标的检测效果不好    

* **[DSSD]** DSSD: Deconvolutional Single Shot Detector       
   DSSD相比SSD做了两点改变，一是低分辨率语义信息和高分辨率边缘信息的融合，类似FPN的结构，但使用了反卷积层，提高小目标的检测效果；二是在分类和回归的预测模块中，将卷积层替换为残差层，提高准确率     

* **[RefineDet]** Single-Shot Refinement Neural Network for Object Detection **[CVPR' 18]**    
   一阶段方法检测精度低的一个主要原因是类别不平衡问题。RefineDet提出ARM和ODM两个阶段，前者过滤掉简单的负Anchor，为第二阶段的分类器减少搜索空间，并粗调回归框，为第二阶段的回归器提供更好的初始候选框，类似RPN阶段；后者进行精调，类似SSD阶段。训练上，第一阶段使用logit loss，得分低于0.01的Anchor被过滤，第二阶段使用OHEM    

### 三、RetinaNet

* **[RetinaNet]** Focal Loss for Dense Object Detection **[ICCV' 17]**     
   RetinaNet直接使用大尺度的输入图片和大网络backbone，导致严重的不平衡问题，因此引入了Focal Loss。Focal Loss类似于一种boosting的方式，在使用时需要初始化最后一层Conv或FC层的偏差，以防梯度爆炸    

* **[ConRetinaNet]** Consistent Optimization for Single-Shot Object Detection    
   提出了训练和测试的不一致问题：训练过程中，设置IoU大于0.5为正例，进行分类和回归计算，但在测试过程中，将原始Anchor训练的得分赋给经过回归调整之后的Anchor。统计发现，分类得分的方差随着IoU的增长会不断变大。
   针对RetinaNet中训练和测试的不一致问题，使用了一种类似Cascade但不增加参数的结构，即过两遍regression/classification head    

* **[Cascade RetinaNet]** Cascade RetinaNet: Maintaining Consistency for Single-Stage Object Detection **[BMVC' 19]**      
   相比ConRetinaNet，Cascade RetinaNet过两遍不一样的regression/classification head，类似Cascade R-CNN。但一阶段和二阶段的本质不同是两阶段的Cascade结构是接在Region-based CNN部分的，因此不会有misalignment的问题。这篇文章提出的解决方法就是用Deformable Conv    

### 四、Anchor Free

这里面的大多数网络，都是将基于框（Anchor）的检测任务替换成了基于位置（像素点）的检测任务。最近一年，这种Anchor-Free的方法获得了和Anchor-based方法相近的效果，主要得益于：1）FPN的引入，不同大小的目标会被分配到不同的特征图上，减缓了目标之间互相遮挡、不同目标分配到同一个位置上等问题；2）Focal Loss的引入，使得候选的位置可以尽可能多地覆盖原图      

* **[DenseBox]** DenseBox: Unifying Landmark Localization with End to End Object Detection    
   DenseBox这个模型和YOLO一样，是工业界常用的实时目标检测模型。直接预测目标框的坐标相对于像素位置的偏移；引入了landmark任务作为辅助监督；多尺度特征的融合；数据增强等

* **[CornerNet]** CornerNet: Detecting Objects as Paired Keypoints **[ECCV' 18]**      
   CornerNet-Lite: Efficient Keypoint Based Object Detection    
   CornerNet不再是基于Anchor或者是基于位置点，而是输出两个热力图，分别预测目标的一对关键点（左上角和右下角），同时预测关键点的Offsets得到更精准的位置。最后，预测一个embedding值，来拉近相关的关键点，拉远不相关的关键点     
    介绍了一种叫做Corner Pooling的方式，以及一种embedding的训练方式     
 
* **[FCOS]** FCOS: Fully Convolutional One-Stage Object Detection      
   NAS-FCOS: Fast Neural Architecture Search for Object Detection     
   输出三组特征图，分别是每一个像素点应该分配的目标，该像素点到目标的边界的距离，以及该像素点到目标中心的距离     


* **[FSAF]** Feature Selective Anchor-Free Module for Single-Shot Object Detection    
   结合Anchor-based预测头和Anchor-free预测头，然后把两者的结果结合起来。对于Anchor-free部分。目标的损失值会在所有的特征图上进行计算，但只有损失值最小的那个特征图才会计算该目标的反向传播值    

* **[CenterNet]** Objects as Points    
   类似人体姿态评估的做法，输出两个热力图，一个预测目标的中心点，一个预测目标的长和宽。这里，只有一个特征图去预测目标的类型和位置        

* FoveaBox: Beyond Anchor-based Object Detector   



### 五、AlignDet   

* **[AlignDet]** Revisiting Feature Alignment for One-stage Object Detection    
   针对feature map和anchor之间的misalignment问题，AlignDet结合Anchor的设置情况，固定Deformable Conv里的offset，使得feature map上的每一个位置，都对其对应的anchor的大小和形状敏感
