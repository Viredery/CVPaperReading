## 两阶段目标检测

### 一、基础框架

两阶段网络的总体流程分为五步，通过Backbone提取特征、RPN网络提取RoI、RoI池化、RCNN网络进行分类和回归以及后处理得到检测结果。

同时，由于目标在原图中的尺度差异较大，单一分辨率的特征无法很好地预测较大和较小尺度的目标，因此引入多尺度建模方法来解决不同尺度目标的预测问题。

1. 两阶段检测的总体流程

   *[R-CNN]*Rich feature hierarchies for accurate object detection and semantic segmentation *[CVPR’ 14]*

   *[Fast R-CNN]*Fast R-CNN *[ICCV’ 15]*

   *[Faster R-CNN, RPN]*Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks *[NIPS’ 15]*


2. 多尺度建模方法

   *[FPN]*Feature Pyramid Networks for Object Detection *[CVPR’ 17]*   
   目前最流行的方法。输出多个不同分辨率的特征图。分辨率大的特征图检测小目标；分辨率小的特征图检测大目标

   *[SNIP]*An Analysis of Scale Invariance in Object Detection – SNIP *[CVPR’ 18]*   
   输入为图像金字塔，对于图像金字塔中的每一个图像，仅选择固定尺度范围内的目标进行训练（反向传播）。也就是说，原图像放大后仅检测小目标，原图像缩小后仅检测大目标。由于每个目标在训练时都会有几个不同的尺寸，那么总有一个尺寸是在指定的尺寸范围内。

   *[TridentNet]*Scale-Aware Trident Networks for Object Detection *[CVPR’ 19]*    
   输出多个不同感受野的特征图。以ResNet50为例，将Stage5卷积移至RCNN，将Stage4分成三个分支，不同分支中的3x3卷积的dilated分别设置为1，2和3，不同分支共享权重。同时，对于不同的分支，选择特定尺度范围内的目标进行训练（反向传播）。也就是说小目标和大目标分别在不同的分支中检测，最后三个分支的预测结果通过NMS结合起来。

