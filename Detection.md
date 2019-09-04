## 检测场景 (DOING)

检测任务作为一个基础任务，在工业界运用广泛。很多机器学习项目的整个pipeline中的第一步就是检测，这时，可以选择一个低延迟的检测网络，尽可能地减少检测的耗时，如YOLO、DenseBox；也可以选择一个基于区别的多任务网络来并行得执行多个任务。这篇笔记的重点将不是从模型本身去改进检测效果，而从任务目标，从问题场景上，针对不同的问题去讨论优化的方向     



### 一、小目标检测    


小目标检测算是一个通用任务，不管是通用目标检测，还是人脸、行人检测，都会遇到这样的问题，效果差的主要原因（彼此相关）：     

* 候选框对目标GT的覆盖率低     
* 候选框数量增大导致FP增大，mAP下降。正样本少，无关负样本过多，使得模型无法学到具有足够判别性的目标特征     


1. 选框对目标GT的覆盖率低    


   GT和Anchor大于0.5才是正例，即该GT被覆盖到。    

   图像多尺度，特征多尺度：增加候选框的密度    
   Seeing small。。。   

   放大图片的数据增强    

2. 选框数量增大导致FP增大，mAP下降。正样本少，无关负样本过多，使得模型无法学到具有足够判别性的目标特征     

   * Augmentation for small object detection    
      针对小目标的数据增强方法，思路是将小目标拷贝到图中的任意位置    

   难样本挖掘？    
   refinedet过滤简单负样本？    
   多个图的目标拼接在一起，单独在RCNN上finetune？    


### 二、遮挡目标检测    

softnms?      

### 三、行人检测    

小目标问题、遮挡问题    

* **[CSP]** Center and Scale Prediction: A Box-free Approach for Object Detection **[CVPR' 19]**    

RepLoss     


### 四、人脸检测     

* **[FaceNet]** FaceNet: A Unified Embedding for Face Recognition and Clustering **[CVPR' 15]**     


* **[MTCNN]** Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks     


* **[DSFD]** DSFD: Dual Shot Face Detector **[CVPR '19]**       


### 五、自然场景下文字检测     

* **[RRPN]** Arbitrary-Oriented Scene Text Detection via Rotation Proposals      

旋转角、长宽比相差太长?    
Anchor-Free，GT外扩     

### 关联目标      

人脸和人头对应同一个anchor？车辆和车牌，人身和人头关联？    



### 固定摄像头（固定背景Context）   

### 检测问题排查     

### 目标的Representation    

框，旋转框，实习分割，模型到底学到了什么？   


