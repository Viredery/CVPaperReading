## 检测场景 (DOING)

检测任务作为一个基础任务，在工业界运用广泛。很多机器学习项目的整个pipeline中的第一步就是检测，这时，可以选择一个低延迟的检测网络，尽可能地减少检测的耗时，如YOLO、DenseBox；也可以选择一个基于区别的多任务网络来并行得执行多个任务。这篇笔记的重点将不是从模型本身去改进检测效果，而从任务目标，从问题场景上，针对不同的问题去讨论优化的方向     


### 一、小目标检测    


小目标检测算是一个通用任务，不管是通用目标检测，还是人脸、行人检测，都会遇到这样的问题，效果差的主要原因（彼此相关）：     

* 候选框对目标GT的覆盖率低     
* 候选框数量增大导致FP增大，mAP下降。正样本少，无关负样本过多，使得模型无法学到具有足够判别性的目标特征     

1. 选框对目标GT的覆盖率低    

   GT和Anchor大于0.5才是正例，即该GT被覆盖到。    

   * 图像多尺度(包括放大图片)，特征多尺度      
      基于FPN系列的特征多尺度，和基于SNIP系列的图像多尺度的建模方法，都增加了候选框的数量和密度，使得小目标被Anchor覆盖的几率变高       
  
   * 增加候选框的密度     
      Seeing Small Faces from Robust Anchor’s Perspective **[CVPR' 18]**        
      对于特征图，增加Anchor的密度，比如原本Anchor采样过程中的stride为1，现调整为1/2         

   * Augmentation for small object detection    
      针对小目标的数据增强方法，思路是将小目标拷贝到图中的任意位置，增加正样本的数量     


2. 选框数量增大导致FP增大，mAP下降。正样本少，无关负样本过多，使得模型无法学到具有足够判别性的目标特征     
   基本上就是正负样本不平衡的问题，常见的解决方法：难样本挖掘，RefineDet过滤简单负样本等    


### 二、遮挡目标检测    

* **[Soft-NMS]** Improving Object DetectionWith One Line of Code **[ICCV' 17]**     
   从后处理的角度，防止被遮挡的目标被过滤掉        

* Repulsion Loss: Detecting Pedestrians in a Crowd **[CVPR' 18]**     
   从损失函数的角度，让不同GT对应的Anchor的回归距离尽可能地拉远        

* A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection    
   从数据集角度，通过GAN生成更多的难正样本，主要是覆盖遮挡和形变问题        

### 三、行人检测      

行人检测的主要问题就是上面提及的遮挡问题，以及小目标情况下的可辨别性较弱，容易漏检             

* **[CSP]** Center and Scale Prediction: A Box-free Approach for Object Detection **[CVPR' 19]**    


### 四、人脸检测     

* **[FaceNet]** FaceNet: A Unified Embedding for Face Recognition and Clustering **[CVPR' 15]**     

* **[MTCNN]** Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks    

* PyramidBox: A Context-assisted Single Shot Face Detector    

* **[DSFD]** DSFD: Dual Shot Face Detector **[CVPR '19]**       

### 五、自然场景下文字检测     

文字检测的主要问题包括：旋转文字，弯曲文字；长宽比过大      
常见Trick：Ground Truth外扩        

* **[RRPN]** Arbitrary-Oriented Scene Text Detection via Rotation Proposals      

### 六、一些探讨

1. Cascade结构      
   讨论下Cascade RCNN, HTC, ConRetineNet, Cascade RPN, AlignDet       

2. 目标检测中的不平衡问题       
   Imbalance Problems in Object Detection: A Review       
   Are Sampling Heuristics Necessary in Object Detectors?         
   类别不平衡、尺度不平衡、空间不平衡、多任务损失优化之间的不平衡   
   类别不平衡: Focal Loss     
   尺度不平衡: Consistent Scale Normalization for Object Recognition   
   空间不平衡: Libra R-CNN、 Cascade R-CNN    
   多任务损失优化之间的不平衡: 调整loss weight，Prime Sample Attention   

3. 不同类别目标之间的关联      
4. 固定摄像头（固定背景Context）   
5. 检测问题排查     
6. 目标的Representation    
 


