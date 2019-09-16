### 一、大网络的发展演变

1. **[LeNet]** Gradient-based learning applied to document recognition     
   5层网络，引入卷积、Sigmoid激活函数、池化、全连接     
   
2. **[AlexNet]** ImageNet Classification with Deep Convolutional Neural Networks     
   7层网络，数据增强、ReLU代替Sigmoid激活函数、Dropout      

3. **[ZFNet]** Visualizing and Understanding Convolutional Networks      
   提出一种对feature map可视化的方法     

4. Network in Network     
   引入1x1卷积、使用 average pooling 取代 FC    

5. **[VGG]** Very Deep Convolutional Networks for Large-Scale Image Recognition      
   堆叠多个3x3的感受野，可以获得类似于更大感受野的效果    
   VGG里面做了很多的数据增强，包括颜色增强 color jittering，PCA jittering，尺度变换：训练集随机缩放到 [256, 512]，然后随机剪切到224x224    
   尺度变换对应的测试方法：(1) 随机裁剪，取平均，类似AlexNet (2) 将FC转为Conv，原始图片直接输入模型，这时输出不再是1x1x1000，而是NxNx1000，然后取平均     

6. Inception系列    
   **[Inception v1]** Going deeper with convolutions       
   **[Inception v2 v3]** Rethinking the Inception Architecture for Computer Vision     
   Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning    



7. ResNet系列
   **[ResNet]** Deep Residual Learning for Image Recognition  
   **[ResNet v2]** Identity Mappings in Deep Residual Networks        
   **[ResNeXt]** Aggregated Residual Transformations for Deep Neural Networks    
   Wide Residual Networks    

8. **[DenseNet]** Densely Connected Convolutional Networks

9. **[SENet]** Squeeze-and-Excitation Networks

10. Dual Path Networks

11. Res2Net


### 二、轻量级网络


1. SqueezeNet    

2. Xception    


3. MobileNet系列    
   v1    
   v2    
   Searching for MobileNetV3   

4. ShuffleNet系列    
   v1   
   v2    

### 三、AutoML系列网络

1. Nas   
   Nasnet    
   MnasNet
   
2. EfficientNet     
