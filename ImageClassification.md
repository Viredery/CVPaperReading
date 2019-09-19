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

   Inception系列论文指出CNN的通用设计思想：1、通常，随着网络层数增加，空间特征数逐渐降低，通道数逐渐增加；2、不要过度压缩损失精度信息，避免表征瓶颈。3、增加非线性特征可以解耦合特征；4、卷积的实现形式为空间聚合；5、1x1卷积降维不会产生严重的影响。猜测：相邻通道之间具有强相关性，卷积前降维基本不会损失信息         
   Inception 模块的目的是设计一种具有优良局部拓扑结构的网络，即对输入图像并行地执行多个卷积运算或池化操作，并将所有输出结果拼接为一个非常深的特征图。因为 1*1、3*3 或 5*5 等不同的卷积运算与池化操作可以获得输入图像的不同信息，并行处理这些运算并结合所有结果将获得更好的图像表征      

   * **[Inception v1]** Going deeper with convolutions       
      Inception结构：对于输入的 feature maps，分别通过1x1卷积、3x3卷积、5x5卷积和 Max-Pooling 层，并将输出的 feature maps 连接起来作为 Inception 的输出【同时获得不同感受野的信息】。在3x3卷积、5x5卷积前面和池化层后面接1x1卷积，起降维的作用     

   * **[Inception v2 v3]** Rethinking the Inception Architecture for Computer Vision     
      5x5卷积的感受野与两个3x3卷积堆叠所对应的感受野相同。使用后者可以大大减少网络参数。7x7同理。此外，两个3x3卷积后各连接一个非线性层的效果优于仅在最后连接一个非线性层      
      NxN的卷积可以用1xN与Nx1的卷积堆叠实现      
      训练时，使用 Label Smoothing 增加网络的正则能力。使用 Batch Normalization 和 RMSProp      

   * Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning    
      相比v3，Inception v4的主要变化是网络的加深，卷积和的个数也大大增加       
      Inception-ResNet即将ResNet的残差结构替换成了简单的Inception结构       


7. ResNet系列
   * **[ResNet]** Deep Residual Learning for Image Recognition       
      网络过深会导致退化问题，通过短路连接（恒等映射）解决该问题     

   * **[ResNet v2]** Identity Mappings in Deep Residual Networks      
      尝试了多种恒等映射的方法，发现shortcut的效果最好。尝试多种卷积堆叠（残差）的方法，发现BN-ReLU-Conv-BN-ReLU-Conv的方法，优于ResNet中的Conv-BN-ReLU-Conv-BN的方法        

   * **[ResNeXt]** Aggregated Residual Transformations for Deep Neural Networks       
      指出 Inception 过于复杂，不易迁移到其他问题上；ResNet 存在 diminishing feature reuse 的问题。提出了基数的概念，残差块采用 split-transform-merge 的策略，基数类似 group，表示 split 的数目。这种架构可以接近 large and dense layers 的表示能力，但只需要很少的计算资源        

   * Wide Residual Networks        
      ResNet存在diminishing feature reuse的问题。网络过深，很多残差块对最终结果只做出了很少的贡献。提出增加残差块的宽度，减少网络深度的WRNs         

8. **[DenseNet]** Densely Connected Convolutional Networks         
   DenseNet 极大地增加了特征重用的能力，其有以下优点。1. 参数少，通过向后连接的方式保留学到的信息；2. 改进了前向、反向传播，更易训练；3. 增加了监督学习的能力；4. 在小数据上不易过拟合，即增加了正则化的能力         


9. **[SENet]** Squeeze-and-Excitation Networks        
    引入SE-Block，自动学习一个feature map中不同的channels对应的权重大小      

10. **[DPN]** Dual Path Networks         
   DPN 融合了 ResNeXt 和 DenseNet 的核心思想       

11. **[Res2Net]** Res2Net: A New Multi Scale Backbone Architecture       
   Res2Net 是在粒度级别上来表示多尺度特征并且增加了每层网络的感受野范围。与现有的增强单层网络多尺度表达能力的 CNNs 方法不同，它是在更细的粒度上提升了多尺度表征能力      


### 二、轻量级网络


1. **[SqueezeNet]** SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size **[ICLR' 17]**        
   SqueezeNet 的基本结构是 Fire Module，输入的 feature maps 先经过1x1卷积降维，然后分别通过1x1卷积和3x3卷积，并将两个输出连接起来，作为这个模块整体的输出。SqueezeNet 的结构就是多个 File Module 堆叠而成，中间夹杂着 max pooling。最后用 deep compression 压缩      

2. **[Xception]** Xception: Deep Learning with Depthwise Separable Convolutions    
   Xception 相当于借鉴了 depth-wise 的思想，简化了 Inception-v3。Xception的结构是，输入的 feature maps 先经过一个1x1卷积，然后将输出的每一个 feature map 后面连接一个3x3的卷积（再逐通道卷积），然后将这些3x3卷积的输出连接起来     
   【和MobileNet v1的区别在于1x1卷积和3x3卷积的先后次序】     

3. MobileNet系列    
   * **[MobileNet v1]** Mobilenets: Efficient Convolutional Neural Networks for Mobile Vision Applications        
      MobileNet 的基本结构是 3x3 depth-wise Conv 加 1x1 Conv。1x1卷积使得输出的每一个 feature map 要包含输入层所有 feature maps 的信息。这种结构减少了网络参数的同时还降低了计算量。整个 MobileNet 就是这种基本结构堆叠而成。其中没有池化层，而是将部分的 depth-wise Conv 的 stride 设置为2来减小 feature map 的大小      

   * **[MobileNet v2]** MobileNetV2: Inverted Residuals and Linear Bottlenecks      
      Inverted Residuals：将ResNet的沙漏型残差结构改为纺锤形；Linear Bottlenecks:去掉最后一个1x1卷积的激活函数

   * **[MobileNet v3]** Searching for MobileNetV3    


4. ShuffleNet系列    
   * **[ShuffleNet v1]** ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices     
      ShuffleNet 认为 depth-wise 会带来信息流通不畅的问题，利用 group convolution 和 channel shuffle 这两个操作来设计卷积神经网络模型, 以减少模型使用的参数数量，同时使用了 ResNet 中的短路连接。ShuffleNet 通过多个 Shuffle Residual Blocks 堆叠而成     

   * **[ShuffleNet v2]** ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design    
      这篇文章讨论了设计一个推断高效的网络的准则。FLOPs反应了模型的计算量，但无法直接反应推断速度（或者吞吐量）。一、当输入、输出channels数目相同时，conv计算所需的MAC(memory accesscost)最为节省。二、过多的group convolution会加大MAC开销。三、网络结构整体的碎片化（如Inception）会减少其可并行优化的程序。四、element-wise操作会消耗较多的时间，不可小视。
      ShuffleNet v2抛弃了v1中的1x1的Group Conv，而是直接使用了输入输出channels数相同的1x1 Conv和3x3 DWConv。提出了一种ChannelSplit新的类型操作，将module的输入channels分为两部分，一部分直接向下传递，另外一部分则进行真正的向后计算。到了module的末尾，直接将两分支上的output channels数目级连起来          

### 三、AutoML系列网络

1. NAS系列
   * **[Nas]** Neural Architecture Search with Reinforcement Learning    
   * **[Nasnet]** Learning Transferable Architectures for Scalable Image Recognition         
   * **[MnasNet]** MnasNet: Platform-Aware Neural Architecture Search for Mobile         
   
2. **[EfficientNet]** EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks             


### 四、权重初始化方式

1. Understanding the difficulty of training deep feedforward neural networks    
2. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification         

### 五、分类损失函数     

1. Softmax

   Softmax、Softmax with temperature parameter、Label Smoothing、多标签分类时Softmax和Sigmoid的联系       
   Softmax with temperature parameter相当于一种soft Softmax方法。在工程使用时，所有的输入值都减去输入值中的最大值，对分母加一，以防止上溢下溢                

2. Focal loss      
   soft sampling方法，降低简单样本的损失值的权重，而更加关注难样本比如属于rare class的样本          

3. Center loss       
   Softmax尽可能地将不同类别的样本分开，但没有考虑样本特征的可辨别性。center loss引入了每一个类别的样本的中心点，在Softmax的基础上，计算一个距离函数，令同一个类别的样本都聚集在该类别的中心点上，提高类内聚合       
   Contrastive center loss      
   Center loss只考虑类内聚合，没有考虑类间分离，Contrastive center loss增加了一个分母项，拉远该类别样本到其他类别的中心点的距离       

4. ArcFace loss    
   
   * **[L-Softmax]** Large-Margin Softmax Loss for Convolutional Neural Networks **[ICML' 16]**     
      将Wx分解为||W|| * ||x|| * cos(theta)，为了提高类内紧凑性和类间分离性，对于正确类，计算cos(m*theta) （需要保证单调下降），对于其他类，不乘参数m，即计算cos(theta)      
      m控制类别之间的差距. 随着m越大(在相同的训练损失下)，类之间的margin变得越大, 学习困难也越来越大         

   * **[A-Softmax]** SphereFace: Deep Hypersphere Embedding for Face Recognition **[CVPR' 17]**      
      相比L-Softmax而言，A-Softmax将||W||固定为1，训练的时候加入Softmax帮助收敛       
  
   * F-Norm SphereFace      
      相比A-Softmax而言，F-Norm SphereFace对特征也做了正则化，使得||W||固定为1，现在||W|| * ||x|| * cos(theta)变成了s * cos(m * theta)       

   * CosineFace     
      相比F-Norm SphereFace，CosineFace将s * cos(m * theta)变成了s * (cos(theta) - m)      

   * **[ArcFace]** ArcFace: Additive Angular Margin Loss for Deep Face Recognition      
      相比CosineFace，ArcFace将s * (cos(theta) - m)变成了s * cos(theta + m)      
