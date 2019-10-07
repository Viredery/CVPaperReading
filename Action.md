### 一、Action Recognition   


1. 双流法    

   * **[Two-stream]** Two-Stream Convolutional Networks for Action Recognition in Videos **[NIPS' 14]**   
      将网络分成空间部分和时序部分，前者的输入是视频中的某一视频帧，后者输入是密集光流     

   * **[TSN]** Temporal Segment Networks: Towards Good Practices for Deep Action Recognition **[ECCV' 16]**     
      将一个视频分成三个片段snippets，每个片段中分明进过双流网络，网络在不同snippets中权重共享。此外，增加了输入数据的可选形式，对时序网络部分进行了初始化     

   * **[StNet]** StNet: Local and Global Spatial-Temporal Modeling for Action Recognition **[AAAI' 19]**   
      StNet将视频采样为T个snippets，每个snippet相当于连续N帧图像级联成一个3N通道的图，用2D卷积对图进行空间联系的建模；在Res3和Res4模块后面，加入3D卷积以进行时序联系的建模，最后使用时序Xception模块进行融合      

   * On the Integration of Optical Flow and Action Recognition    
      作者认为光流在行为识别模型中效果好的原因在于其对于图像表观的不变性。若舍弃光流，同时针对表观的色彩/纹理/光照做数据增强，那么只用RGB图像可能也能获得不错的效果；可以通过提高网络本身对表观变化的学习能力，来替代光流表观不变性的作用

2. 3D卷积    

   * **[C3D]** Learning Spatiotemporal Features with 3D Convolutional Networks    
      直接使用3D卷积网络，输入是连续16个视频帧     

   * **[I3D]** Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset **[CVPR' 17]**       
      对Two-Stream结构的两个输入（一个是连续的光流帧，一个是连续的视频帧），分别使用3D卷积网络提取特征，然后将两个流的结果取平均     

### 二、Temporal Action Detection

1. **[SSAD]** Single Shot Temporal Action Detection **[ACM MM' 17]**      
   SSAD的输入包括某个视频帧，连续光流帧，连续视频帧三部分，分别经过Two-Stream和C3D得到三组分类得分输出，然后拼在一起作为SAS特征，truncate到固定长度后经过一系列卷积生成特征金字塔，金字塔每层输出“该层Anchor数 *（类别数，delta x，delta w，overlap score）”      
    训练时，truncate的滑窗的overlap ratio为0.75，移除所有没有anno的滑窗；设置IoU大于0.5的proposals为正利；难样本挖掘。推断时，runcate的滑窗的overlap ratio为0.25；保留所有滑窗；最终预测得分与SAS的三组得分融合；NMS       

2. BSN: Boundary Sensitive Network for Temporal Action Proposal Generation **[ECCV' 18]**    
   Two-Stream提取特征，然后输出三个密集概率：是否是starting，ending，duration，然后组装成proposals，接着预测是否存在行为的概率，推断时使用softNMS来进行后处理    

### 三、Spatio-Temporal Action Detection       

1. TACNet: Transition-Aware Context Network for Spatio-Temporal Action Detection    
   网络分为两部分，Two-Stream Temporal Context Detector和Transition-Aware Classification and Regression。前者使用SSD提取每一帧的信息，不同帧之间的同一尺度的特征图后面接ConvLSTM，来进行上下文信息的建模。后者预测每一个目标所属的动作类别，和该目标的动作是否是Transition State        
