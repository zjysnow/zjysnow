# Face Recognition
于2015年起，在公司开始人脸识别的研究，从DeepID到Center Loss，再到Am-Softmax，尝试进行复现和改进，至2016年，公司不再将重心放在人脸识别上，转而进行人脸图像超分辨的任务上。

## Database
公司获取的数据集主要包括WebFace，VGGFace2，Microsoft Celeb1M，CACD2000，CelebA，LFW以及TGFace（亚洲人脸数据集）
TGFace由于存在较多噪声，因此被人工清洗了一次，最后基本上用于人脸识别的测试，以及人脸图像超分辨的训练

## Method

### DeepID
商汤于2014年至2015年发表了4个版本，其核心在于
1. 利用Patches保证模型的尺度、平移不变性，但是需要训练多个模型对应不同的Patches，最后将特征串联起来，再利用PCA压缩（DeepID1）
2. 特征使用了不同层的Feature Map生成（DeepID1）
3. 使用交叉熵（DeepFace2）
$$I(f,t,\theta)=-\sum_{i=1}^n{p_i\log{\hat{p}_i}}$$
4. 增加了Verification Loss（DeepID2）
$$V(f_i, f_j, y_{ij},\theta)=\begin{cases}
\frac{1}{2}||f_i-f_j||^2_2 & \mathrm{if} &y_{ij}=1\\
\frac{1}{2}\max{(0,m-||f_i-f_j||^2_2)} & \mathrm{if} & y_{ij}=-1
\end{cases}$$
5. 对每层卷积都增加监督（DeepFace2+）
   其实这点类似于Resnet中的skip connection
6. 更深更好（DeepID3）

在当时公司无法获取商汤CelebA数据集的标签，因此只能够使用WebFace数据集和TGFace数据集进行尝试。但从以上总结可以看出，DeepID需要同时训练多个模型，同时由于公司设备问题，并没有复现，后续人脸识别领域就陷入了修改Loss的竞赛

#### Face Verification
王峰开源了[FaceVerification](https://github.com/happynear/FaceVerification)项目，提供了一个VGG backbone的识别模型

### Center Loss
为每一个类别提供一个中心，使相同类特征更加紧凑，即
$$\mathcal{L}=\mathcal{L}_S+\mathcal{L}_C=-\sum_{i=1}^m{\log{\frac{e^{W^T_{y_i}x_i+b_{y_i}}}{\sum^n_{j=1}{e^{W^T_jx_i+b_j}}}}+\frac{\lambda}{2}\sum^m_{i=1}{||x_i-c_{y_i}||^2_2}}$$

### L-Softmax
考虑类距离
$$\mathcal{L}=-\sum_{i=1}^{m}{\log{\frac{e^{||W_{y_i}||\times ||x_i||\psi(\theta_{y_i})}}{e^{||W_{y_i}|| \times ||x_i||\psi(\theta_{y_i})}+\sum_{j \ne y_i}{e^{||W_j|| \times ||x_i|| \cos(\theta_j)}}}}}$$
$$\psi(\theta)=\begin{cases}
\cos(m\theta) & 0 \le \theta \le \frac{\pi}{m} \\
\mathcal{D}(\theta) & \frac{\pi}{m} \lt \theta \le \pi
\end{cases}$$
其中，$m$是一个与分类边界相关的参数，$m$越大，越难训练。

### A-Softmax
A-Softmax在考虑margin的时候，将权重$W$进行了归一化，并令偏置$b=0$，此时模型的预测仅取决于权重和特征之间的角度
$$\mathcal{L}=-\sum_{i=1}^{m}{\log{\frac{e^{||x_i||\cos(m\theta_{y_i})}}{e^{||x_i||\cos(m\theta_{y_i})}+\sum_{j \ne y_i}{e^{||x_i||\cos(\theta_j)}}}}}$$

### NormFace
在A-Softmax出现之后，我想到了一个与NormFace相似的idea，即对特征也做归一化，但因工作中心放在实现和部署上，并没有深入进行思考，直至NormFace

### CosFace
$$\mathcal{L}=-\sum_{i=1}^{m}{\log{\frac{e^{s(\cos(\theta_{y_i})-m)}}{e^{s(\cos{\theta_{y_i}}-m)}+\sum_{j \ne y_i}{e^{s\cos(\theta_j)}}}}}$$

### InsightFace
$$\mathcal{L}=-\sum_{i=1}^{m}{\log{\frac{e^{s(\cos(\theta_{y_i}+m))}}{e^{s(\cos(\theta_{y_i}+m))}+\sum_{j \ne y_i}{e^{s\cos\theta_j}}}}}$$

## Implement
### Framework
实际部署使用了caffe和mxnet两个框架，主要是由于最早和最好的模型分别使用这两个框架训练

为了将公司的业务和算法分离，因此设计了一个三层模型。顶层为应用层，由深圳部门实现；中间层提供应用层使用，将公司业务逻辑的算法部分抽象出来，如人脸检测、人脸识别、人脸图像超分辨，应用层只用关心调用何种功能，而不用关心算法的具体实现，接口经协商确定后几乎不会改变；底层为算法层，即AlphaPic，为中间层提供具体的算法实现，算法层不关心业务逻辑。


### Accuracy
鉴于人力和物力的差异，公司并能不能够花足够的时间对识别模型进行深入的研究，最后保存了4个模型进行部署。

最早期的模型（源自Face Verification项目）由于已经应用于移动天眼项目，准确率请参考项目文档；

后续复现A-Softmax，使用resnet20作为backbone，并尝试进行剪枝以优化速度，最后在LFW上Acc=99.18，模型大小为4.39M，推理速度23ms（Intel Core i5-4590）

最后提供的是InsightFace模型

在公司测试集上的Acc仅有参考作用，在实际使用过程中，并没有凸显出微调的作用，甚至某些情况下更差，因此最后的版本是开源版本。

# Face Super Resolution
