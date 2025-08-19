# everyday plan

| 周数 | 日期范围 | 主题 | 学习内容 | 实践任务 | 资源推荐 |
|------|------ --|------|---------|---------|---------|
| 1    | 第1周          | 数学基础：线性代数与概率     | 向量/矩阵运算、条件概率、贝叶斯、分布                                    | 用 NumPy 实现矩阵乘法、PCA 简化版                                        | 《深度学习》Ch2-3, Khan Academy |
| 2    | 第2周          | 微积分与优化                 | 导数、梯度、链式法则、梯度下降                                           | 手动实现线性回归（无框架）                                               | 同上 |
| 3    | 第3周          | Python 与数据处理            | Python 基础、NumPy、Pandas、Matplotlib                                   | 加载 CSV 数据并可视化鸢尾花数据集                                        | 《Python数据科学手册》 |
| 4    | 第4周          | 经典机器学习算法             | 线性回归、逻辑回归、KNN、决策树、模型评估                                | 使用 scikit-learn 完成泰坦尼克生存预测                                   | Kaggle 入门赛 |
| 5    | 第5周          | 神经网络基础                 | MLP、激活函数、前向/反向传播                                             | 用 NumPy 实现两层神经网络分类                                            | D2L 中文版 Ch3 |
| 6    | 第6周          | PyTorch 入门                 | 张量、autograd、nn.Module、训练循环                                      | 用 PyTorch 实现 MLP 在 MNIST 上训练                                      | PyTorch 官方教程 |
| 7    | 第7周          | CNN 基础                     | 卷积、池化、特征图、LeNet/AlexNet                                        | 实现 CNN 分类 CIFAR-10                                                   | CS231n 笔记 |
| 8    | 第8周          | 现代 CNN 架构                | VGG、Inception、ResNet、BatchNorm                                        | 复现 ResNet-18 并训练                                                    | ResNet 论文 |
| 9    | 第9周          | RNN 与 LSTM                  | RNN 结构、LSTM 门机制、梯度问题                                          | 用 LSTM 生成字符级文本                                                   | D2L Ch8 |
| 10   | 第10周         | NLP 基础                     | 词嵌入、Word2Vec、文本预处理                                             | 使用预训练词向量做文本分类                                               | word2vec Google |
| 11   | 第11周         | Encoder-Decoder              | Seq2Seq 模型结构、应用场景                                               | 实现英文→法文翻译模型（小数据集）                                        | TensorFlow Seq2Seq 教程 |
| 12   | 第12周         | Attention 机制               | Bahdanau/Luong Attention、注意力权重计算                                 | 在 Seq2Seq 中加入 Attention 提升效果                                     | "Neural Machine Translation by Jointly Learning to Align and Translate" |
| 13   | 第13周         | Self-Attention               | QKV 计算、缩放点积、多头注意力                                           | 手动实现 Self-Attention 层                                               | "Attention is All You Need" |
| 14   | 第14周         | Transformer 架构             | Encoder/Decoder 块、位置编码、残差连接                                   | 复现 Transformer Encoder 部分                                            | The Annotated Transformer |
| 15   | 第15周         | 预训练模型：BERT & GPT       | MLM、NSP、自回归生成、Hugging Face 库                                    | 使用 `transformers` 微调 BERT 做文本分类                                 | Hugging Face 官网 |
| 16   | 第16周         | 微调与 Prompt Engineering    | Fine-tuning 流程、Zero-shot 学习                                         | 构建情感分析或文本生成 Demo                                              | Hugging Face Course |
| 17   | 第17周         | Deconvolution 与上采样       | 转置卷积原理、与插值对比、U-Net 中应用                                   | 实现转置卷积层并可视化输出                                               | CS231n ConvNets |
| 18   | 第18周         | 生成模型：VAE & GAN          | 自编码器、隐空间、生成对抗网络                                           | 用 VAE 生成 MNIST 图像                                                   | D2L Ch14 |
| 19   | 第19周         | 位置编码与 ViT               | 正弦 vs 可学习位置编码、Vision Transformer                               | 将 ResNet 特征送入 Transformer 分类                                      | ViT 论文 |
| 20   | 第20周         | 模型压缩与部署               | 知识蒸馏、量化、剪枝、ONNX 导出                                          | 将训练模型导出为 ONNX 格式                                               | ONNX 官网 |
| 21   | 第21周         | 项目实战1：图像分割          | U-Net 结构、跳跃连接、医学图像分割                                       | 使用 U-Net + ResNet 编码器完成分割任务                                   | Kaggle: Carvana Image Masking |
| 22   | 第22周         | 项目实战2：NLP 应用          | 问答系统、文本摘要                                                       | 基于 BERT/T5 实现问答或摘要系统                                          | SQuAD 数据集 |
| 23   | 第23周         | 论文精读                     | ResNet、Transformer、BERT 三篇经典论文                                   | 写读书笔记 + 手绘结构图 + 代码片段解析                                   | arXiv.org |
| 24   | 第24周         | 拓展方向探索                 | 扩散模型、大语言模型、多模态                                             | 撰写技术博客或PPT总结一个前沿方向                                        | Hugging Face Blog |
