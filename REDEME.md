# NLP 自然语言处理学习项目

## 📋 项目概述

本项目是一个系统性的自然语言处理（NLP）学习与实践项目，涵盖了从基础分词到现代 Transformer 架构的完整技术栈。项目通过 10 个阶段（Day01-Day10）循序渐进地展示了 NLP 核心技术的理论与实践实现。

**项目路径**: `C:\Users\Administrator\PycharmProjects\PythonProject\NLP`

---

## 🎯 项目目标

- 掌握中文文本处理基础技术（分词、词性标注）
- 理解词向量表示方法（One-Hot、Word2Vec、Embedding）
- 实现经典序列模型（RNN、LSTM、GRU）
- 构建 Seq2Seq 机器翻译系统
- 深入理解 Transformer 架构
- 应用预训练模型解决实际问题（BERT、FastText）

---

## 📚 技术栈

### 核心框架
- **深度学习框架**: PyTorch, TensorFlow/Keras
- **NLP 工具库**: jieba, transformers, fasttext, datasets
- **数据处理**: pandas, numpy, seaborn, matplotlib
- **可视化工具**: tqdm, matplotlib.pyplot

### 主要技术
- 文本预处理与分词
- 词向量表示学习
- 循环神经网络（RNN/LSTM/GRU）
- 注意力机制（Attention）
- Transformer 架构
- 预训练语言模型（BERT）
- 快速文本分类（FastText）

---

## 🗂️ 项目结构
NLP/ 
├── Day01/ # 中文分词基础 
├── Day02/ # 词向量表示 
├── Day03/ # 数据探索与分析 
├── Day04/ # RNN 基础理论 
├── Day05/ # RNN 分类实战 
├── Day06/ # Seq2Seq 机器翻译 
├── Day07/ # Transformer 架构 
├── Day08/ # FastText 文本分类 
├── Day09/ # 预训练模型 Pipeline 
└── Day10/ # BERT 微调实战
---

## 📖 详细模块说明

### Day01: 中文分词基础

**核心内容**: 使用 jieba 进行中文文本分词

**主要功能**:
1. **精确模式分词** - `dm1_jieba()`
   - 适合文本分析，精确切分词语
   
2. **全模式分词** - `dm2_jieba()`
   - 扫描所有可能的词语，速度较快
   
3. **搜索引擎模式** - `dm3_jieba()`
   - 在精确模式基础上对长词再次切分
   
4. **自定义词典** - `dm4_jieba()`
   - 加载用户词典，优先识别专业术语
   - 示例词典: `userdict.txt`
   
5. **词性标注** - `dm5_jieba()`
   - 使用 `jieba.posseg` 进行词性标注

---

### Day02: 词向量表示

**核心内容**: 从 One-Hot 到 Word2Vec 的词表示方法

**主要功能**:

1. **One-Hot 编码** - `01_use_onehot.py`
   - 使用 Keras Tokenizer 构建词汇表
   - 将词语转换为稀疏向量
   - 保存/加载训练好的分词器（joblib）
   
2. **Word2Vec 词向量** - `02_word2vec.py`
   - 基于 gensim 训练词向量
   - 支持 CBOW 和 Skip-gram 模型
   
3. **Embedding 层** - `03_embedding.py`
   - PyTorch Embedding 层的使用
   - 可学习的稠密向量表示

**关键技术点**:
- 词汇表索引映射（word_index / index_word）
- 向量维度选择
- 模型持久化存储

---

### Day03: 数据探索与分析

**核心内容**: 文本数据的统计分析

**主要功能**:

1. **标签分布可视化** - `dm01_labels.py`
   - 使用 seaborn 绘制类别计数图
   - 分析训练集/验证集的标签平衡性
   
2. **文本长度统计** - `dm02_length.py`
   - 统计句子长度分布
   - 确定合适的最大序列长度
   
3. **词汇表构建** - `dm03_vocbs.py`
   - 提取高频词汇
   - 构建词频统计表
   
4. **词云生成** - `dm04_wordcloud.py`
   - 可视化文本关键词
   
5. **N-Gram 特征** - `dm05_N_gram.py`
   - 提取二元/三元语法特征
   - 序列填充与截断（padding/truncating）
---

### Day04: RNN 基础理论

**核心内容**: 循环神经网络的原理与实现

**主要功能**:

1. **基础 RNN** - `dm01_RNN.py`
   - 理解 RNN 输入输出维度
   - 参数说明: input_size, hidden_size, num_layers
   
2. **不同序列长度测试** - `dm2_rnn_for_len()`
   - 单步输入 vs 批量输入
   
3. **Batch First 模式** - `dm3_rnn_for_batch()`
   - 调整 batch 维度位置
   
4. **多层 RNN** - `dm4_rnn_for_numlayers()`
   - 堆叠多个 RNN 层

### Day04: 循环神经网络（RNN）基础与应用

**核心内容**: 三种经典序列模型的 PyTorch 实现与参数详解

**主要功能**:

1. **标准 RNN 模型** - `dm01_RNN.py`
   - **基础用法演示**：展示 `nn.RNN` 的核心参数配置
     - `input_size`: 词嵌入维度
     - `hidden_size`: 隐藏层输出维度
     - `num_layers`: RNN 层数
   - **输入张量形状理解**：
     - 默认模式 `(sequence_len, batch_size, input_size)`
     - `batch_first=True` 模式 `(batch_size, sequence_len, input_size)`
   - **时间步处理**：对比一次性送入完整序列 vs 逐 token 输入的差异
   - **多层 RNN**：演示双层 RNN 的初始状态 `h0` 维度配置

2. **LSTM 长短期记忆网络** - `dm02_LSTM.py`
   - **双状态机制**：LSTM 同时维护隐藏状态 `h0` 和细胞状态 `c0`
   - **输入格式**：`(h0, c0)` 元组形式传递初始状态
   - **输出解析**：返回 `(output, (h_n, c_n))` 三元组结构
   - **参数配置**：支持多层堆叠，每层需独立初始化状态

3. **GRU 门控循环单元** - `dm03_GRU.py`
   - **简化架构**：相比 LSTM 仅维护单一隐藏状态 `h0`
   - **接口一致性**：使用方式与标准 RNN 完全相同
   - **计算效率**：参数量更少，训练速度更快

4. **辅助工具** - `review.py`
   - 张量形状查询方法演示（`.shape` vs `.size()`）

**技术要点**:
- 掌握三种循环神经网络的输入/输出张量维度规范
- 理解 `batch_first` 参数对数据排列的影响
- 区分 LSTM（双状态）与 GRU/RNN（单状态）的状态管理差异
- 为后续序列分类、机器翻译等任务奠定模型基础


### Day05: 基于循环神经网络的姓名分类任务

**核心内容**: 使用 RNN、LSTM、GRU 三种模型实现人名国籍分类，并进行性能对比分析

**主要功能**:

1. **完整分类流程实现** - `dm01_RNN_classfication.py`
   - **数据预处理**：
     - 构建包含 18 个国家类别的数据集（Korean、Chinese、English 等）
     - 自定义 `NameDataset` 类继承 PyTorch `Dataset`
     - One-hot 编码：将人名字符转换为 57 维向量（字母 + 标点符号）
     - 使用 `DataLoader` 实现批量数据加载（batch_size=1）
   
   - **三种模型架构**：
     - **NameRNN**：标准 RNN + Linear 输出层 + LogSoftmax
     - **NameLSTM**：LSTM 双状态机制（h0, c0）+ 相同输出结构
     - **NameGRU**：GRU 单状态机制 + 相同输出结构
     - 统一配置：input_size=57, hidden_size=128, output_size=18
   
   - **训练策略**：
     - 损失函数：负对数似然损失 `NLLLoss`
     - 优化器：Adam（学习率 1e-3）
     - 语义提取：取序列最后时间步的隐藏状态代表整个人名
     - 日志记录：每 100 样本记录平均损失/准确率，每 2000 步打印训练状态
   
   - **模型评估与可视化**：
     - 保存训练结果至 JSON 文件（损失曲线、准确率曲线）
     - 绘制三种模型的损失对比图和准确率对比图
     - 模型参数持久化存储为 `.pth` 文件
   
   - **预测功能**：
     - 实现 Top-K 预测（k=3），返回概率最高的三个国家类别
     - 支持单个姓名的实时分类预测

2. **辅助工具** - `review.py`
   - JSON 序列化/反序列化操作演示
   - `torch.topk` 用法示例（获取前 K 个最大值及其索引）
   - Softmax 激活函数测试

**技术要点**:
- 掌握字符级 One-hot 编码的实现方法
- 理解序列分类任务的建模思路（取最后时间步状态作为句子表示）
- 学会使用 PyTorch Dataset/DataLoader 构建自定义数据管道
- 对比 RNN/LSTM/GRU 在长序列依赖捕捉上的性能差异
- 掌握模型训练过程的监控指标记录与可视化方法
- 熟悉 Top-K 预测在实际应用中的价值


### Day06: 注意力机制与 Seq2Seq 机器翻译

**核心内容**: 实现基于注意力机制的编码器-解码器架构，完成英法机器翻译任务

**主要功能**:

1. **注意力机制基础实现** - `dm01_attention.py`
   - **自定义注意力模块**：
     - `MyAttention`：带维度升降的注意力计算
       - 第一步：Q 与 K 拼接 → Linear → Softmax 得到注意力权重
       - 第二步：权重与 V 矩阵相乘得到上下文向量
       - 第三步：上下文向量与 Q 拼接 → Linear 输出最终结果
     - `OrignAttention`：简化版注意力（保持原始维度）
   - **张量维度理解**：
     - Query: `[1, 1, 32]`（当前时间步查询）
     - Key: `[1, 1, 32]`（编码器隐藏状态）
     - Value: `[1, 32, 64]`（编码器所有时间步输出）

2. **Seq2Seq 机器翻译系统** - `dm02_seq2seq.py`
   - **数据预处理**：
     - 文本清洗：正则表达式去除特殊字符、统一小写、标点符号分离
     - 双语词典构建：英文/法文单词 ↔ 索引映射（含 SOS/EOS 标记）
     - 自定义 `MyDataset`：将句子转换为索引序列并添加 EOS 标记
     - DataLoader：批量加载英法句对
   
   - **编码器架构** - `EncoderGRU`：
     - Embedding 层：将单词索引转换为 256 维词向量
     - GRU 层：`batch_first=True`，输入 `[batch, seq_len, embedding]`
     - 输出：所有时间步隐藏状态 + 最后时间步隐藏状态
   
   - **无注意力解码器** - `DecoderGRU`：
     - 逐 token 解码：每次输入一个单词索引
     - Embedding + ReLU 激活防止过拟合
     - GRU 处理 + Linear 输出层 + LogSoftmax
     - Teacher Forcing：以 0.5 概率使用真实标签作为下一步输入
   
   - **带注意力解码器** - `AttentionDecoder`：
     - **注意力计算流程**：
       1. 当前输入嵌入 + Dropout 防过拟合
       2. 与编码器隐藏状态拼接 → Linear → Softmax 得到注意力权重 `[1, 1, max_length]`
       3. 权重与编码器输出加权求和得到上下文向量
       4. 上下文向量与当前输入拼接 → Linear → ReLU
       5. GRU 处理 + 输出层生成下一个单词概率分布
     - **中间语义张量 C**：填充至固定长度 `max_length=10`，不足部分补零
   
   - **训练策略**：
     - 优化器：Adam（学习率 1e-4），编码器和解码器独立优化
     - 损失函数：负对数似然损失 `NLLLoss`
     - Teacher Forcing：随机选择是否使用真实标签作为下一步输入
     - 日志记录：每 1000 步打印平均损失，每 100 步保存损失曲线数据
     - 模型保存：分别存储编码器和解码器参数
   
   - **预测与可视化**：
     - `seq2seq_evaluate`：贪婪搜索生成翻译结果（Top-1 选择）
     - `show_attention`：绘制注意力权重热力图，展示源语言与目标语言的对齐关系
     - 支持多句批量测试，对比预测结果与真实翻译

3. **辅助工具** - `review.py`
   - 正则表达式文本清洗技巧（`re.sub` 用法）
   - 字典构建与索引映射
   - PyTorch 张量切片操作演示

**技术要点**:
- 掌握注意力机制的三种核心计算步骤（Q-K 相似度 → 权重归一化 → 加权求和）
- 理解 Encoder-Decoder 架构在序列到序列任务中的应用
- 学会使用 Teacher Forcing 加速训练收敛
- 掌握机器翻译中的数据预处理流程（清洗、分词、数值化）
- 能够通过注意力权重可视化分析模型关注的源语言位置
- 区分训练阶段（Teacher Forcing）与推理阶段（自回归生成）的差异


### Day07: Transformer 架构详解与实现

**核心内容**: 从零实现完整的 Transformer 模型，包含编码器、解码器及各核心组件

**主要功能**:

1. **输入表示层** - `transform_input.py`
   - **词嵌入层** - `Embeddings`：
     - 将单词索引转换为 512 维词向量
     - 缩放因子：乘以 `√d_model` 使嵌入符合标准正态分布，增强 Embedding 影响
   
   - **位置编码层** - `PostionalEncoding`：
     - 使用正弦/余弦函数生成固定位置编码（无需学习）
     - 公式实现：偶数位用 sin，奇数位用 cos
     - 注册为 buffer：不参与训练但随模型保存加载
     - Dropout 正则化防止过拟合
     - 支持动态句子长度截取

2. **编码器模块** - `transform_encoder.py`
   - **注意力计算公式** - `attention`：
     - `scores = Q·K^T / √d_k`（缩放点积注意力）
     - Mask 填充：将无效位置设为 `-1e9`（Softmax 后趋近于 0）
     - Dropout 应用于注意力权重
   
   - **多头注意力机制** - `MultiHeadAttention`：
     - 8 个头并行计算，每个头维度 `d_k = 512/8 = 64`
     - 4 个 Linear 层：Q、K、V 各一个，输出拼接后一个
     - 维度变换：`[batch, seq, d_model]` → `[batch, head, seq, d_k]`
     - 多头拼接后通过最终 Linear 层融合信息
   
   - **前馈神经网络** - `FeedForward`：
     - 两层全连接：`Linear(d_model→d_ff) → ReLU → Dropout → Linear(d_ff→d_model)`
     - 扩展维度：512 → 1024 → 512，增强非线性表达能力
   
   - **层规范化** - `layerNorm`：
     - 可学习参数 `a`（增益）和 `b`（偏置）
     - 对最后一个维度进行标准化：`(x - mean) / (std + eps)`
   
   - **残差连接** - `SubLayerConnection`：
     - Post-Norm 结构：`x + Dropout(LayerNorm(sublayer(x)))`
     - Pre-Norm 备选：`x + Dropout(sublayer(LayerNorm(x)))`
     - 缓解梯度消失，促进深层网络训练
   
   - **编码器层** - `EncoderLayer`：
     - 第一个子层：多头自注意力（Q=K=V）+ Add & Norm
     - 第二个子层：前馈神经网络 + Add & Norm
   
   - **完整编码器** - `Encoder`：
     - 堆叠 6 层 EncoderLayer
     - 最终 LayerNorm 输出

3. **解码器模块** - `transform_decoder.py`
   - **解码器层** - `DecoderLayer`（三个子层）：
     - **第一子层**：多头自注意力（Masked Self-Attention）
       - 使用 `target_mask`（下三角矩阵）防止看到未来信息
       - Q=K=V=当前解码器输入
     
     - **第二子层**：多头交叉注意力（Cross-Attention）
       - Q=解码器输出，K=V=编码器输出
       - 使用 `score_mask` 进行 Padding 掩码
       - 实现编码器与解码器的信息交互
     
     - **第三子层**：前馈神经网络 + Add & Norm
   
   - **完整解码器** - `Decoder`：
     - 堆叠 6 层 DecoderLayer
     - 最终 LayerNorm 输出

4. **输出层** - `transform_generator.py`
   - **Generator**：
     - Linear 层：512 → vocab_size（目标语言词表大小）
     - LogSoftmax：输出对数概率分布
     - 便于直接计算 NLLLoss

5. **完整 Transformer 模型** - `transform.py`
   - **EncoderDecoder 架构**：
     - 源语言输入：Word Embedding + Position Encoding
     - 目标语言输入：Word Embedding + Position Encoding
     - 编码器处理：接收 source + source_mask1（自注意力掩码）
     - 解码器处理：接收 encoder_output + target_mask（自注意力掩码）+ source_mask2（交叉注意力掩码）
     - 生成器输出：decoder_output → 词汇概率分布
   
   - **三种掩码类型**：
     - `source_mask1`：`[head, source_seq_len, source_seq_len]` 编码器自注意力掩码
     - `source_mask2`：`[head, target_seq_len, source_seq_len]` 解码器交叉注意力掩码
     - `target_mask`：`[head, target_seq_len, target_seq_len]` 解码器自注意力掩码（下三角）

6. **辅助工具** - `review.py`
   - 基础测试代码

**技术要点**:
- 掌握 Transformer 的核心创新：自注意力机制替代循环结构
- 理解多头注意力的并行计算与维度变换逻辑
- 学会使用 Mask 机制处理 Padding 和未来信息泄露问题
- 掌握 Residual Connection + LayerNorm 的稳定训练技巧
- 区分编码器自注意力（无掩码）与解码器自注意力（因果掩码）
- 理解位置编码的数学原理及其在序列建模中的作用
- 熟悉 Transformer 的模块化设计思想（EncoderLayer/DecoderLayer 复用）

### Day08: FastText 文本分类与超参数调优

**核心内容**: 使用 FastText 库实现高效的多标签文本分类，探索数据预处理与超参数优化策略

**主要功能**:

1. **FastText 基础使用**
   - **数据格式规范**：
     - 标签格式采用 `__label__category` 形式（每行可包含多个标签）
     - 示例：一条烹饪问题可以同时属于 "sauce" 和 "cheese" 两个类别
     - 数据集划分：训练集、验证集分别存储为独立文件
   
   - **基础模型训练流程**：
     - 第一步：调用监督学习接口训练分类模型
     - 第二步：将训练好的模型保存为二进制文件
     - 第三步：从磁盘加载预训练模型
     - 第四步：对新文本进行预测，返回标签及置信度概率
     - 第五步：在验证集上评估模型性能，获取精确率和召回率指标

2. **数据预处理对比实验**
   - **原始数据特征**：
     - 保留大小写混合
     - 标点符号紧贴单词无空格
     - 特殊字符未规范化处理
   
   - **预处理数据优化**：
     - 统一转换为小写字母
     - 标点符号前后添加空格实现分离
     - 特殊字符（如撇号）进行规范化处理
     - 提升模型对词汇的泛化能力
   
   - **效果对比**：通过验证集的精确率和召回率指标，量化评估数据预处理带来的性能提升

3. **超参数调优策略**
   - **增加训练轮数（epoch）**：
     - 默认训练轮数为 5 轮
     - 增加到 25 轮可提升模型收敛度
     - 避免欠拟合，充分学习数据特征
   
   - **调整学习率（lr）**：
     - 默认学习率为 0.1
     - 提高至 1.0 可加速收敛速度
     - 需权衡稳定性与训练效率
   
   - **引入 N-gram 特征（wordNgrams）**：
     - 捕获二元语法特征（如 "cheese sauce" 短语）
     - 增强对词组级别语义的理解
     - 弥补单一词向量无法捕捉上下文关系的缺陷
   
   - **修改损失函数（loss）**：
     - **softmax**（默认）：适用于单标签分类任务
     - **hs**（Hierarchical Softmax）：层次化 Softmax，加速大规模类别训练，降低计算复杂度
     - **ova**（One-vs-All）：一对多策略，更适合多标签分类场景，每个类别独立判断

4. **自动超参数调优**
   - **autotune 自动化功能**：
     - 自动搜索最优超参数组合（epoch、lr、dim、wordNgrams 等）
     - 指定验证集文件用于实时评估不同参数配置
     - 设置调优时长限制（如 600 秒 = 10 分钟）
     - 节省人工试错成本，快速找到最佳配置

5. **多标签预测配置**
   - **Top-K 预测策略**：
     - 设置 k=-1 返回所有可能的标签
     - 通过 threshold 参数过滤低概率标签（如阈值 0.5）
     - 返回符合条件的多个标签及其对应概率值
     - 灵活控制预测结果的粒度与准确性平衡

6. **模型文件说明**
   - 基于原始数据训练的模型文件体积较大（约 6.2 MB）
   - 基于预处理数据训练的优化模型体积更小（约 3.9 MB）
   - 文件大小差异反映词汇表规模和嵌入矩阵大小的变化
   - 预处理后的模型通常具有更高的压缩率和更好的泛化能力

**技术要点**:
- 掌握 FastText 的监督学习 API 使用方法和工作流程
- 理解文本预处理对分类性能的重要影响（小写化、标点分离、特殊字符规范化）
- 学会通过调整 epoch、lr、wordNgrams 等关键超参数优化模型表现
- 区分不同损失函数的适用场景（softmax 适合单标签，ova 适合多标签，hs 适合大规模类别）
- 利用 autotune 自动化超参数搜索功能节省人工调参时间和成本
- 支持多标签分类任务，通过阈值参数灵活控制预测标签数量和质量
- FastText 的核心优势：训练速度快、支持子词信息捕获、内存占用低、适合大规模文本分类场景
- 数据质量决定模型上限，预处理是提升性能的关键环节


### Day09: Hugging Face Transformers 预训练模型应用

**核心内容**: 使用 Hugging Face Transformers 库调用各类预训练模型，完成多种自然语言处理任务

**主要功能**:

1. **Pipeline 快速调用方式**
   - **情感分类任务**：
     - 直接加载中文情感分析预训练模型
     - 输入文本自动返回情感极性标签（正面/负面）及置信度
     - 适用于快速原型开发和简单分类场景
   
   - **特征提取任务**：
     - 使用 BERT 基础模型提取文本语义向量
     - 输出包含特殊标记 CLS 和 SEP
     - 返回每个 token 的上下文感知嵌入表示
     - 可用于下游任务的特征输入
   
   - **完形填空任务**：
     - 基于掩码语言模型（MLM）预测缺失词汇
     - 支持多位置同时填空的迭代推理
     - 返回候选词列表及其概率排序
     - 可用于文本补全和语言生成
   
   - **机器阅读理解任务**：
     - 给定上下文和问题，自动定位答案片段
     - 支持批量问题输入
     - 返回答案在原文中的起止位置及文本内容
     - 适用于问答系统和信息抽取
   
   - **文本摘要任务**：
     - 使用序列到序列模型生成简洁摘要
     - 基于 BART 架构的蒸馏版本提升效率
     - 自动捕捉文章核心信息
     - 适用于新闻摘要和文档压缩
   
   - **命名实体识别任务**：
     - 识别文本中的人名、地名、机构名等实体
     - 基于 RoBERTa 微调的中文 NER 模型
     - 返回每个 token 的实体类型标注
     - 适用于信息抽取和知识图谱构建

2. **AutoModel 灵活调用方式**
   - **文本分类任务**：
     - 分步加载：分词器 + 分类模型
     - 文本编码：支持最大长度填充和截断策略
     - 返回 logits  logits 通过 argmax 获取最终类别索引
     - 需设置模型为评估模式并禁用梯度计算
   
   - **特征提取任务**：
     - 支持批量句子输入
     - 返回两种特征表示：
       - last_hidden_state：每个 token 的隐藏层输出
       - pooler_output：CLS token 经过池化层的句子级表示
     - 包含 input_ids、token_type_ids、attention_mask 三个关键张量
   
   - **完形填空任务**：
     - 定位 MASK 标记在序列中的位置索引
     - 从 logits 中提取对应位置的词汇概率分布
     - 通过 argmax 获取最可能的预测词
     - 将索引转换回可读的 token 字符串
   
   - **机器阅读理解任务**：
     - 使用 _encode_plus 方法同时编码问题和上下文
     - 模型输出 start_logits 和 end_logits 分别表示答案起始和结束位置的概率
     - 通过 argmax 找到答案边界索引
     - 从 input_ids 中切片并解码得到答案文本
   
   - **文本摘要任务**：
     - 使用 generate 方法进行自回归生成
     - 支持 beam search 等解码策略
     - 解码时跳过特殊标记并清理分词空格
     - 生成连贯流畅的摘要文本
   
   - **命名实体识别任务**：
     - 加载模型配置文件获取标签映射字典（id2label）
     - 遍历每个 token 对应的 logits 向量
     - 过滤特殊标记（如 CLS、SEP、PAD）
     - 将预测索引转换为实体类型标签
     - 输出 token-标签对列表

3. **特定模型专用调用**
   - **BertTokenizer + BertForMaskedLM 组合**：
     - 针对特定模型架构使用专用类而非 Auto 类
     - 提供更精细的控制和定制化能力
     - 适用于需要深入理解模型内部机制的场景
     - 与 Auto 类接口保持一致性，便于切换

4. **关键技术要点**
   - **分词器编码参数**：
     - padding：按最大长度补齐短句子
     - truncation：截断超过最大长度的句子
     - return_tensors：指定返回 PyTorch 张量格式
     - max_length：设定序列最大长度限制
   
   - **模型评估模式**：
     - 调用 eval() 方法关闭 Dropout 和 BatchNorm 的训练行为
     - 使用 torch.no_grad() 上下文管理器禁用梯度计算
     - 提升推理速度并减少内存占用
   
   - **输出数据结构**：
     - Pipeline 返回简化的字典或列表格式
     - AutoModel 返回包含多个属性的 ModelOutput 对象
     - logits：原始预测分数，需进一步处理
     - hidden_states：各层隐藏状态（可选输出）
   
   - **特殊标记处理**：
     - CLS（分类标记）：用于句子级任务的代表向量
     - SEP（分隔标记）：区分不同句子或段落
     - PAD（填充标记）：补齐序列至统一长度
     - MASK（掩码标记）：用于完形填空任务
   
   - **解码策略**：
     - skip_special_tokens：去除特殊标记使输出更清晰
     - clean_up_tokenization_spaces：优化标点符号和空格格式
     - generate 方法支持多种解码算法（贪婪搜索、束搜索等）

5. **应用场景总结**
   - **快速开发**：Pipeline API 适合原型验证和简单任务
   - **精细控制**：AutoModel API 适合复杂任务和自定义流程
   - **多任务支持**：覆盖分类、生成、抽取、标注等主流 NLP 任务
   - **中文优化**：使用中文预训练模型（如 chinese-bert-wwm）提升本土化效果
   - **迁移学习**：基于大规模语料预训练的模型在小样本场景下表现优异

**技术要点**:
- 掌握 Hugging Face Transformers 库的两种调用范式（Pipeline 快速调用 vs AutoModel 灵活调用）
- 理解不同 NLP 任务的输入输出格式和处理流程
- 学会正确配置分词器的编码参数以适应不同任务需求
- 熟悉模型评估模式和梯度禁用的最佳实践
- 能够处理特殊标记并进行有效的结果解码
- 了解各类预训练模型的适用场景和优势特点
- 具备将预训练模型集成到实际项目中的能力


### Day10: BERT 微调实战 - 三大经典任务

**核心内容**: 基于预训练 BERT 模型进行下游任务微调，实现文本分类、完形填空和下一句预测任务

**主要功能**:

1. **文本分类任务**
   - **数据准备**：
     - 使用 datasets 库加载 CSV 格式的训练集、验证集和测试集
     - 自定义 collate_fn 函数处理批次数据：分词、填充、截断至统一长度（200 token）
     - 返回 input_ids、attention_mask、token_type_ids 和标签张量
   
   - **模型架构**：
     - 冻结 BERT 预训练模型参数（不计算梯度，节省显存）
     - 在 BERT 输出层之上添加全连接分类头（768 维 → 2 类）
     - 使用 CLS token 的 pooler_output 作为句子级语义表示
   
   - **训练策略**：
     - 优化器：AdamW（学习率 1e-5，适合微调场景）
     - 损失函数：交叉熵损失
     - 批次大小：8，训练 3 个 epoch
     - 仅更新分类头参数，BERT 主干网络保持固定
   
   - **评估指标**：
     - 准确率：预测正确的样本数占总样本数的比例
     - 实时监控训练过程中的损失和准确率变化

2. **完形填空任务（掩码语言模型）**
   - **数据构造**：
     - 筛选文本长度大于 35 的样本确保有足够上下文
     - 固定选取第 16 个位置的 token 作为掩码目标
     - 将该位置替换为 [MASK] 标记，原始 token 作为标签
     - 最大序列长度设为 32 token
   
   - **模型设计**：
     - BERT 编码器输出 last_hidden_state（每个 token 的上下文表示）
     - 提取第 16 个位置的 768 维向量
     - 通过线性层映射到词汇表大小（21128 个候选词）
     - 同样冻结 BERT 参数，仅训练输出层
   
   - **训练细节**：
     - 训练 5 个 epoch，学习率使用 AdamW 默认值
     - 每 20 步打印累计准确率
     - 测试时对比预测词与真实词的解码结果

3. **下一句预测任务（NSP）**
   - **自定义数据集**：
     - 继承 Dataset 类实现句子对构建逻辑
     - 正样本：从同一文本中截取连续的两个片段（各 22 字符）
     - 负样本：随机选择另一条文本的第二片段与原第一片段配对
     - 标签：1 表示连续关系，0 表示非连续关系
   
   - **数据处理**：
     - collate_fn 接收句子对列表，使用分词器的双句子编码模式
     - 自动添加 [CLS]、[SEP] 分隔符并生成 token_type_ids
     - 填充至最大长度 50 token
   
   - **模型结构**：
     - 与文本分类任务类似，使用 CLS token 的 pooler_output
     - 二分类输出层（768 → 2）判断句子对是否连续
     - 学习率设为 3e-5（略高于分类任务）

4. **关键技术要点**
   - **迁移学习范式**：
     - 预训练阶段：在大规模语料上学习通用语言表示
     - 微调阶段：针对特定任务调整少量参数
     - 优势：小样本场景下仍能取得良好效果
   
   - **参数冻结策略**：
     - 通过 requires_grad_(False) 禁用 BERT 参数梯度计算
     - 大幅减少可训练参数量，加速训练并防止过拟合
     - 适用于数据量有限的下游任务
   
   - **分词器编码技巧**：
     - padding='max_length'：统一序列长度便于批处理
     - truncation=True：自动截断超长文本
     - return_tensors='pt'：直接返回 PyTorch 张量格式
     - 双句子编码：自动处理 [CLS] sentence1 [SEP] sentence2 [SEP] 格式
   
   - **数据加载优化**：
     - 使用 datasets 库的 filter 方法预处理筛选数据
     - 自定义 collate_fn 实现动态批处理和编码
     - drop_last=True 丢弃不足批次大小的剩余样本
   
   - **GPU 加速**：
     - 自动检测设备并将模型和数据移至 GPU
     - BERT 预训练模型也需同步转移到 GPU
     - 显著提升训练和推理速度

5. **应用场景总结**
   - **文本分类**：情感分析、主题分类、垃圾邮件检测等
   - **完形填空**：文本补全、语言理解能力评估、数据增强
   - **下一句预测**：对话系统连贯性判断、文档结构分析、自然语言推理

**技术要点**:
- 掌握 BERT 微调的标准流程和最佳实践
- 理解不同任务如何复用预训练模型的语义表示能力
- 学会根据任务特点设计合适的模型架构和数据预处理方案
- 熟悉参数冻结、学习率选择等微调关键技巧
- 能够使用 Hugging Face 生态工具高效完成 NLP 下游任务开发