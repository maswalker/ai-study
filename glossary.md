# 术语表（中英对照 + 大白话）

读文档遇到不懂的词先来这里查。"阶段"一列表示在哪个阶段会深入学习它。
欢迎你自己往里面补充新词。

---

## 基础概念

| 术语 | 英文 | 大白话解释 | 阶段 |
|------|------|-----------|------|
| 模型 | Model | 一个带有大量可调参数的数学函数，输入数据，输出预测 | 0 |
| 参数 / 权重 | Parameters / Weights | 模型里需要"学"出来的数字。64M = 6400 万个数字 | 0 |
| 训练 | Training | 不断调整参数，让模型预测得越来越准的过程 | 2 |
| 推理 | Inference | 用训练好的模型做预测（比如和它聊天） | 0 |
| 损失函数 | Loss | 衡量"预测"和"正确答案"差多远的一个数，越小越好 | 2 |
| 梯度 | Gradient | 每个参数"往哪个方向调、调多少能让 loss 下降"的指示 | 1, 2 |
| 梯度下降 | Gradient Descent | 沿着梯度的反方向一小步一小步地调参数 | 2 |
| 反向传播 | Backpropagation | 用链式法则从输出往输入方向，算出每个参数的梯度 | 2 |
| 学习率 | Learning Rate (lr) | 每次调参数的步子大小 | 2 |
| 优化器 | Optimizer | 决定"怎么根据梯度更新参数"的算法，如 SGD、Adam、AdamW | 2, 3 |
| 批 / 轮 / 步 | Batch / Epoch / Step | batch = 一次喂给模型的一组样本；epoch = 把全部数据过一遍；step = 一次参数更新 | 2 |
| 过拟合 | Overfitting | 模型把训练数据"背下来"了，但没学到规律，遇到新数据就不行 | 2 |
| 激活函数 | Activation Function | 给网络加入非线性的函数（ReLU、GELU、SiLU 等），没有它多层网络等于一层 | 2 |
| 张量 | Tensor | 多维数组，PyTorch 里的基本数据结构 | 3 |
| 前向 / 反向 | Forward / Backward | 前向 = 从输入算到输出；反向 = 从 loss 往回算梯度 | 2, 3 |

## 语言模型

| 术语 | 英文 | 大白话解释 | 阶段 |
|------|------|-----------|------|
| 大语言模型 | LLM (Large Language Model) | 用海量文本训练、能理解和生成语言的大型神经网络 | 0 |
| 词元 | Token | 模型处理文本的最小单位，可能是一个字、半个词或一个词 | 0, 4 |
| 词表 | Vocabulary (vocab) | 模型认识的所有 token 的集合。minimind 词表大小 6400 | 4 |
| 分词器 | Tokenizer | 把文字切成 token 并转成数字 id 的工具 | 4 |
| BPE | Byte Pair Encoding | 一种分词算法：不断把最常一起出现的两个片段合并成新 token | 4 |
| 词嵌入 | Embedding | 把每个 token id 变成一个向量（一串数字），让模型能计算"意思" | 4 |
| 下一个词预测 | Next-token Prediction | 语言模型的核心任务：根据前文预测下一个 token | 0, 4 |
| logits | Logits | 模型输出的"原始分数"，经过 softmax 后变成概率 | 4 |
| softmax | Softmax | 把一组任意分数变成加起来等于 1 的概率 | 1 |
| 交叉熵 | Cross Entropy | 语言模型的 loss：正确 token 的预测概率越低，loss 越大 | 1, 4 |
| 困惑度 | Perplexity (PPL) | exp(loss)，可以理解为"模型在多少个选项里犹豫"，越低越好 | 4 |
| 上下文长度 | Context Length | 模型一次能看到的最大 token 数 | 4 |
| 温度 | Temperature | 生成时的随机程度：低温保守稳定，高温天马行空 | 0, 4 |
| top-k / top-p | Top-k / Top-p (Nucleus) Sampling | 只从概率最高的 k 个 / 累计概率达到 p 的 token 里采样 | 4 |

## Transformer 结构

| 术语 | 英文 | 大白话解释 | 阶段 |
|------|------|-----------|------|
| Transformer | Transformer | 2017 年提出的神经网络结构，几乎所有现代 LLM 的基础 | 4 |
| 仅解码器 | Decoder-only | GPT 类模型的结构：只能看到前文，逐个生成 | 4 |
| 自注意力 | Self-Attention | 让每个 token "看看"其他 token，决定从谁那里获取多少信息 | 4 |
| Q / K / V | Query / Key / Value | 查询（我想找什么）/ 键（我有什么标签）/ 值（我的实际内容） | 4 |
| 多头注意力 | Multi-Head Attention (MHA) | 同时做多组注意力，每组关注不同方面 | 4 |
| 因果掩码 | Causal Mask | 遮住"未来"的 token，防止模型训练时偷看答案 | 4 |
| 前馈网络 | FFN / MLP | 注意力之后对每个 token 单独做的两三层全连接计算 | 4 |
| 残差连接 | Residual Connection | `x = x + f(x)`，给信息和梯度留一条"直通车" | 3, 4 |
| 层归一化 | LayerNorm | 把每个 token 的向量调整到稳定的数值范围，让训练更稳 | 3, 4 |
| RMSNorm | Root Mean Square Norm | LayerNorm 的简化版，只除以均方根、不减均值。minimind 使用 | 5 |
| Pre-Norm | Pre-Normalization | 先归一化再进注意力/FFN，比 Post-Norm 训练更稳定 | 4, 5 |
| 位置编码 | Positional Encoding | 告诉模型每个 token 在第几个位置（注意力本身不知道顺序） | 4 |
| RoPE | Rotary Position Embedding | 旋转位置编码：用旋转角度表示位置，天然表达相对距离 | 5 |
| YaRN | YaRN | 一种让 RoPE 支持比训练时更长文本的外推方法 | 5 |
| GQA | Grouped-Query Attention | 多个 Q 头共享一组 K/V 头，减少显存 | 5 |
| SwiGLU | SwiGLU | 一种带"门控"的 FFN，效果比普通 FFN 好 | 5 |
| MoE | Mixture of Experts | 混合专家：有多个 FFN"专家"，每个 token 只激活其中几个 | 5 |
| KV 缓存 | KV Cache | 生成时把之前算过的 K、V 存起来，避免重复计算 | 4, 5 |
| 权重共享 | Tied Embeddings | 输入的 embedding 和输出层共用一个矩阵，省参数 | 5 |

## 训练流程

| 术语 | 英文 | 大白话解释 | 阶段 |
|------|------|-----------|------|
| 预训练 | Pretraining | 用海量文本学"续写"，获得语言能力和知识 | 0, 5 |
| 基座模型 | Base Model | 只做过预训练的模型，只会续写，不会对话 | 5 |
| 指令微调 | SFT (Supervised Fine-Tuning) | 用"问题-回答"数据训练，教模型按对话格式回答 | 0, 5 |
| 对话模板 | Chat Template | 把多轮对话拼成一个字符串的固定格式（含特殊 token） | 5 |
| 全参数微调 | Full Fine-tuning | 训练时更新模型所有参数 | 5 |
| LoRA | Low-Rank Adaptation | 冻结原模型，只训练旁边加的两个小矩阵，省显存 | 6 |
| 知识蒸馏 | Knowledge Distillation | 让小模型模仿大模型的输出分布 | 6 |
| 混合精度 | Mixed Precision | 用 16 位浮点数（fp16 / bf16）计算来省显存、提速 | 3 |
| 梯度累积 | Gradient Accumulation | 多个小 batch 的梯度加起来再更新一次，模拟大 batch | 3 |
| 梯度裁剪 | Gradient Clipping | 梯度太大时按比例缩小，防止训练崩掉 | 3 |
| 学习率调度 | LR Schedule | 训练过程中动态调整学习率，如余弦退火（cosine） | 3 |
| 检查点 | Checkpoint | 训练过程中保存的模型和优化器状态，用于续训 | 3 |
| 分布式数据并行 | DDP | 多张 GPU 各算一部分数据，再同步梯度 | 7 |

## 对齐与强化学习

| 术语 | 英文 | 大白话解释 | 阶段 |
|------|------|-----------|------|
| 对齐 | Alignment | 让模型的行为符合人类期望（有用、诚实、无害） | 6 |
| 强化学习 | RL (Reinforcement Learning) | 通过"试错 + 奖励"来学习的方法 | 6 |
| 策略 | Policy | RL 中决定"做什么动作"的模型；在 LLM 里就是语言模型本身 | 6 |
| 奖励 / 奖励模型 | Reward / Reward Model | 给模型回答打分的信号 / 专门用来打分的模型 | 6 |
| 优势函数 | Advantage | 这个动作比"平均水平"好多少 | 6 |
| 策略梯度 | Policy Gradient | 让高奖励的动作概率变大、低奖励的变小 | 6 |
| KL 散度 | KL Divergence | 衡量两个概率分布差多少；RL 中用来防止模型跑偏太远 | 1, 6 |
| RLHF | RL from Human Feedback | 用人类偏好反馈做强化学习 | 6 |
| RLAIF | RL from AI Feedback | 用 AI（奖励模型/规则）反馈做强化学习 | 6 |
| PPO | Proximal Policy Optimization | 经典 RL 算法，限制每次更新幅度 | 6 |
| DPO | Direct Preference Optimization | 直接用"好/坏回答对"训练，无需奖励模型 | 6 |
| GRPO | Group Relative Policy Optimization | DeepSeek 提出：同题多答，组内比较算优势，无需 critic | 6 |
| 奖励作弊 | Reward Hacking | 模型钻奖励规则的空子，分数高但并没有真的变好 | 6 |

## 工程与部署

| 术语 | 英文 | 大白话解释 | 阶段 |
|------|------|-----------|------|
| MPS | Metal Performance Shaders | PyTorch 在 Apple 芯片上用 GPU 加速的后端 | 0 |
| CUDA | CUDA | NVIDIA 的 GPU 编程平台 | 7 |
| 量化 | Quantization | 用更少的位数（如 4 bit）存参数，模型变小变快，精度略降 | 7 |
| GGUF | GGUF | llama.cpp 使用的模型文件格式 | 7 |
| FlashAttention | FlashAttention | 减少显存读写的注意力加速算法 | 7 |
| 工具调用 | Tool Calling / Function Calling | 模型输出结构化指令去调用外部工具（搜索、计算器等） | 6, 7 |
| 智能体 | Agent | 能自主规划、调用工具、多步完成任务的 LLM 系统 | 7 |
