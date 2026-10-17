# 学习进度打卡

开始日期：____年__月__日
每周计划投入：____ 小时

---

## 阶段 0：全局认知 + 环境搭建（1 周）
- [ ] 看完 Karpathy《Deep Dive into LLMs like ChatGPT》
- [ ] 看完 3B1B 深度学习第 5-7 集
- [ ] 玩过 Transformer Explainer 和 bbycroft 可视化
- [ ] 装好 uv + Python 3.11 + PyTorch，`mps.is_available()` 为 True
- [ ] 跑通 minimind 推理，和模型聊过天
- [ ] 完成自测，写下 `notes/00-全局认知.md`

## 阶段 1：Python + NumPy + 数学直觉（3-4 周）
- [ ] Python 基础（或确认已掌握）
- [ ] jsonl 统计脚本 + argparse
- [ ] NumPy 新手教程 + numpy-100 前 50 题
- [ ] 数值稳定的 softmax
- [ ] 3B1B 线性代数第 1-9 集
- [ ] 3B1B 微积分第 1-4 集
- [ ] 概率 / 交叉熵 / softmax 的直觉
- [ ] 完成自测

## 阶段 2：神经网络原理（3-4 周）
- [ ] 3B1B 神经网络第 1-4 集
- [ ] 跟敲 micrograd
- [ ] ⭐ 不看资料独立重写 micrograd
- [ ] 用 micrograd 训练 MLP 并画出决策边界
- [ ] 纯 NumPy 手推梯度的 MLP 拟合 sin(x)
- [ ] 学习率 / 激活函数实验
- [ ] 完成自测

## 阶段 3：PyTorch 与深度学习实践（3-4 周）
- [ ] PyTorch Learn the Basics
- [ ] d2l 第 2-5 章
- [ ] d2l 批量规范化 + 残差网络
- [ ] Fashion-MNIST MLP > 85%
- [ ] Fashion-MNIST CNN > 90%
- [ ] 自定义 jsonl Dataset
- [ ] 训练循环加入：余弦学习率 / 梯度裁剪 / 梯度累积 / 续训
- [ ] ⭐ 逐行对比 minimind `train_pretrain.py`
- [ ] 完成自测

## 阶段 4：语言模型与 Transformer（4-5 周）
- [ ] makemore Part 1（bigram）
- [ ] makemore Part 2（MLP）
- [ ] makemore Part 3（BatchNorm）
- [ ] Let's build GPT 跟敲完成，tiny shakespeare 训练成功
- [ ] ⭐ 不看资料写出 CausalSelfAttention
- [ ] Let's build the GPT Tokenizer + 实现 BPE
- [ ] 读 Attention Is All You Need
- [ ] 中文版小 GPT
- [ ] 消融实验（位置编码 / 残差 / mask / √d）
- [ ] 完成自测

## 阶段 5：精读并复现 minimind（4-6 周）
- [ ] 学习 RMSNorm / RoPE / GQA / SwiGLU / MoE
- [ ] 读完 `model_minimind.py`，手算参数量 ≈ 64M
- [ ] 读完 `lm_dataset.py`，看过调试打印
- [ ] 读完 `trainer_utils.py`、`train_pretrain.py`、`train_full_sft.py`
- [ ] 读懂 chat template
- [ ] 本机小规模预训练冒烟测试
- [ ] 云 GPU 完整 pretrain + SFT（或下载官方权重）
- [ ] 对比 pretrain vs SFT 模型输出
- [ ] 对比实验 1：__________
- [ ] 对比实验 2：__________
- [ ] 完成自测

## 阶段 6：微调、对齐与强化学习（4-6 周）
- [ ] LoRA 原理 + 读 `model_lora.py`
- [ ] 训练自我认知 LoRA
- [ ] 知识蒸馏原理 + 读 `distillation_loss`
- [ ] HF Deep RL Course Unit 1-4
- [ ] CartPole REINFORCE
- [ ] PPO 原理（HF Deep RL Unit 8）
- [ ] 读 InstructGPT
- [ ] DPO 原理 + 读 `dpo_loss` + 跑一次 DPO
- [ ] GRPO 原理 + 读 `train_grpo.py` + 跑一次 GRPO
- [ ] 读懂 README「PO 算法的统一视角」
- [ ] 写 SFT / DPO / PPO / GRPO 对比笔记
- [ ] 完成自测

## 阶段 7：进阶方向
- [ ] 选定方向：__________
- [ ] 第一个项目：__________
- [ ] 论文阅读（在 07 文件的论文表里打勾）

---

## 周复盘模板

复制到 `notes/weekly/YYYY-WW.md`：

```markdown
# 第 __ 周复盘（YYYY-MM-DD ~ YYYY-MM-DD）

## 本周投入
__ 小时

## 完成了什么
-

## 学到的最重要的 3 个概念（用自己的话解释）
1.
2.
3.

## 卡住的地方 / 还没懂的
-

## 下周计划
-
```
