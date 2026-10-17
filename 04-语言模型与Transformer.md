# 阶段 4：语言模型与 Transformer（4-5 周）★ 最核心

## 本阶段目标

- [ ] 理解"语言模型 = 预测下一个 token"，以及 embedding、采样这些基本概念
- [ ] **从零手写 GPT**：不看任何资料能写出带 mask 的自注意力
- [ ] 理解并实现 BPE 分词器
- [ ] 在莎士比亚文本 / 中文文本上训练出你自己的小 GPT

> 这一阶段结束，你就已经掌握了大模型最核心的部分。阶段 5 的 minimind 只是"现代化改良版"。

---

## 资源

### ★主线：Karpathy *Zero to Hero* 第 2-8 集

[课程主页](https://karpathy.ai/zero-to-hero.html)（B 站搜标题有中文字幕搬运）

| 集 | 标题 | 必看？ | 学到什么 |
|----|------|--------|----------|
| 2 | makemore Part 1: bigram | ★必看 | 最简单的语言模型；计数法 vs 神经网络法；采样；负对数似然 |
| 3 | makemore Part 2: MLP | ★必看 | embedding（词向量）；Bengio 2003 神经语言模型；train/val/test 切分 |
| 4 | makemore Part 3: Activations, Gradients, BatchNorm | 强烈建议 | 为什么网络难训练；初始化；归一化 |
| 5 | makemore Part 4: Backprop Ninja | 选看（难） | 手动反向传播，硬核练习 |
| 6 | makemore Part 5: WaveNet | 可跳 | |
| 7 | **Let's build GPT: from scratch, in code, spelled out** | ★★★ | **本阶段最重要的视频**，从零写出 GPT |
| 8 | Let's build the GPT Tokenizer | ★★ | BPE 分词算法 |
| 9 | Let's reproduce GPT-2 (124M) | 选看 | 可放到阶段 5 之后再看 |

配套代码：[makemore](https://github.com/karpathy/makemore) · [nanoGPT](https://github.com/karpathy/nanoGPT) · [minbpe](https://github.com/karpathy/minbpe)

### 辅助理解（视频看不懂时用）

| 资源 | 特点 |
|------|------|
| [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 图解 Transformer，网上有多个中文翻译版 |
| 3Blue1Brown 第 5-7 集**重看** | 这次你应该能完全看懂了 |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) 重新玩一遍 | 对照你写的代码 |
| 李宏毅 2021 春 Self-attention / Transformer 两讲 | 中文讲解 |
| [datawhale happy-llm](https://github.com/datawhalechina/happy-llm) 第 2-3 章 | 中文图文教程，从注意力到 Transformer |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | 逐行注释原论文代码 |
| 李沐《论文精读》Transformer 篇（[paper-reading](https://github.com/mli/paper-reading)） | 带你读原论文 |

### 想要一本书？

Sebastian Raschka《Build a Large Language Model (From Scratch)》（中文版《从零构建大模型》）
[代码仓库](https://github.com/rasbt/LLMs-from-scratch)——第 2-5 章与本阶段完全对应：文本数据处理 → 注意力 → GPT 模型 → 预训练。文字版比视频更方便反复查阅。

---

## 周计划

### 第 1 周：语言模型基础
- makemore 1-2
- 搞懂：什么是语言模型、next-token prediction、embedding 是什么、怎么从概率分布里采样

### 第 2 周：训练稳定性 + 注意力直觉
- makemore 3
- 读 Illustrated Transformer，重看 3B1B 第 6 集（注意力）
- 在笔记里用自己的话写：Q、K、V 分别是什么

### 第 3 周：手写 GPT（分 3-4 次完成）
跟着 *Let's build GPT* 依次实现：
1. 数据处理：字符级 tokenizer、train/val 切分、取 batch
2. bigram baseline
3. 自注意力的"数学小技巧"（下三角矩阵求平均 → mask + softmax）
4. 单头自注意力 → 多头自注意力
5. FeedForward
6. 残差连接 + LayerNorm
7. 组装 Block，堆叠成完整 GPT
8. 在 tiny shakespeare 上训练（Mac MPS 可以跑，几分钟到几十分钟）

### 第 4 周：Tokenizer
- *Let's build the GPT Tokenizer* + 跟着实现 minbpe
- 用 transformers 加载 minimind 的 tokenizer，看它怎么切中文：

```python
from transformers import AutoTokenizer
import os
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/work/ai/minimind/model"))
ids = tok("大语言模型是如何工作的？").input_ids
print(ids, [tok.decode([i]) for i in ids])
print(len(tok))   # 词表大小，应为 6400
```

### 第 5 周：读论文 + 迁移到中文
- 读 *Attention Is All You Need*（先看李沐精读视频，再读原文）
- 把你的 GPT 改成中文版：用 minimind 的 tokenizer + 一小部分 minimind 预训练数据（阶段 5 会下载）训练，看能生成什么 ← 衔接下一阶段

---

## 动手练习

1. 跟敲 Karpathy 视频里的 `gpt.py`（约 200 行）
2. ⭐ **过关考试**：不看任何资料，在空文件里写出 `CausalSelfAttention`（含多头、causal mask、缩放），并通过一个形状测试
3. 实现 BPE：在一段中文文本上训练 `vocab_size=500` 的分词器，对比 GPT-4 / minimind tokenizer 的切分结果
4. 在中文语料上训练你的小 GPT，看生成效果
5. 消融实验（每个改一处，对比 val loss，写进笔记）：
   - 去掉位置编码
   - 去掉残差连接
   - 去掉 causal mask（注意：这时 loss 会"好得离谱"，想想为什么）
   - 注意力不除以 √d

代码放到 `study/code/04-gpt/`。

---

## 关键概念清单

学完后，下面每个词你都应该能讲清楚（对照 [glossary.md](./glossary.md)）：

token / vocab · embedding · next-token prediction · logits · causal mask · Q/K/V · scaled dot-product attention · multi-head attention · FFN · 残差连接 · LayerNorm · Pre-Norm · 位置编码 · 上下文长度 · temperature · top-k · top-p · KV cache（概念） · BPE

---

## 自测

1. 用自己的话解释 Q、K、V
2. 注意力分数为什么要除以 √d_k？
3. causal mask 是干什么的？训练时去掉会怎样？
4. Transformer 为什么需要位置编码？
5. `vocab=6400, d_model=768` 的 embedding 层有多少参数？
6. 训练时 input 和 label 是什么关系？（提示：错开一位）
7. BPE 是怎么一步步合并出词表的？
8. 为什么训练时可以并行处理整句话，而生成时只能一个一个 token 往外蹦？
9. temperature、top-k、top-p 分别怎么影响生成结果？
10. 一个 GPT 的参数主要集中在哪些层？

✅ 完成后进入 [05-精读并复现minimind.md](./05-精读并复现minimind.md)
