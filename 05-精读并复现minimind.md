# 阶段 5：精读并复现 minimind（4-6 周）

## 本阶段目标

- [ ] 理解从 GPT-2 到现代 LLM（Llama / Qwen）的架构演进：RMSNorm、RoPE、GQA、SwiGLU、MoE
- [ ] **逐行读懂** minimind 的模型、数据、训练代码
- [ ] 亲手从零完成 **预训练 + SFT**，得到一个能对话的模型
- [ ] 做 2-3 个对比实验，写实验报告

> 前置：完成阶段 4。你已经手写过 GPT，minimind 本质上就是"现代化改良版的 GPT"。

---

## 一、从你的 GPT 到 minimind：改了什么

minimind-3 的结构对齐 Qwen3。和阶段 4 你写的 GPT 相比，主要改动如下：

| 组件 | 你在阶段 4 写的 GPT | minimind | 为什么改 | 代码位置（`model/model_minimind.py`） |
|------|-------------------|----------|----------|------------------------------------|
| 归一化 | LayerNorm | **RMSNorm** | 去掉减均值，更简单更快，效果相当 | `class RMSNorm`（约第 50 行） |
| 归一化位置 | Pre-Norm | Pre-Norm | 训练更稳定 | `class MiniMindBlock`（约第 183 行） |
| 位置编码 | 可学习的绝对位置 | **RoPE** 旋转位置编码（+ YaRN 外推） | 编码相对位置，可外推到更长文本 | `precompute_freqs_cis` / `apply_rotary_pos_emb`（约第 62-84 行） |
| 注意力 | 多头注意力 MHA | **GQA**（8 个 Q 头共享 4 个 KV 头）+ QK-Norm | 减少 KV cache 显存 | `class Attention`（约第 91 行）、`repeat_kv`（约第 86 行） |
| 前馈网络 | Linear → GELU → Linear | **SwiGLU**（gate / up / down 三个矩阵） | 效果更好 | `class FeedForward`（约第 136 行） |
| 输入输出词表 | 两个独立矩阵 | **共享权重**（tie_word_embeddings） | 小模型省参数 | `MiniMindForCausalLM.__init__`（约第 247 行） |
| 可选 | — | **MoE** 混合专家 | 增加容量但不增加每个 token 的计算量 | `class MOEFeedForward`（约第 148 行） |
| 生成 | 每次重算全部 | **KV cache** + top-k / top-p 采样 | 生成加速 | `generate`（约第 262 行） |

> 行号以当前下载的版本为准，作者更新后可能有偏移，按类名/函数名搜索即可。

### 学习这些新组件的资源

- **RoPE**：苏剑林《[Transformer 升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265)》（RoPE 原作者的中文博客）
- **整体架构**：[happy-llm](https://github.com/datawhalechina/happy-llm) 第 5 章"动手搭建大模型"（中文，代码风格与 minimind 接近）
- **GPT → Llama → Qwen3 逐步改造**：[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 第 5 章的 bonus 材料
- 论文（选读）：RMSNorm、RoFormer（RoPE）、GQA、GLU Variants（SwiGLU）——见 [07-进阶方向.md](./07-进阶方向.md) 论文清单

---

## 二、代码阅读顺序

**读代码的技巧**：用 Jupyter 导入模块，构造一个 `batch=2, seq_len=10` 的假输入跑一遍；在每个 `forward` 里打印 shape。

```python
import os, sys; sys.path.append(os.path.expanduser("~/work/ai/minimind"))
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M")   # 应约等于 64M

x = torch.randint(0, 6400, (2, 10))
out = model(x, labels=x)
print(out.logits.shape, out.loss)    # (2, 10, 6400)，loss 约为 ln(6400)≈8.76
```

| 顺序 | 文件 | 看什么 | 思考题 |
|------|------|--------|--------|
| 1 | `model/model_minimind.py`（292 行） | 先看 `MiniMindConfig` 里有哪些超参；然后**自底向上**：RMSNorm → FeedForward → RoPE → Attention → Block → Model → ForCausalLM → generate | `forward` 里 loss 为什么是 `logits[..., :-1]` 对 `labels[..., 1:]`？ |
| 2 | `dataset/lm_dataset.py` | `PretrainDataset`：bos + 文本 + eos，padding，pad 位置的 label 设为 -100<br>`SFTDataset`：用 chat template 拼对话，`generate_labels` 只给 assistant 回复算 loss | 把 `__getitem__` 里注释掉的"调试打印"打开，看每个 X → Y 的对应关系 |
| 3 | `trainer/trainer_utils.py` | `get_lr`（余弦退火，最低降到初始的 10%）、`init_model`、`lm_checkpoint` | 把 `get_lr` 的曲线画出来 |
| 4 | `trainer/train_pretrain.py` | 对照阶段 3 你自己写的训练循环 | 梯度累积是怎么实现的？ |
| 5 | `trainer/train_full_sft.py` | 用 diff 工具和 pretrain 脚本对比 | 两者唯一的本质区别是什么？ |
| 6 | `eval_llm.py` | 推理流程、`apply_chat_template` | pretrain 权重和 sft 权重的输入为什么不同？ |
| 7 | `model/tokenizer_config.json` | 里面的 `chat_template`（Jinja 模板）和特殊 token | 一段多轮对话最终被拼成了什么样的字符串？ |
| 8 | `trainer/train_tokenizer.py` | 对照阶段 4 你写的 BPE | |

---

## 三、动手训练

### Step 1：下载数据（本机）

从 [ModelScope 数据集页面](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 下载 `pretrain_t2t_mini.jsonl` 和 `sft_t2t_mini.jsonl`，放到 `minimind/dataset/`。

```bash
cd ~/work/ai/minimind/dataset
head -n 3 pretrain_t2t_mini.jsonl     # 看格式
head -n 1 sft_t2t_mini.jsonl
wc -l *.jsonl                         # 看条数
```

### Step 2：本机小规模冒烟测试

目标不是训出好模型，而是**确认流程跑通、loss 在下降**。

```bash
cd ~/work/ai/minimind/dataset
head -n 20000 pretrain_t2t_mini.jsonl > pretrain_tiny.jsonl

cd ../trainer
python train_pretrain.py \
  --device mps \
  --data_path ../dataset/pretrain_tiny.jsonl \
  --hidden_size 512 --num_hidden_layers 8 \
  --batch_size 16 --accumulation_steps 1 \
  --num_workers 0 --log_interval 10 --epochs 1
```

> 参考命令，未在你的机器上实测。脚本里混合精度是按 CUDA 写的，在 MPS 上会退化为 float32 训练，速度慢一些但能用。遇到报错把报错贴给 AI 一起分析。
> 用了 `--hidden_size 512`，后面推理时也要加 `--hidden_size 512`。

### Step 3：云 GPU 完整训练（约几块钱）

在 AutoDL 等平台租一张 3090 / 4090，按 `minimind/README.md` 的「2.1 预训练」「2.2 指令微调」执行：

```bash
cd trainer
python train_pretrain.py      # 约 1.2 小时（3090）
python train_full_sft.py      # 约 1.1 小时（3090）
```

- 加 `--use_wandb` 记录 loss 曲线（实际用的是 swanlab），截图保存到笔记里
- 训练完把 `out/` 下的 `.pth` 下载回本机
- 预算紧张也可以直接下载作者训练好的权重，见 README「训练结果开源」一节

### Step 4：评估对比

```bash
cd ~/work/ai/minimind
python eval_llm.py --weight pretrain --device mps    # 预训练模型：只会"续写"
python eval_llm.py --weight full_sft --device mps    # SFT 模型：会"对话"
```

**亲眼对比两者的区别**——这就是 SFT 的作用。

---

## 四、对比实验（选 2-3 个，写实验报告）

每个实验在笔记里写：假设 → 设置 → 结果（loss 曲线截图 + 生成样例） → 结论。

1. **深 vs 宽**：参数量相近时，`hidden_size` 和 `num_hidden_layers` 的不同组合（参考 README「模型配置」一节和 MobileLLM 论文）
2. **Dense vs MoE**：`--use_moe 1`
3. **学习率**：`5e-4` vs `5e-3` vs `5e-5`
4. **组件消融**：把 RMSNorm 换回 LayerNorm，或把 SwiGLU 换回普通 FFN
5. **生成参数**：temperature / top_p 对回答质量的影响
6. **长度外推**：`--inference_rope_scaling`（README「RoPE 长度外推」一节）

---

## 五、扩展对照项目（选学）

同样的流程，看不同作者怎么组织代码，能帮你区分"本质"和"实现细节"：

- Karpathy *Let's reproduce GPT-2 (124M)*（视频 + [build-nanogpt](https://github.com/karpathy/build-nanogpt)）：工业级训练技巧（混合精度、FlashAttention、DDP、超参设置）
- [nanochat](https://github.com/karpathy/nanochat)（Karpathy，2025）：完整的"ChatGPT 克隆"，包含 tokenizer、预训练、SFT、RL、推理、Web UI，和 minimind 定位最接近的对照项目
- [tiny-universe](https://github.com/datawhalechina/tiny-universe)（datawhale）：手写 LLM、RAG、Agent 等"小而全"的白盒实现

---

## 自测

1. 不看 README 的图，画出 minimind 的完整结构图
2. 手算 minimind-3（`d=768`、8 层、`vocab=6400`、`q_heads=8`、`kv_heads=4`）的参数量，验证约等于 64M
   （提示：`head_dim = 768/8 = 96`；`intermediate_size = ceil(768·π/64)·64 = 2432`；词表权重是共享的）
3. `PretrainDataset` 为什么把 pad 位置的 label 设为 -100？
4. SFT 为什么只对 assistant 的回复部分计算 loss？
5. GQA 里 `n_rep = 2` 是什么意思？省掉了什么？
6. KV cache 在 `generate` 里是怎么工作的？（看 `past_len`）
7. pretrain 模型和 SFT 模型对"你好"的回应有何不同？为什么？
8. RMSNorm 和 LayerNorm 的公式区别是什么？
9. RoPE 为什么能表示"相对位置"？

✅ 完成后进入 [06-微调对齐与强化学习.md](./06-微调对齐与强化学习.md)
