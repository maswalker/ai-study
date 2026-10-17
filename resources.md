# 学习资源汇总

★ = 本计划的主线资源。其他资源用于"卡住时换个角度理解"。**不要同时开很多门课。**

---

## 视频课程

| 资源 | 语言 | 对应阶段 | 说明 |
|------|------|---------|------|
| ★ [Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | 英（B 站有中字搬运） | 2, 4, 5 | micrograd → makemore → GPT → tokenizer → GPT-2 复现 |
| ★ Karpathy: Deep Dive into LLMs like ChatGPT | 英（B 站有中字搬运） | 0 | 3.5 小时，零基础 LLM 全景 |
| ★ [3Blue1Brown](https://www.3blue1brown.com)：线性代数的本质 / 微积分的本质 / 深度学习系列 | 英（B 站官方中文） | 0, 1, 2, 4 | 动画建立数学直觉 |
| ★ 李沐《动手学深度学习》视频 | 中 | 3 | 配合 d2l 教材 |
| 李沐《论文精读》（[列表](https://github.com/mli/paper-reading)） | 中 | 4-7 | Transformer、GPT、InstructGPT 等经典论文逐段讲 |
| 李宏毅《机器学习》（[2021 春](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)）/《[生成式 AI 导论 2024](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php)》 | 中 | 0, 2, 4, 6 | 中文最好的 ML 课之一，幽默易懂 |
| 吴恩达 Machine Learning / Deep Learning Specialization | 英（B 站有搬运） | 2, 3 | 系统但节奏慢 |
| [Stanford CS224N](https://web.stanford.edu/class/cs224n/)（NLP） | 英 | 4 之后 | 经典 NLP 课 |
| ★ [Stanford CS336](https://stanford-cs336.github.io)（Language Modeling from Scratch） | 英 | 7 | 硬核"毕业设计" |
| StatQuest（YouTube） | 英 | 1 | 统计和 ML 概念讲得极慢极清楚 |

## 教材与书

| 资源 | 对应阶段 | 说明 |
|------|---------|------|
| ★ [《动手学深度学习》zh.d2l.ai](https://zh.d2l.ai) | 3 | 免费在线，PyTorch 版 |
| [《Build a Large Language Model (From Scratch)》](https://github.com/rasbt/LLMs-from-scratch)（中文版《从零构建大模型》） | 4-6 | 从数据处理到指令微调、DPO，代码完整 |
| [RLHF Book](https://rlhfbook.com) | 6 | 免费在线，系统讲后训练 |
| [Mathematics for Machine Learning](https://mml-book.github.io/) | 1（查阅用） | 免费，当参考书 |
| 《Programming Massively Parallel Processors》 | 7 | CUDA 编程经典 |

## 中文开源教程

| 资源 | 对应阶段 | 说明 |
|------|---------|------|
| ★ [廖雪峰 Python 教程](https://www.liaoxuefeng.com/wiki/1016959663602400) | 1 | Python 入门 |
| [datawhale happy-llm](https://github.com/datawhalechina/happy-llm) | 4, 5 | 从零搭建大模型的中文教程 |
| [datawhale tiny-universe](https://github.com/datawhalechina/tiny-universe) | 5 | 手写 LLM / RAG / Agent 白盒实现 |
| [datawhale 蘑菇书 EasyRL](https://github.com/datawhalechina/easy-rl) | 6 | 强化学习中文教程 |
| [苏剑林 · 科学空间](https://kexue.fm) | 5, 7 | RoPE 作者博客，大量 Transformer 深度文章 |

## 动手项目（代码仓库）

| 项目 | 对应阶段 | 学什么 |
|------|---------|--------|
| ★ [micrograd](https://github.com/karpathy/micrograd) | 2 | 约 150 行的自动求导引擎 |
| ★ [makemore](https://github.com/karpathy/makemore) | 4 | 字符级语言模型 |
| ★ [nanoGPT](https://github.com/karpathy/nanoGPT) | 4 | 最简洁的 GPT 训练代码 |
| ★ [minbpe](https://github.com/karpathy/minbpe) | 4 | BPE 分词器 |
| ★ minimind（本地 `../minimind`，[GitHub](https://github.com/jingyaogong/minimind)） | 5, 6 | 现代 LLM 全流程：预训练 → SFT → LoRA → DPO → PPO/GRPO → Agent |
| [build-nanogpt](https://github.com/karpathy/build-nanogpt) | 5 | 复现 GPT-2 124M |
| [nanochat](https://github.com/karpathy/nanochat) | 5 | Karpathy 的完整 ChatGPT 克隆，minimind 的最佳对照 |
| [minimind-v](https://github.com/jingyaogong/minimind-v) | 7 | 视觉语言模型 |
| [llm.c](https://github.com/karpathy/llm.c) | 7 | 纯 C/CUDA 训练 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 7 | 推理引擎 |
| [vLLM](https://github.com/vllm-project/vllm) | 7 | 推理服务 |
| [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm) | 7 | Apple 芯片原生框架 |
| [MiniTorch](https://minitorch.github.io) / [tinygrad](https://github.com/tinygrad/tinygrad) | 7 | 理解深度学习框架内部 |

## 图解与交互工具

| 资源 | 说明 |
|------|------|
| ★ [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | 浏览器里看 GPT-2 每一步计算 |
| ★ [LLM Visualization (bbycroft)](https://bbycroft.net/llm) | 3D 可视化小 GPT |
| ★ [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 图解 Transformer |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | 原论文逐行代码注释 |
| [Transformer Circuits](https://transformer-circuits.pub) | Anthropic 可解释性研究 |

## 在线课程（HuggingFace）

| 资源 | 对应阶段 |
|------|---------|
| [HF LLM Course](https://huggingface.co/learn/llm-course) | 4-6（学会用 transformers 生态） |
| ★ [HF Deep RL Course](https://huggingface.co/learn/deep-rl-course) | 6 |
| [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | 7 |

## 算力

| 平台 | 说明 |
|------|------|
| 本机 Mac（MPS） | 阶段 0-4 足够 |
| Kaggle Notebook | 每周有免费 GPU 额度 |
| Google Colab | 免费版有 GPU（需要能访问） |
| AutoDL 等国内租卡平台 | 3090 约 1~2 元/小时，minimind 完整预训练 + SFT 约几块钱 |

## 数据集

- minimind 数据集：[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- tiny shakespeare：Karpathy *Let's build GPT* 视频中使用，nanoGPT 仓库里有下载脚本
