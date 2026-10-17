# 阶段 3：PyTorch 与深度学习实践（3-4 周）

## 本阶段目标

- [ ] 熟练写出标准训练循环，不用查资料
- [ ] 掌握 `Tensor` / `autograd` / `nn.Module` / `Dataset` / `DataLoader` / `optimizer`
- [ ] 理解残差连接、归一化、Dropout 这些"Transformer 也在用"的组件
- [ ] 掌握工程技巧：学习率调度、梯度裁剪、梯度累积、混合精度、checkpoint 续训
- [ ] **能逐行读懂 `minimind/trainer/train_pretrain.py`**

---

## 资源

| 优先级 | 资源 | 说明 |
|--------|------|------|
| ★主线 | 《动手学深度学习》PyTorch 版 [zh.d2l.ai](https://zh.d2l.ai) + 李沐 B 站配套视频 | 中文、免费、代码可运行，业界公认的入门教材 |
| ★辅助 | [PyTorch 官方 Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) | 官方 1 小时速成，先过一遍 |
| 选学 | 吴恩达 Deep Learning Specialization | 系统但偏慢 |

### d2l 读哪些章节

| 章节 | 必读/选读 | 重点 |
|------|----------|------|
| 2 预备知识 | 必读 | 张量操作、自动微分（对照你的 micrograd！） |
| 3 线性神经网络 | 必读 | 线性回归、softmax 回归、交叉熵 |
| 4 多层感知机 | 必读 | MLP、过拟合、权重衰减、Dropout |
| 5 深度学习计算 | 必读 | 自定义层、参数管理、读写模型、GPU |
| 6 卷积神经网络 | 选读前半 | 了解 CNN 即可，LLM 不用 |
| 7 现代卷积神经网络 | 只读"批量规范化"和"残差网络" | **归一化**和**残差连接**是 Transformer 的关键组件 |
| 8-9 循环神经网络 | 选读 | 了解 RNN 为什么被 Transformer 取代 |
| 11 优化算法 | 选读 | SGD、Momentum、Adam、学习率调度 |

---

## 周计划

### 第 1 周：Tensor 与 autograd
- Tensor 操作（对比 NumPy，几乎一样）
- `requires_grad`、`.backward()`、`.grad`——和你写的 micrograd 一一对应
- 设备：`cpu` / `mps`（你的 Mac）/ `cuda`（NVIDIA）
- 数据类型：`float32` / `float16` / `bfloat16` 的区别

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

### 第 2 周：标准训练循环
- 线性回归 → softmax 回归 → MLP，数据集用 Fashion-MNIST
- 把下面的模板**背下来**（所有深度学习训练都是这个骨架）：

```python
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                      # 1. 前向
        loss = F.cross_entropy(logits, y)      # 2. 算 loss
        optimizer.zero_grad()                  # 3. 梯度清零
        loss.backward()                        # 4. 反向传播
        optimizer.step()                       # 5. 更新参数

    model.eval()
    with torch.no_grad():
        ...                                    # 在验证集上评估
```

### 第 3 周：网络组件
- CNN 入门（LeNet）——了解即可
- **残差连接** `x = x + f(x)`：为什么能训练很深的网络
- **归一化**：BatchNorm vs LayerNorm 的区别（Transformer 用 LayerNorm / RMSNorm）
- Dropout

### 第 4 周：工程技巧（直接对应 minimind 训练脚本）

| 技巧 | 作用 | minimind 里的位置 |
|------|------|------------------|
| 保存/加载 `state_dict` | 存模型 | `trainer_utils.py` → `lm_checkpoint` |
| 断点续训 | 训练中断后接着训 | `--from_resume 1` |
| 余弦学习率 | 学习率先大后小 | `trainer_utils.py` → `get_lr` |
| 梯度裁剪 | 防止梯度爆炸 | `clip_grad_norm_` |
| 梯度累积 | 显存不够时模拟大 batch | `--accumulation_steps` |
| 混合精度 | 省显存、提速 | `autocast` + `GradScaler` |
| 随机种子 | 实验可复现 | `setup_seed` |
| 日志可视化 | 看 loss 曲线 | `--use_wandb`（实际用 swanlab） |

---

## 动手练习

1. 从零写 Fashion-MNIST 的 MLP 分类器，测试准确率 > 85%，在 MPS 上训练
2. 换成 CNN，准确率 > 90%
3. 写一个自定义 `Dataset` 类读取 jsonl 文件（为 minimind 做准备）
4. 在你的训练循环里依次加入：余弦学习率、梯度裁剪、梯度累积、checkpoint 保存与续训
5. ⭐ **检验题**：打开 `minimind/trainer/train_pretrain.py`，和你自己写的训练循环逐行对比，在笔记里写下有哪些异同

代码放到 `study/code/03-pytorch/`。

---

## 自测

1. `nn.Module` 的 `__init__` 和 `forward` 分别做什么？
2. 为什么每一步都要 `optimizer.zero_grad()`？忘了会怎样？
3. `model.train()` 和 `model.eval()` 有什么区别？`torch.no_grad()` 有什么用？
4. `batch_size=32, accumulation_steps=8` 等价于多大的 batch？为什么要这么做？
5. `bfloat16`、`float16`、`float32` 有什么区别？为什么 `float16` 需要 `GradScaler`？
6. 残差连接解决了什么问题？
7. 你能逐行解释 minimind `train_pretrain.py` 里 `train_epoch` 函数的每一行吗？

✅ 完成后进入 [04-语言模型与Transformer.md](./04-语言模型与Transformer.md)
