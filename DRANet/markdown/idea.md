
# some idea of mine

## 论文原本的思路

已有：x_s, y_s：来自 MNIST（源域），有标签。（手写数字图像 + 标签）。
x_t：来自 MNIST-M（目标域），无标签。（彩色背景干扰的手写数字）
希望：已有一个在原域上很好的模型，现在想它训练一下在目标域上也很好。

特征提取器 G，分类器 C
把 x_s 送入 G 得到特征 f_s = G(x_s)
然后送入 Classifier C 得到预测 ŷ_s = C(f_s)
分类器 C 是不动的，现在问题是 G 在处理 MNIST-M 时特征提取不够好 f_t = G(x_t)。
我们希望两者可以提取出相同的特征，其实就是特征的对齐。
G 学习提取的目标域特征 f_t 与源域特征 f_s 分布接近

为了衡量接近程度，我们用：判别器 D 判断特征来自源域还是目标域
然后这里用一个对抗，效果会比较好。

## 我的改进

DRANet 的核心是：

- 对抗性训练：使源域和目标域的特征在判别器中不可区分；
- Domain Regularizer：保持源域特征具有良好的判别性；

### 改进的启用

- 启用改进1：改变`trainer.py`的`Trainer`类的`self.arg_use_entropy`变量
- 启用改进2：改变`trainer.py`的`Trainer`类的`self.arg_use_vat`变量
- 启用改进3：改变`DRANet.py`里面的 `USE_NEW_Discriminator、USE_NEW_Generator`变量

- 启用 D_optimization: 改变`trainer.py`的`Trainer`类的`self.arg_use_D_optimization`变量
  - 启用原本的方法：改变`trainer.py`的`Trainer`类的`self.arg_D_try1`变量
  - 启用改进1：改变`trainer.py`的`Trainer`类的`self.arg_D_try2`变量
  - 启用改进2：改变`trainer.py`的`Trainer`类的`self.arg_D_try3`变量
  - 启用改进3：改变`trainer.py`的`Trainer`类的`self.arg_D_try4`变量

### 改进1

**思想**：只是通过“模仿源域”来间接学习。考虑鼓励模型在目标域上输出低熵，表示它对自己的预测更有信心。就是加一项损失：

$$
\mathcal{L}_{ent} = - \sum_x \sum_{c} p_c(x) \log p_c(x)
$$

* $p_c(x)$ 是 softmax 后第 $c$ 类的概率；
* 对目标域无标签样本使用，可作为损失加到 Generator 的优化目标中。

**好处**：让目标域特征靠近分类边界的中心；
* 抑制“模棱两可”的预测结果；
* 有助于提升迁移泛化效果。

### 改进2

**思想**：加入扰动后仍应保持目标域预测一致，提升模型对小扰动的鲁棒性。

* 对目标域样本 $x$，添加对抗扰动 $r_{adv}$：

  $$
  r_{adv} = \arg \max_r D_{KL}(p(y|x) \| p(y|x + r))
  $$
* 最小化：

  $$
  \mathcal{L}_{vat} = D_{KL}(p(y|x) \| p(y|x + r_{adv}))
  $$
  
可以在 DRANet 原始损失函数的基础上加入：

```python
total_loss = loss_adv + loss_src_cls + alpha * loss_ent + beta * loss_vat
```

* `loss_ent` 是目标域 entropy loss；
* `loss_vat` 是 VAT loss；
* `alpha, beta` 是两个可调系数。

loss_function里面：epsilon=1.0, xi=10.0, iterations=1 也是可调参数

* 固定 iterations=1，调整 epsilon（如 0.01, 0.1, 0.3）。
* 固定 epsilon，调整 xi（如 epsilon/10, epsilon/5, epsilon/2）。
* 最后调整 iterations（如 1, 3, 5）。

### 改进3

结果见 `MyExp1`，**改进3有较好的优化效果**
在 DRANet.py 里的 Discriminator_MNIST 用 CNN 结构。考虑给它换模型。
因为 MNIST 比较小，所以 transformer 可能性能反而不好（大样本时更优）

- 在 MNIST 下：考虑用 ResNet-18 替换 CNN
- 其他比较大的 dataset：考虑用 transformer 替换 CNN
- 由于 Discriminator 性能加强，所以为了平衡对抗，也加强了 Generateor

可调整参数：

- 卷积层的层数
- nn.BatchNorm2d(256)，nn.Dropout2d(0.3)等等

实验结果：

- 原本模型：M2MM: 80.79 | MM2M: 97.79
- 改进模型：M2MM: 82.00 | MM2M: 97.07

有价值的实验过程：

- 发现只加强 D 是不行的，会使对抗失去平衡，所以也要加强 G

### 改进4

* **思路：** 将输入到判别器的不是 `feature`，而是 `feature × classifier_output` 的联合表示。

  $\text{CDAN input} = f(x) \otimes p(y|x)$
  
  * `f(x)` 是图像特征
  * `p(y|x)` 是分类器的输出概率（包含类别信息）

* **优点**：
  * 二者联合，可以让判别器“知道”哪些特征对应哪些类别
  * 这样对抗训练能更加**类别感知（class-aware）**
  * 比 DANN 更稳定

* 有价值的实验过程：
  * 困惑一段时间，问 gpt 也问不出来，发现 $ p(y|x) $ 的权重不能太大，不然就变成学习这个分布了，所以要合理调控 $ p(y|x) $。
  
## 第二个研究方向

针对上面的问题做出优化。

### 原本思路：

* 正常做外积，结果见 `MyExp2`。正确率：57.19%

### 改进1：

* 做投影，结果见 `MyExp3`。正确率：59.05%

### 改进2：

* 加 Gradient Reversal Layer（GRL），避免 D 使用 p 时对特征学习的影响反向传播到分类器：
* 结果见 `MyExp4`。正确率：57.99%

```python
joint = grad_reverse(joint, lambda_=1.0)
domain_logits = D(joint)
```

**大体过程**：

```less
features + classifier_outputs
   ↓
门控融合生成 combined: [B, C, H, W]
   ↓
adapter 输出 RGB 图像: [B, 3, 32, 32]
   ↓
F.interpolate → [B, 3, 64, 64]
   ↓
grad_reverse → [B, 3, 64, 64] (梯度翻转)
   ↓
D（判别器）
```

核心是这个代码：

```python
combined[dset] = grad_reverse(combined[dset], lambda_=self.grl_lambda)
```

这样 `f` 会被推动去“混淆领域信息”，而不是让 `p` 决定一切。

这里的另外一个优化是对 grl_lamda 进行一个动态赋值，

### 改进3

* 思路：用随机投影（CDAN+R）降低维度避免过拟合
* 结果见 `MyExp5`。正确率：61.70%

## 实验结果

对于第一个研究话题，统一采用10000步的时候的数据

- 改进1 + 改进2：
Acc: 70.359% (6333/9001) Acc: 93.650% (9365/10000)
- 改进1：
Acc: 68.903% (6202/9001) Acc: 97.070% (9707/10000)
- 改进2：
  - epsilon=1.0, xi=10.0, iterations=1：Acc: 64.048% (5765/9001) Acc: 95.640% (9564/10000)
  - epsilon=0.3, xi=0.1, iterations=1： Acc: 74.792% (6732/9001) Acc: 97.930% (9793/10000)
- 改进3：
M2MM: 82.00 | MM2M: 97.07
