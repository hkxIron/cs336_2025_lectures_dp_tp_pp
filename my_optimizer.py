import math
import torch
from torch.optim import Optimizer

"""
AdamW（Decoupled Weight Decay Regularization）


为了彻底理解 AdamW，我们需要先定义好所有的数学符号，然后一步步拆解它的计算过程。

AdamW 的核心贡献在于：**将权重衰减（Weight Decay）与基于梯度的自适应学习率更新完全解耦（Decoupled）。**

---

### 1. 符号定义 (Notation)

*   $t$ : 当前的迭代步数 (Step)，从 1 开始。
*   $\theta_t$ : 第 $t$ 步的模型参数（权重）。
*   $f(\theta)$ : 损失函数 (Loss function)。
*   $g_t$ : 第 $t$ 步的梯度，即 $\nabla_{\theta} f(\theta_{t-1})$。
*   $\eta$ : 学习率 (Learning rate, `lr`)。
*   $\lambda$ : 权重衰减系数 (Weight decay coefficient)。
*   $\beta_1, \beta_2$ : 一阶矩和二阶矩的指数衰减率（通常默认为 0.9 和 0.999）。
*   $m_t$ : 一阶矩估计 (First moment)，即梯度的指数移动平均（动量）。
*   $v_t$ : 二阶矩估计 (Second moment)，即梯度平方的指数移动平均。
*   $\hat{m}_t, \hat{v}_t$ : 经过偏差校正后的一阶矩和二阶矩。
*   $\epsilon$ : 防止除零的微小常数（通常为 $10^{-8}$）。

---

### 2. AdamW 完整数学公式 (Step-by-Step)

在每一步迭代 $t$ 中，AdamW 执行以下计算（注意：所有的乘法、除法、平方和开方都是**逐元素 (element-wise)** 进行的）：

#### Step 1: 计算当前梯度
$$ g_t = \nabla_{\theta} f(\theta_{t-1}) $$

#### Step 2: 更新一阶矩估计 (动量)
$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $$

#### Step 3: 更新二阶矩估计 (RMSProp)
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$

#### Step 4: 偏差校正 (Bias Correction)
由于 $m_0$ 和 $v_0$ 初始化为 0，在训练初期它们会严重偏向于 0。因此需要除以 $(1 - \beta^t)$ 进行放大校正：
$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$
$$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$

#### Step 5: 参数更新 (包含解耦的权重衰减)
这是 AdamW 最核心的公式。参数的更新由两部分组成：**自适应梯度更新** + **独立的权重衰减**。
$$ \theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right) $$

为了更清晰地看出“解耦”的过程，这个公式通常被拆解为等价的两步（这也是 PyTorch 底层的实际执行逻辑）：
1. **应用权重衰减 (Weight Decay):**
   $$ \theta_{t-1}' = \theta_{t-1} - \eta \lambda \theta_{t-1} = \theta_{t-1}(1 - \eta \lambda) $$
2. **应用 Adam 梯度更新:**
   $$ \theta_t = \theta_{t-1}' - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

---

### 3. 深度解析：AdamW 为什么要“解耦”？(Adam vs AdamW)

要理解 AdamW 的伟大之处，必须看看老版的 **Adam** 是怎么做权重衰减的。

在标准的 Adam 中，权重衰减等价于 **L2 正则化**。它是通过直接修改**梯度**来实现的：
$$ g_t = \nabla_{\theta} f(\theta_{t-1}) + \lambda \theta_{t-1} $$

然后，这个被修改过的 $g_t$ 会被送入 Step 2 和 Step 3，去计算 $m_t$ 和 $v_t$。
最终的参数更新公式变成了：
$$ \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

**致命缺陷在哪里？**
在老版 Adam 中，包含权重衰减的梯度被放到了分子 $\hat{m}_t$ 中，但同时，它也被放到了分母 $\sqrt{\hat{v}_t}$ 中！
这意味着：**对于那些历史梯度很大（$\hat{v}_t$ 很大）的参数，它的权重衰减惩罚会被分母除以一个很大的数，从而导致惩罚力度大打折扣！**

换句话说，老版 Adam 的 L2 正则化效果是不均匀的，它无法有效地将无用的权重推向 0，这就是为什么早期 NLP 和 CV 任务中，大家发现 Adam 泛化能力不如 SGD + Momentum 的根本原因。

**AdamW 的解决方案：**
AdamW 提出：**不要把 $\lambda \theta_{t-1}$ 加到梯度 $g_t$ 里！**
让 $g_t$ 纯粹只包含 Loss 的梯度，用纯粹的梯度去计算 $m_t$ 和 $v_t$。
然后在最后一步更新参数时，直接在参数本身上减去 $\eta \lambda \theta_{t-1}$。

这样一来，权重衰减的力度 $\eta \lambda$ 对所有参数都是公平、一致的，不再受历史梯度大小（分母 $\sqrt{\hat{v}_t}$）的干扰。这就是所谓的 **Decoupled Weight Decay（解耦权重衰减）**。


### 2. 核心代码原理解析

在上面的代码中，有几个非常关键的 PyTorch 底层优化技巧（这也是官方源码的写法）：

1. **原地操作（In-place Operations）**：
   * 深度学习中模型参数动辄几十亿，如果写成 `p = p - lr * grad`，PyTorch 会在显存中开辟一块**新内存**来存放结果，导致显存瞬间爆炸（OOM）。
   * 必须使用带下划线的方法：`p.mul_()`、`p.add_()`、`p.addcmul_()`、`p.addcdiv_()`。这些方法直接在原内存地址上修改数据，**零额外显存开销**。

2. **解耦权重衰减（Decoupled Weight Decay）**：
   ```python
   p.mul_(1 - lr * weight_decay)
   ```
   这行代码完美体现了 AdamW 的 "W"。它在利用梯度更新参数之前，先让参数自身按比例缩小了一点点。这等价于公式 $w_t = w_{t-1} - \eta \lambda w_{t-1}$。

3. **偏差校正的数学等价替换**：
   标准的数学公式是：
   $\hat{m}_t = m_t / (1 - \beta_1^t)$
   $\hat{v}_t = v_t / (1 - \beta_2^t)$
   $w_t = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
   
   如果直接按公式写，需要为 $\hat{m}_t$ 和 $\hat{v}_t$ 分配新的 Tensor。为了省显存，代码将其转换为：
   `step_size` = $\frac{\eta}{1 - \beta_1^t}$
   `denom` = $\frac{\sqrt{v_t}}{\sqrt{1 - \beta_2^t}} + \epsilon$
   $w_t = w_t - \text{step\_size} \times \frac{m_t}{\text{denom}}$
   这样只需要为 `denom` 分配一次临时内存，极大地优化了性能。

"""
class ManualAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        手动实现的 AdamW 优化器
        Args:
            params: 待优化的参数迭代器
            lr: 学习率 (Learning Rate)
            betas: 用于计算一阶矩和二阶矩的平滑常数 (beta1, beta2)
            eps: 防止除零的微小常数
            weight_decay: 权重衰减系数 (Decoupled Weight Decay)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")

        # 将超参数保存在 defaults 字典中，这是继承 Optimizer 的标准做法
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad() # 优化器的更新步骤不需要计算梯度图
    def step(self, closure=None):
        """执行一次单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有的参数组 (param_groups)
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # 遍历当前组内的所有参数
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ManualAdamW does not support sparse gradients')

                # 获取当前参数的状态字典
                state = self.state[p]

                # 1. 状态初始化 (仅在第一次 step 时执行)
                if len(state) == 0:
                    state['step'] = 0
                    # 一阶矩 (Momentum): 梯度的指数移动平均
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 二阶矩 (RMSProp): 梯度平方的指数移动平均
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                step = state['step']

                # =====================================================
                # 2. 解耦的权重衰减 (Decoupled Weight Decay) - AdamW 的核心
                # =====================================================
                # 公式: w_t = w_{t-1} - lr * weight_decay * w_{t-1}
                # 注意：这里直接修改参数 p，而不是修改梯度 grad
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # =====================================================
                # 3. 更新一阶矩和二阶矩 (Moving Averages)
                # =====================================================
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                # addcmul_ 是原地操作: tensor1 + value * tensor2 * tensor3
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # =====================================================
                # 4. 偏差校正 (Bias Correction)
                # =====================================================
                # 因为初始的 exp_avg 和 exp_avg_sq 都是 0，早期迭代会严重偏向 0
                # 所以需要除以 (1 - beta^t) 进行放大校正
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # =====================================================
                # 5. 计算步长并更新参数
                # =====================================================
                # 原始公式: p = p - lr * (m_hat / (sqrt(v_hat) + eps))
                # 为了极致的显存效率，我们不创建 m_hat 和 v_hat 的新 Tensor，而是通过数学等价变换：

                # 步长缩放: step_size = lr / bias_correction1
                step_size = lr / bias_correction1

                # 分母: denom = sqrt(exp_avg_sq / bias_correction2) + eps
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 参数更新: p = p - step_size * (exp_avg / denom)
                # addcdiv_ 是原地操作: tensor1 + value * (tensor2 / tensor3)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    

if  __name__ == "__main__":
    # 创建两个完全相同的简单模型
    torch.manual_seed(42)
    model_manual = torch.nn.Linear(10, 2)
    model_official = torch.nn.Linear(10, 2)
    model_official.load_state_dict(model_manual.state_dict().copy()) # 保证初始权重绝对一致

    # 初始化优化器
    opt_manual = ManualAdamW(model_manual.parameters(), lr=1e-2, weight_decay=0.1)
    opt_official = torch.optim.AdamW(model_official.parameters(), lr=1e-2, weight_decay=0.1)

    # 模拟一次前向和反向传播
    data = torch.randn(5, 10)
    target = torch.randn(5, 2)

    # 手动优化器 Step
    loss_manual = torch.nn.functional.mse_loss(model_manual(data), target)
    loss_manual.backward()
    opt_manual.step()

    # 官方优化器 Step
    loss_official = torch.nn.functional.mse_loss(model_official(data), target)
    loss_official.backward()
    opt_official.step()

    # 验证权重是否一致
    diff = torch.max(torch.abs(model_manual.weight - model_official.weight))
    print(f"权重最大差异: {diff.item()}")
    assert diff < 1e-6, "实现有误，与官方结果不一致！"
    print("验证通过！手动实现的 AdamW 与官方版本完全一致。")