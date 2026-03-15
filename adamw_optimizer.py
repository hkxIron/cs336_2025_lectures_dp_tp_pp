import math
import torch
from torch.optim import Optimizer

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