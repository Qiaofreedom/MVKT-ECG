import torch
import torch.nn as nn

def multi_label_KL_loss(logits_S, logits_T, temperature, num_classes):  # 论文用的这个地方. 这是一个基于 每个类别的二分类 KL 散度计算 的蒸馏损失函数。
    logits_S = logits_S.sigmoid().unsqueeze(2)
    logits_T = logits_T.sigmoid().unsqueeze(2)
    
    logits_S = torch.cat([logits_S, 1-logits_S], dim=2)
    logits_T = torch.cat([logits_T, 1-logits_T], dim=2)
    ans = 0
    for i in range(num_classes):
        logits_S_i = logits_S[:, i, :]
        logits_T_i = logits_T[:, i, :]
        ans += nn.KLDivLoss()(torch.log(logits_S_i / temperature + 1e-8), logits_T_i / temperature + 1e-8)  # 对每个类别都计算一个二分类分布之间的 KLDivLoss；
    return ans
    
def multi_label_KL_loss_v2(logits_S, logits_T, temperature): # 从 多标签角度 直接推导的对称 KL 损失（实际上类似二值交叉熵的 KL 重写）
    logits_S = logits_S.sigmoid() / temperature
    logits_T = logits_T.sigmoid() / temperature
    
    loss = -logits_T * torch.log(logits_T) + logits_T * torch.log(logits_S) - (1-logits_T) * torch.log(1-logits_T) + (1-logits_T) * torch.log(1-logits_S)
    return torch.sum(loss) / logits_S.shape[0]
    
