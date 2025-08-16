import torch
from torch import nn
from .memory import ContrastMemory

eps = 1e-7

# 该代码是 MVKT-ECG 论文中 CLT（Contrastive Lead-information Transferring）对比学习损失函数的实现，采用了 双向 anchor 对比策略，即 Teacher 和 Student 都分别作为 anchor 参与计算。


class CRDLoss(nn.Module):   # Contrastive Lead-information Transferring 部分。也就是论文里面的 CLT函数
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        args.s_dim: the dimension of student's feature
        args.t_dim: the dimension of teacher's feature
        args.feat_dim: the dimension of the projection space
        args.nce_k: number of negatives paired with each positive
        args.nce_t: the temperature
        args.nce_m: the momentum for updating the memory buffer
        args.n_data: the number of samples in the training set, therefor the memory buffer is: args.n_data x args.feat_dim  # 这里包含的是一个训练集的所有样本
    """
    def __init__(self, args):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(args.s_dim, args.feat_dim)
        self.embed_t = Embed(args.t_dim, args.feat_dim)
        self.contrast = ContrastMemory(args.feat_dim, args.n_data, args.nce_k, args.nce_t, args.nce_m)
        # 创建 memory bank，用于构建负样本对比空间。nce_k: 每个正样本配对的负样本数量。n_data: 训练集总样本数量。nce_t: 温度参数（如 0.07）。nce_m: memory 更新的 momentum。
        self.criterion_t = ContrastLoss(args.n_data)  # 分别定义以 Teacher 或 Student 为 anchor 的对比损失loss。
        self.criterion_s = ContrastLoss(args.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None): 
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]  # f_s, f_t: Student 和 Teacher 的特征输出（如 [B, 222]）
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]  # 当前 batch 的样本索引
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        #print(f_s.shape,f_t.shape)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)  # out_s: student-anchor，对比 teacher 生成的 queue。out_s 中的 anchor 是 student， out_t 中的 anchor 是 teacher
        #print(out_s.shape, out_t.shape)
        s_loss = self.criterion_s(out_s)  # student 为 anchor
        t_loss = self.criterion_t(out_t)  # teacher 为 anchor
        loss = s_loss + t_loss  # 计算双向对比损失并相加，返回最终对比损失。
        return loss


class ContrastLoss(nn.Module):  # CLT损失函数部分。 实现对比损失公式，对正负样本做 softmax。
    """
    contrastive loss, corresponding to Eq (18)
    在对比学习中，每个样本通常会形成一个 正对（positive pair）（例如来自 teacher 和 student 的表示），并配有多个 负对（negative pairs）（即不属于该 anchor 的其他样本）。
    为每一个 anchor 计算它与正样本、负样本之间的相似度，然后使用 softmax 使得正样本的相似度得分尽量高，负样本得分尽量低。
    """
    #举例解释
    # Anchor：外部定义，当前行代表的主样本：可能是 f_s[i] 或 f_t[i]
    # Positive：第 0 列：x[:, 0] = sim(anchor, positive)：同一个病人的另一个视图（如单导联 vs 多导联）
    # Negatives：第 1~K 列：x[:, 1:] = sim(anchor, negatives)：其它病人的表示，batch 或 memory bank 中的干扰样本
    
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]  #bach size
        m = x.size(1) - 1  #负样本个数。第一个为正样本。 其他K 个负样本。 x: [B, K+1] 的 logits，第一列是正样本（positive），后续是负样本（negative）
        # 每一行第0列 x[:, 0] 是 anchor–positive 对的相似度打分（如 dot product），后面的 x[:, 1:] 是anchor–positive 对相似度打分
        # x: shape [B, 1 + K]，每一行：
        # [s(anchor, positive), s(anchor, negative_1), ..., s(anchor, negative_K)]
        
        
        # noise distribution   # noise 分布
        Pn = 1 / float(self.n_data) # 噪声分布的概率（均匀分布）

        # loss for positive pair
        P_pos = x.select(1, 0)  # positive logits。 # 取第0列，即正样本得分，形状为 [B]。 P_pos 是每个 anchor 与它正样本之间的匹配分数（例如 cosine 相似度或者 dot product）
        #print(P_pos.shape)
        
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()  
        
        # “计算 anchor 与其正样本之间相似度的 softmax log 概率，目标是最大化这个值，从而拉近正对距离，推远负对。
        # 计算正样本的 log-softmax  概率部分（infoNCE）。m * Pn + eps是负样本的总概率，m * Pn就是 infoNCE 中的 噪声对比项，eps 是为了数值稳定性（防止除以0）


        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)  # anchor 和 K 个负样本的打分，相当于 x[:, 1:]
        # dim=1：在第 1 维（即列的方向）操作；start=1：从第 1 列开始（注意是从第 1 列，不包括第 0 列）；length=m：总共提取 m 列（即负样本个数，m = K）
        #print(P_neg.shape)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()   # 负样本对应的 log softmax 概率部分

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz  # 总体损失：正负样本对的 softmax 对数概率总和，取反，做均值。

        return loss



class Embed(nn.Module): # 投影网络（MLP + 归一化），将原始特征投影到对比空间。
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):  # dim_in 是原始特征维度（如222）， dim_out 是对比空间维度（如128）
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):  # L2 归一化模块。 实现 L2 norm，使得所有向量映射到单位球面上，方便余弦相似度对比。
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
if __name__ =='__main__':
    criterion_kd = CRDLoss()
