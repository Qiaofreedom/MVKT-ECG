import torch
from torch import nn
import math


class ContrastMemory(nn.Module): # Contrastive Memory Bank（对比学习中的内存机制），用于支撑 MVKT-ECG 中 CLT（Contrastive Lead-information Transferring） 损失函数的核心——即如何为 正负样本构造大量的 对比对。
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):  # inputSize: 每个特征向量的维度（通常是 projection 后的 128 维），outputSize: 全训练集样本数量（或 memory bank 中的行数）。K: 每个 anchor 对应的负样本个数。T: 温度缩放系数（越小，对相似度的敏感性越高）。momentum: 控制记忆库更新的动量。
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams) # 为所有样本创建一个 均匀分布 采样器 AliasMethod，用于高效采样 K 个负样本（采样策略来源于 noise contrastive estimation，避免全量计算）。unigrams 是一个全为 1 的数组，代表均匀采样概率。
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))  # 用一个 tensor 存储一些参数（如温度 T、动量等），方便 state_dict() 存储。 -1 是占位符，用来存 normalization constant Z，等后续计算。
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))  # memory_v1 和 memory_v2 是两个 对比内存库，存储每个样本的历史嵌入（用于查负样本）
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None): # v1, v2: 当前 batch 中 student 和 teacher 的特征 shape: [B, D]。y: 当前 batch 中样本在整个数据集中的全局索引（用于 memory bank 查值）。idx: 采样得到的正负样本的索引组合矩阵，shape: [B, K+1]，第一列是正样本（即 y），其余为随机负样本
        K = int(self.params[0].item())  # 每个 anchor 采样的负样本数量
        T = self.params[1].item()       # 温度参数，用于 softmax 缩放
        Z_v1 = self.params[2].item()    # 用于归一化 out_v1 的常数
        Z_v2 = self.params[3].item()    # 用于归一化 out_v2 的常数

        momentum = self.params[4].item()   # 动量，用于 memory bank 更新
        batchSize = v1.size(0)             # 当前 batch 的大小
        outputSize = self.memory_v1.size(0)  # memory bank 的样本总数（即训练集大小）
        inputSize = self.memory_v1.size(1)   # 每个样本的特征维度（例如128维）

        # original score computation  采样部分
        if idx is None:  # 如果未给定 idx，就采样正负样本
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)   # 随机采样 K+1 个索引：1 个正样本 + K 个负样本。
            #print(idx.shape)
            idx.select(1, 0).copy_(y.data)   # 然后将第 0 列替换为对应正样本索引 y（确保 anchor 与正样本的 index 对齐）
            #print(idx.shape)
        # sample
        #print(idx.view(-1).shape)
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()  # 在 memory_v1（teacher memory）中按采样索引选出 [B, K+1, D] 的特征矩阵。detach() 断开反向传播
        
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)  # [B(K+1), D] → [B, K+1, D]。 第一列是正样本，后续是负样本
        #print(weight_v1.shape)
        
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))   # student (v2) 作为 anchor，计算 anchor → memory_v1 的 dot-product。用 bmm（批量矩阵乘法）计算每个 anchor 与其正负样本的 dot-product 相似度。
        out_v2 = torch.exp(torch.div(out_v2, T))   # 点积计算 similarity，除以 T 再取 exp，构成 softmax 准备项。 out_v2 shape 最终是 [B, K+1, 1]，表示 anchor-student 与正负样本的相似度得分（未归一化）
        #print(out_v2.shape)
        # sample 采样 memory_v2 中的正负样本向量 → 用于计算 teacher→student
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()  # 查找 memory_v2 中对应的 [B, K+1, D] 特征矩阵，用于计算 teacher → memory_v2 的相似度。
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)   # [B(K+1), D] → [B, K+1, D]。 第一列是正样本，后续是负样本
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))  # teacher (v1) 作为 anchor，计算 anchor → memory_v2 的 dot-product
        out_v1 = torch.exp(torch.div(out_v1, T))    # 并做温度缩放。结果是 [B, K+1, 1]，每行为一个 anchor 与其正负样本的 exp(similarity/T)
        #print(weight_v2.shape)
        #print(out_v1.shape)
        # set Z if haven't been set yet. Z 是用于 normalize softmax 概率的常数（为了 NCE 中的概率项。这个是 NCE Loss 特有部分，区别于 InfoNCE（后者不显式使用 Z）），归一化常数 。Z 是为模拟 softmax 的分母项（近似 ∑Esim/T）。只在第一次 forward 时初始化为期望值 × memory 大小。设置归一化常数 Z（只第一次计算）
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2 归一化输出
        out_v1 = torch.div(out_v1, Z_v1).contiguous()  # 将相似度得分除以Z，模拟 softmax 的分布概率（无 log）
        out_v2 = torch.div(out_v2, Z_v2).contiguous()  # out_v1, out_v2 都是 [B, K+1, 1]，每行为 anchor 与正负样本的 NCE 概率

        # update memory 更新 memory bank（动量更新）.   按照动量 momentum 更新 memory 中对应的 v1 向量。
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))  # 取出 memory_v1 中正样本位置（y 索引对应的向量）
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)    # 按照动量公式更新：new_vec=momentum×old_vec+(1−momentum)×new_vec
            self.memory_v1.index_copy_(0, y, updated_v1)   # 归一化后替换掉原来的向量（确保单位向量）

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))  # 下方对 memory_v2 进行完全相同的更新逻辑。
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2  # 输出 shape 为 [B, K+1]，包含 anchor 与正负样本的 exp 相似度（已除以温度、归一化 Z）. 将输入给 ContrastLoss 进行 NCE 或 InfoNCE 损失计算


class AliasMethod(object): # 该类实现的是 高效的 Multinomial 抽样算法（别名采样法），用于快速从大规模类别中采样负样本。
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    你这段 AliasMethod 是一种 高效采样算法（也叫“别名法”），适用于从大量类别中 以离散概率分布进行抽样，核心目的是：实现 O(1) 时间复杂度的采样。
它常用于对比学习的 负样本采样（如 memory bank 中采 K 个样本）
    """
    def __init__(self, probs):  # probs: 一个长度为 K 的一维向量，表示每个类别被采样的概率（离散分布）。在你的应用中，常常是均匀分布 torch.ones(K)（即负样本等概率采样）

        if probs.sum() > 1:
            probs.div_(probs.sum())  # 如果给出的 probs 不规范（总和 > 1），就强制归一化为合法概率分布（总和 = 1）
        K = len(probs)
        self.prob = torch.zeros(K)   # # 存储重新计算后的采样概率。 self.prob[i]: 最终使用的采样概率表，介于 0 和 1 之间
        self.alias = torch.LongTensor([0]*K)  # 存储 alias 候选索引（备用）。 self.alias[i]: 如果不走 prob[i] 分支，就跳转到的 alias 候选（i 的“备胎”）

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob  # 将原始概率乘以 K，使得平均值为 1
            if self.prob[kk] < 1.0:  # 将 <1 的放进 smaller，≥1 的放进 larger，后续用于平衡构建表格
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """  # 采样 N 个索引值
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj  # 它在每次调用时返回 N 个索引值，用于替代普通 multinomial 方法，更快地进行负样本选择。
if __name__ =='__main__':
    a = torch.ones(5501)
    b = AliasMethod(a)
    print(b.prob.shape)
