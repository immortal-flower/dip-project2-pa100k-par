"""行人属性识别模型（baseline、共享 SE 与分组 SE 变体）。

结构：CNN backbone 提取特征 -> 每个属性组一个独立分类头。
输入 (B, 3, 256, 128) 图片，输出 {组名: (B, 组内属性数) logits}。

鼓励修改的方向（见 README 第七节）：
  - 换更深的 backbone（ResNet34/50）
  - 在 backbone 和 head 之间加入注意力模块（如 SE）
  - 修改 head 结构，为 margin softmax 等损失改造留出特征接口
"""
import math

import torch
import torch.nn as nn
from torchvision import models


def make_head(in_dim, out_dim, hidden=256, drop=0.3):
    """单个属性组的分类头：Linear -> ReLU -> Dropout -> Linear。"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(drop),
        nn.Linear(hidden, out_dim),
    )


class SEBlock(nn.Module):
    """为池化后的特征向量学习逐通道权重。"""

    def __init__(self, dim, reduction=16, centered=False,
                 residual_alpha=None, adaptive_alpha=False):
        super().__init__()
        if reduction < 1 or dim // reduction < 1:
            raise ValueError("SE reduction must be between 1 and feature dimension")
        if residual_alpha is not None and not 0 <= residual_alpha <= 1:
            raise ValueError("SE residual_alpha must be between 0 and 1")
        if adaptive_alpha and residual_alpha is not None:
            raise ValueError("adaptive_alpha and fixed residual_alpha are exclusive")
        self.centered = centered
        self.residual_alpha = residual_alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            # sigmoid(-log(3)) = 0.25, matching the fixed residual gate initially.
            self.alpha_logit = nn.Parameter(torch.tensor(-math.log(3.0)))
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.gate(x)
        if self.residual_alpha is not None or self.adaptive_alpha:
            # alpha=0 为恒等映射；固定 0.25 时系数范围为 (0.75, 1.25)。
            alpha = (torch.sigmoid(self.alpha_logit) if self.adaptive_alpha
                     else self.residual_alpha)
            weights = 1 + alpha * (2 * weights - 1)
        elif self.centered:
            weights = 2 * weights
        # sigmoid 只是注意力激活，不改变 BCE/交叉熵损失函数。
        return x * weights


class PAResNet(nn.Module):
    """ResNet18 + 可选 SE 特征门控 + 分组属性头。

    ``se`` 为原始 E2：九组共用同一个 512 维通道门控。
    ``se_group`` 为改进 E2：各组使用独立门控，分类头与训练逻辑不变。
    ``se_group_center`` 在分组 SE 基础上，仅将门控改为 2*sigmoid(z)。
    ``se_group_residual025`` 只把中心门控的幅度缩为 0.25。
    ``se_group_adaptive025`` 九组分别学习门控幅度，均从 0.25 开始。
    ``se_attr_residual025`` 26 个属性分别拥有固定幅度 0.25 的 SE。
    """

    def __init__(self, spec, hidden=256, drop=0.3, attention="none",
                 se_reduction=16, pretrained=True):
        super().__init__()
        self.groups = spec["groups"]
        if attention not in ("none", "se", "se_group", "se_group_center",
                             "se_group_residual025", "se_group_adaptive025",
                             "se_attr_residual025"):
            raise ValueError(f"Unsupported attention: {attention}")
        self.attention_type = attention

        # ResNet18 去掉最后的 fc 分类层，输出 (B, 512) 全局特征
        # 使用 ImageNet 预训练权重（迁移学习）
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        self.feat_dim = 512

        # 每个属性组一个独立分类头
        self.heads = nn.ModuleDict({
            g["name"]: make_head(self.feat_dim, len(g["attrs"]), hidden, drop)
            for g in self.groups
        })
        # 先构建分类头，再构建 SE，确保固定种子时各实验的 backbone、
        # 分类头初值相同。隔离 SE 初始化的随机数，保持后续数据打乱顺序一致。
        self.attention = nn.Identity()
        self.group_attention = None
        self.attr_attention = None
        if attention != "none":
            with torch.random.fork_rng(devices=[]):
                if attention == "se":
                    self.attention = SEBlock(self.feat_dim, se_reduction)
                elif attention == "se_attr_residual025":
                    self.attr_attention = nn.ModuleDict({
                        attr: SEBlock(self.feat_dim, se_reduction,
                                      residual_alpha=0.25)
                        for group in self.groups for attr in group["attrs"]
                    })
                else:
                    self.group_attention = nn.ModuleDict({
                        g["name"]: SEBlock(
                            self.feat_dim, se_reduction,
                            centered=(attention == "se_group_center"),
                            residual_alpha=(0.25 if attention ==
                                            "se_group_residual025" else None),
                            adaptive_alpha=(attention ==
                                            "se_group_adaptive025"))
                        for g in self.groups
                    })

    def forward(self, x):
        feats = self.attention(self.backbone(x))
        # 注意: 这里输出的是 logits，不加 sigmoid/softmax，
        # 由损失函数内部处理（数值更稳定）
        if self.group_attention is not None:
            return {name: head(self.group_attention[name](feats))
                    for name, head in self.heads.items()}
        if self.attr_attention is not None:
            outputs = {}
            for group in self.groups:
                name = group["name"]
                head = self.heads[name]
                gated = torch.stack([self.attr_attention[attr](feats)
                                     for attr in group["attrs"]], dim=1)
                hidden = head[1](head[0](gated))
                # Preserve one shared dropout mask per group as in the trained E5.
                mask = head[2](torch.ones_like(hidden[:, 0, :]))
                hidden = hidden * mask.unsqueeze(1)
                outputs[name] = (torch.einsum("bah,ah->ba", hidden,
                                              head[3].weight)
                                 + head[3].bias)
            return outputs
        return {name: head(feats) for name, head in self.heads.items()}
