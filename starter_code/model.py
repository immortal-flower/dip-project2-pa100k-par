"""行人属性识别模型（baseline 完整实现）。

结构：CNN backbone 提取特征 -> 每个属性组一个独立分类头。
输入 (B, 3, 256, 128) 图片，输出 {组名: (B, 组内属性数) logits}。

鼓励修改的方向（见 README 第七节）：
  - 换更深的 backbone（ResNet34/50）
  - 在 backbone 和 head 之间加入注意力模块（如 SE）
  - 修改 head 结构，为 margin softmax 等损失改造留出特征接口
"""
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

    def __init__(self, dim, reduction=16):
        super().__init__()
        if reduction < 1 or dim // reduction < 1:
            raise ValueError("SE reduction must be between 1 and feature dimension")
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class PAResNet(nn.Module):
    """ResNet18 + 可选 SE 特征门控 + 分组属性头。"""

    def __init__(self, spec, hidden=256, drop=0.3, attention="none",
                 se_reduction=16, pretrained=True):
        super().__init__()
        self.groups = spec["groups"]
        if attention not in ("none", "se"):
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
        # 先构建分类头，再构建 SE，确保固定种子时 E0/E2 的分类头初值相同。
        self.attention = (SEBlock(self.feat_dim, se_reduction) if attention == "se"
                          else nn.Identity())

    def forward(self, x):
        feats = self.attention(self.backbone(x))
        # 注意: 这里输出的是 logits，不加 sigmoid/softmax，
        # 由损失函数内部处理（数值更稳定）
        return {name: head(feats) for name, head in self.heads.items()}
