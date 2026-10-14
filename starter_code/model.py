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


class PAResNet(nn.Module):
    """ResNet18 backbone + 分组属性头（baseline）。"""

    def __init__(self, spec, hidden=256, drop=0.3):
        super().__init__()
        self.groups = spec["groups"]

        # ResNet18 去掉最后的 fc 分类层，输出 (B, 512) 全局特征
        # 使用 ImageNet 预训练权重（迁移学习）
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        self.feat_dim = 512

        # 每个属性组一个独立分类头
        self.heads = nn.ModuleDict({
            g["name"]: make_head(self.feat_dim, len(g["attrs"]), hidden, drop)
            for g in self.groups
        })

    def forward(self, x):
        feats = self.backbone(x)
        # 注意: 这里输出的是 logits，不加 sigmoid/softmax，
        # 由损失函数内部处理（数值更稳定）
        return {name: head(feats) for name, head in self.heads.items()}
