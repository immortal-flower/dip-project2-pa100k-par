"""PA100K 数据集加载与图像预处理（starter code）。

每个样本返回 (image, target)：
  image  : (3, H, W) float tensor，已归一化
  target : (num_attrs,) float tensor，26 个全局属性标签（0/1）

列表文件格式：每行 "相对路径<TAB>v1,v2,...,v26"，v_i ∈ {0, 1}。
标签列与 ATTR_ORDER 一一对应。
"""
import os

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ImageNet 归一化参数：使用预训练模型时统一使用
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# PA100K 列表文件中 26 列标签对应的属性名（顺序固定，不可改动）
ATTR_ORDER = [
    "hat", "glasses",
    "short_sleeve", "long_sleeve", "upper_stripe", "upper_logo",
    "upper_plaid", "upper_splice",
    "lower_stripe", "lower_pattern", "long_coat", "trousers",
    "shorts", "skirt_dress",
    "boots",
    "backpack", "shoulder_bag", "handbag",
    "holding",
    "age_lt18", "age_18_60", "age_gt60",
    "male",
    "front", "back", "side",
]


def load_spec(path="attributes.yaml"):
    """加载全局属性词表。

    返回:
      groups   : [{name, type, attrs, indices}]，indices 为该组在全局标签向量中的位置
      attr2idx : {属性名: 全局索引}
      num_attrs: 属性总数（26）
      loss_weights: {组名: 权重}
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    groups, offset = [], 0
    for name, g in cfg["groups"].items():
        attrs = list(g["attrs"])
        indices = list(range(offset, offset + len(attrs)))
        groups.append({"name": name, "type": g["type"],
                       "attrs": attrs, "indices": indices})
        offset += len(attrs)
    attr2idx = {a: i for g in groups for a, i in zip(g["attrs"], g["indices"])}
    return {
        "groups": groups,
        "attr2idx": attr2idx,
        "num_attrs": offset,
        "loss_weights": cfg.get("loss_weights", {}),
    }


def build_transform(train, image_size):
    """构建图像预处理/增强流水线。image_size: [高, 宽]。

    baseline 只使用了随机水平翻转。鼓励尝试更多增强并做对比实验：
      - T.ColorJitter(...)   模拟光照/白平衡变化
      - T.RandomErasing(...) 模拟局部遮挡（需放在 ToTensor 之后）
    注意: RandomHorizontalFlip 对 orientation(front/back/side) 属性的影响，
         是一个值得在报告中分析的问题。
    """
    h, w = image_size
    if train:
        return T.Compose([
            T.Resize((h, w)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PA100KDataset(Dataset):
    """PA100K 数据集。

    把列表文件中的 26 列标签映射到全局属性向量 target。
    """

    def __init__(self, root, list_file, spec, transform=None):
        self.root = root
        self.transform = transform
        self.spec = spec
        self.col2global = [spec["attr2idx"][a] for a in ATTR_ORDER]
        self.samples = []
        with open(os.path.join(root, list_file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, vals = line.split("\t")
                vals = [int(v) for v in vals.split(",")]
                assert len(vals) == len(ATTR_ORDER)
                self.samples.append((path, vals))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, vals = self.samples[i]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = torch.zeros(self.spec["num_attrs"])
        for v, g in zip(vals, self.col2global):
            target[g] = float(v)
        return img, target


def compute_pos_weight(dataset, spec):
    """统计每个属性的 pos_weight = 负例数 / 正例数（选做：类别不平衡处理）。"""
    num_attrs = spec["num_attrs"]
    pos = torch.zeros(num_attrs)
    neg = torch.zeros(num_attrs)
    for _, vals in dataset.samples:
        for v, g in zip(vals, dataset.col2global):
            if v == 1:
                pos[g] += 1
            else:
                neg[g] += 1
    return neg / pos.clamp(min=1)


def attribute_stats(dataset):
    """打印每个属性的正例比例，帮助理解数据集分布。"""
    num_attrs = dataset.spec["num_attrs"]
    pos = torch.zeros(num_attrs)
    total = 0
    for _, vals in dataset.samples:
        for v, g in zip(vals, dataset.col2global):
            pos[g] += v
        total += 1
    for g in dataset.spec["groups"]:
        for a, idx in zip(g["attrs"], g["indices"]):
            print(f"  {a:16s} 正例率: {pos[idx] / total * 100:5.2f}%")


if __name__ == "__main__":
    # 第 1 周可直接运行本文件，观察数据集分布与样例图片
    spec = load_spec(os.path.join(os.path.dirname(__file__), "..", "attributes.yaml"))
    ds = PA100KDataset(os.path.join(os.path.dirname(__file__), "..", "data", "pa100k"),
                       "train_list.txt", spec)
    print(f"训练集样本数: {len(ds)}")
    print("各属性正例率:")
    attribute_stats(ds)
