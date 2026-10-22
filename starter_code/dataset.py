"""PA100K 数据集加载与图像预处理。

每个样本返回 (image, target)：
  image  : (3, H, W) float tensor，已归一化
  target : (num_attrs,) float tensor，26 个全局属性标签（0/1）

支持原始 PA-100K 的 train/val/test.csv + data/*.jpg，也支持课程
train/val/test_list.txt。两种来源都按列名转换为 ATTR_ORDER 的语义。
"""
import argparse
import csv
import os

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ImageNet 归一化参数：使用预训练模型时统一使用
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 模型使用的 26 个属性顺序；它不等于原始 CSV 或课程列表文件的列顺序。
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

# 原始 CSV 中的列名。Female 与模型的 male 语义相反。
CSV_COLUMNS = {
    "hat": "Hat", "glasses": "Glasses",
    "short_sleeve": "ShortSleeve", "long_sleeve": "LongSleeve",
    "upper_stripe": "UpperStride", "upper_logo": "UpperLogo",
    "upper_plaid": "UpperPlaid", "upper_splice": "UpperSplice",
    "lower_stripe": "LowerStripe", "lower_pattern": "LowerPattern",
    "long_coat": "LongCoat", "trousers": "Trousers",
    "shorts": "Shorts", "skirt_dress": "Skirt&Dress",
    "boots": "boots", "backpack": "Backpack",
    "shoulder_bag": "ShoulderBag", "handbag": "HandBag",
    "holding": "HoldObjectsInFront", "age_lt18": "AgeLess18",
    "age_18_60": "Age18-60", "age_gt60": "AgeOver60",
    "male": "Female", "front": "Front", "back": "Back", "side": "Side",
}

# 课程 *_list.txt 的真实列顺序，与 starter code 原有的 ATTR_ORDER 不同。
LIST_COLUMNS = [
    "Hat", "Glasses", "ShortSleeve", "LongSleeve", "UpperStride",
    "UpperLogo", "UpperPlaid", "UpperSplice", "LowerStripe", "LowerPattern",
    "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots", "HandBag",
    "ShoulderBag", "Backpack", "HoldObjectsInFront", "AgeOver60",
    "Age18-60", "AgeLess18", "Female", "Front", "Side", "Back",
]


def _ordered_labels(columns):
    """按模型属性顺序读取 0/1 标签，并把 Female 转为 male。"""
    values = []
    for attr in ATTR_ORDER:
        value = int(columns[CSV_COLUMNS[attr]])
        if value not in (0, 1):
            raise ValueError(f"{attr} must be 0 or 1, got {value}")
        values.append(1 - value if attr == "male" else value)
    return values


def load_spec(path="attributes.yaml"):
    """加载全局属性词表。

    返回:
      groups   : [{name, type, attrs, indices}]，indices 为该组在全局标签向量中的位置
      attr2idx : {属性名: 全局索引}
      num_attrs: 属性总数（26）
      loss_weights: {组名: 权重}
    """
    # attributes.yaml contains Chinese comments.  Specify UTF-8 explicitly so
    # Windows does not try to decode it with the system ANSI code page (GBK).
    with open(path, encoding="utf-8") as f:
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

    CSV 模式：root 是原始 PA-100K 目录，图片在 root/data/。
    列表模式：root 是课程 pa100k 目录，图片在 root/train|val|test/。
    """

    def __init__(self, root, list_file, spec, transform=None):
        self.root = root
        self.transform = transform
        self.spec = spec
        self.col2global = [spec["attr2idx"][a] for a in ATTR_ORDER]
        self.samples = []
        source = os.path.join(root, list_file)
        if list_file.endswith(".csv"):
            with open(source, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                required = {"Image", *CSV_COLUMNS.values()}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"Missing CSV columns: {sorted(missing)}")
                for row in reader:
                    self.samples.append((os.path.join("data", row["Image"]),
                                         _ordered_labels(row)))
        else:
            with open(source, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    path, raw = line.split("\t")
                    vals = raw.split(",")
                    if len(vals) != len(LIST_COLUMNS):
                        raise ValueError(f"Expected 26 labels for {path}")
                    columns = dict(zip(LIST_COLUMNS, vals))
                    self.samples.append((path, _ordered_labels(columns)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, vals = self.samples[i]
        with Image.open(os.path.join(self.root, path)) as source:
            img = source.convert("RGB")
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


def label_matrix(dataset):
    """Return all labels as an ``N x num_attrs`` float tensor."""
    source = torch.tensor([values for _, values in dataset.samples], dtype=torch.float32)
    labels = torch.zeros((len(dataset), dataset.spec["num_attrs"]), dtype=torch.float32)
    labels[:, dataset.col2global] = source
    return labels


def compute_sample_weights(dataset, power=0.5, max_weight=10.0):
    """Compute conservative rare-positive sampling weights for E3.

    For attribute ``j`` with positive rate ``p_j``, its rarity factor is
    ``sqrt((1-p_j)/p_j)`` by default.  Each image receives the maximum factor
    among its positive attributes.  A floor of 1 avoids down-weighting common
    positives, while ``max_weight`` prevents a handful of rare examples from
    dominating an epoch.  Returned image weights are normalized to mean 1.
    """
    if power <= 0:
        raise ValueError("power must be positive")
    if max_weight < 1:
        raise ValueError("max_weight must be at least 1")

    labels = label_matrix(dataset)
    positive_rate = labels.mean(dim=0)
    eps = 1.0 / max(len(dataset), 1)
    rarity = ((1.0 - positive_rate) / positive_rate.clamp(min=eps)).pow(power)
    rarity = rarity.clamp(min=1.0, max=max_weight)
    active_factors = torch.where(labels > 0, rarity.unsqueeze(0), torch.ones_like(labels))
    sample_weights = active_factors.max(dim=1).values
    sample_weights = sample_weights / sample_weights.mean().clamp(min=eps)
    return sample_weights.to(torch.double), positive_rate, rarity


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "pa100k"))
    args = parser.parse_args()
    spec = load_spec(os.path.join(os.path.dirname(__file__), "..", "attributes.yaml"))
    train_file = "train.csv" if os.path.isfile(os.path.join(args.data_root, "train.csv")) else "train_list.txt"
    ds = PA100KDataset(args.data_root, train_file, spec)
    print(f"训练集样本数: {len(ds)}")
    print("各属性正例率:")
    attribute_stats(ds)
