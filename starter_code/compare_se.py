"""核对 E0/E2 是否只有 SE 开关不同，并汇总结构实验结果。

用法：python starter_code/compare_se.py runs/E0_baseline runs/E2_se
"""
import argparse
import json
from pathlib import Path


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_run(path):
    config = read_json(path / "config.json")
    summary = read_json(path / "summary.json")
    with (path / "metrics.jsonl").open(encoding="utf-8") as f:
        epochs = [json.loads(line) for line in f if line.strip()]
    if not epochs:
        raise ValueError(f"No epoch metrics found in {path}")
    return config, summary, epochs


def compare_config(base, se):
    ignored = {"attention", "out"}
    left = {k: v for k, v in base["args"].items() if k not in ignored}
    right = {k: v for k, v in se["args"].items() if k not in ignored}
    differences = {k: (left.get(k), right.get(k))
                   for k in left.keys() | right.keys() if left.get(k) != right.get(k)}
    if differences:
        raise ValueError(f"E0/E2 training configurations differ: {differences}")
    if base["args"]["attention"] != "none" or se["args"]["attention"] != "se":
        raise ValueError("Expected E0 attention=none and E2 attention=se")
    if base.get("data_format") != se.get("data_format"):
        raise ValueError("E0/E2 use different data formats")
    if base.get("selection_metric") != se.get("selection_metric"):
        raise ValueError("E0/E2 use different checkpoint selection metrics")


def percentage(value):
    return f"{value * 100:.2f}%"


def render_report(base, se, base_epochs, se_epochs):
    lines = ["# E0 与 E2（SE）对比", "",
             "E0/E2 的训练配置已核对：仅 `attention` 和输出目录不同。",
             "最佳模型按验证集组均 AttrAcc 选择。", "",
             "| 实验 | 参数量 | 平均训练秒/轮 | 最佳 epoch | 最佳组均 AttrAcc |",
             "|---|---:|---:|---:|---:|"]
    for label, summary, epochs in (("E0", base, base_epochs), ("E2", se, se_epochs)):
        avg_time = sum(e["train_seconds"] for e in epochs) / len(epochs)
        lines.append(f"| {label} | {summary['parameters']:,} | {avg_time:.1f} | "
                     f"{summary['best_epoch']} | {percentage(summary['best_score'])} |")

    lines += ["", "## 各属性组", "",
              "| 属性组 | E0 | E2 | E2 - E0（百分点） |",
              "|---|---:|---:|---:|"]
    for name, base_value in base["best_group_metrics"].items():
        se_value = se["best_group_metrics"][name]
        lines.append(f"| {name} | {percentage(base_value)} | {percentage(se_value)} | "
                     f"{(se_value - base_value) * 100:+.2f} |")

    focus = ("accessory", "upper", "lower", "bag")
    lines += ["", "## 重点属性", "",
              "| 属性组 | 属性 | E0 | E2 | E2 - E0（百分点） |",
              "|---|---|---:|---:|---:|"]
    # 属性与组的对应关系按 attributes.yaml 固定词表记录。
    groups = {
        "accessory": ("hat", "glasses"),
        "upper": ("short_sleeve", "long_sleeve", "upper_stripe",
                  "upper_logo", "upper_plaid", "upper_splice"),
        "lower": ("lower_stripe", "lower_pattern", "long_coat",
                  "trousers", "shorts", "skirt_dress"),
        "bag": ("backpack", "shoulder_bag", "handbag"),
    }
    for group in focus:
        for name in groups[group]:
            base_value = base["best_attr_metrics"][name]
            se_value = se["best_attr_metrics"][name]
            lines.append(f"| {group} | {name} | {percentage(base_value)} | "
                         f"{percentage(se_value)} | {(se_value - base_value) * 100:+.2f} |")
    lines += ["", "注：此表使用 starter_code/evaluate.py 当前返回的 AttrAcc；"
              "A 的 mA、Precision、Recall、F1 完成后，需对两个 checkpoint 统一复评。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path, help="E0 输出目录")
    parser.add_argument("se", type=Path, help="E2 输出目录")
    parser.add_argument("--out", type=Path, help="可选：保存 Markdown 报告")
    args = parser.parse_args()
    base_config, base_summary, base_epochs = read_run(args.baseline)
    se_config, se_summary, se_epochs = read_run(args.se)
    compare_config(base_config, se_config)
    report = render_report(base_summary, se_summary, base_epochs, se_epochs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Saved {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
