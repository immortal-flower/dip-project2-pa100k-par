"""E0、九组 SE、中心门控九组 SE 的同图成对错误分析（不训练）。

复用 E0 的 e0_val_predictions.pt；若服务器缺少此文件，则由 E0 best.pt
补算一次。三模型结果均按 val.csv 图片文件名对齐，并用既有 26 属性
TP/FP/FN/TN 记录做严格一致性检查。
"""
import argparse
from collections import Counter
import html
import json
from pathlib import Path, PurePosixPath
import random
import shutil


FOCUS = ("glasses", "short_sleeve", "long_sleeve",
         "backpack", "shoulder_bag", "handbag")
CHANGES = ("new_fn", "new_fp", "fixed_fn", "fixed_fp")
RELATED = {
    "glasses": ("glasses", "hat"),
    "short_sleeve": ("short_sleeve", "long_sleeve"),
    "long_sleeve": ("short_sleeve", "long_sleeve"),
    "backpack": ("backpack", "shoulder_bag", "handbag"),
    "shoulder_bag": ("backpack", "shoulder_bag", "handbag"),
    "handbag": ("backpack", "shoulder_bag", "handbag"),
}
CASE_LABELS = {
    "new_fn": "E0 对 / 新版漏报",
    "new_fp": "E0 对 / 新版误报",
    "fixed_fn": "E0 漏报 / 新版纠正",
    "fixed_fp": "E0 误报 / 新版纠正",
}


def normalize_path(path):
    path = str(path).replace("\\", "/")
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or ".." in parts or ":" in path:
        raise ValueError(f"不安全的图片相对路径：{path}")
    return "/".join(parts)


def case_type(truth, e0, center):
    """区分新错误和修正错误，避免只挑对新模型不利的图片。"""
    truth, e0, center = bool(truth), bool(e0), bool(center)
    if e0 == center:
        return "both_correct" if e0 == truth else "both_wrong"
    if e0 == truth:
        return "new_fn" if truth else "new_fp"
    return "fixed_fn" if truth else "fixed_fp"


def spec_signature(spec):
    return tuple((g["name"], g["type"], tuple(g["attrs"]),
                  tuple(g["indices"])) for g in spec["groups"])


def dataset_truth(dataset, torch):
    truth = torch.zeros((len(dataset), dataset.spec["num_attrs"]), dtype=torch.bool)
    for row, (_, values) in enumerate(dataset.samples):
        for value, column in zip(values, dataset.col2global):
            truth[row, column] = bool(value)
    paths = [normalize_path(path) for path, _ in dataset.samples]
    if len(set(paths)) != len(paths):
        raise ValueError("val.csv 中有重复图片路径，不能按文件名唯一对齐")
    return paths, truth


def load_e0_cache(path, paths, truth, torch):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if not all(k in cache for k in ("paths", "truth", "pred", "prob")):
        raise ValueError("E0 逐图预测文件缺少 paths/truth/pred/prob")
    old_paths = [normalize_path(p) for p in cache["paths"]]
    if len(old_paths) != len(set(old_paths)) or set(old_paths) != set(paths):
        raise ValueError("E0 逐图预测文件与当前 val.csv 的图片集合不同")
    index = {p: i for i, p in enumerate(old_paths)}
    order = [index[p] for p in paths]
    ordered = {k: cache[k][order].cpu() for k in ("truth", "pred", "prob")}
    shape = tuple(truth.shape)
    if any(tuple(t.shape) != shape for t in ordered.values()):
        raise ValueError(f"E0 预测形状与当前验证集不一致，期望 {shape}")
    if not torch.equal(ordered["truth"].bool(), truth):
        raise ValueError("E0 逐图预测的真实标签与当前 dataset.py/val.csv 不一致")
    return {"paths": paths, "truth": truth, "pred": ordered["pred"].bool(),
            "prob": ordered["prob"].float()}


def predict_checkpoint(path, expected_attention, dataset, paths, truth,
                       spec, image_size, batch_size, workers, device, torch):
    from torch.utils.data import DataLoader
    from model import PAResNet

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args["attention"] != expected_attention:
        raise ValueError(f"{path} 的 attention 不是 {expected_attention}")
    if tuple(args["image_size"]) != tuple(image_size):
        raise ValueError(f"{path} 的图像尺寸与 E0 不同")
    if spec_signature(checkpoint["spec"]) != spec_signature(spec):
        raise ValueError(f"{path} 的属性词表或顺序与 val.csv 不同")

    model = PAResNet(spec, hidden=args["hidden"], drop=args["drop"],
                     attention=args["attention"],
                     se_reduction=args["se_reduction"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    all_pred, all_prob = [], []
    position = 0
    with torch.no_grad():
        for images, target in loader:
            count = len(images)
            if not torch.equal(target.bool(), truth[position:position + count]):
                raise ValueError("DataLoader 顺序或标签与 val.csv 不一致")
            logits = model(images.to(device, non_blocking=True))
            pred = torch.zeros((count, spec["num_attrs"]), dtype=torch.bool)
            prob = torch.zeros((count, spec["num_attrs"]), dtype=torch.float32)
            for group in spec["groups"]:
                idx = group["indices"]
                raw = logits[group["name"]].float().cpu()
                if group["type"] == "softmax":
                    values = torch.softmax(raw, dim=1)
                    prob[:, idx] = values
                    pred[:, idx] = torch.nn.functional.one_hot(
                        values.argmax(dim=1), num_classes=len(idx)).bool()
                else:
                    prob[:, idx] = torch.sigmoid(raw)
                    pred[:, idx] = raw > 0
            all_pred.append(pred)
            all_prob.append(prob)
            position += count
    if position != len(dataset):
        raise ValueError("模型推理没有覆盖全部验证图片")
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"paths": paths, "truth": truth, "pred": torch.cat(all_pred),
            "prob": torch.cat(all_prob)}


def verify_existing_metrics(results, spec, reference_path, torch):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    by_mode = {r["attention"]: r for r in reference}
    for mode, result in results.items():
        if mode not in by_mode:
            raise ValueError(f"统一指标文件缺少模型：{mode}")
        truth, pred = result["truth"], result["pred"]
        if by_mode[mode]["samples"] != len(truth):
            raise ValueError(f"{mode} 验证集样本数与旧评估不符")
        for attr, idx in spec["attr2idx"].items():
            y, p = truth[:, idx], pred[:, idx]
            actual = {"TP": int((y & p).sum()), "FP": int((~y & p).sum()),
                      "FN": int((y & ~p).sum()), "TN": int((~y & ~p).sum())}
            expected = by_mode[mode]["attributes"][attr]
            if any(actual[k] != expected[k] for k in actual):
                old_counts = {k: expected[k] for k in actual}
                raise ValueError(f"{mode}/{attr} 与旧评估计数不一致："
                                 f"当前 {actual}，旧版 {old_counts}")
        print(f"核对通过：{mode} 的 26 属性混淆计数与既有记录一致", flush=True)
    return by_mode


def choose_examples(indices, probabilities, limit, seed):
    """同时展示临界、高置信度与随机案例，避免只挑极端图片。"""
    if len(indices) <= limit:
        return list(indices)
    rng = random.Random(seed)
    near = sorted(indices, key=lambda i: abs(probabilities[i] - 0.5))
    strong = list(reversed(near))
    chosen = []
    for source in (near[:2], strong[:2]):
        for i in source:
            if i not in chosen and len(chosen) < limit:
                chosen.append(i)
    remaining = [i for i in indices if i not in chosen]
    chosen += rng.sample(remaining, min(limit - len(chosen), len(remaining)))
    return chosen


def make_reports(output, data_root, paths, results, spec, focus,
                 existing_metrics, image_limit):
    predictions_file = output / "paired_predictions.jsonl"
    changes_file = output / "case_changes.jsonl"
    selected_file = output / "selected_cases.jsonl"
    images_dir = output / "images"
    images_dir.mkdir()
    counts = {}
    gallery = []
    selected_rows = []
    copied = {}
    root = data_root.resolve()
    with (predictions_file.open("w", encoding="utf-8") as full,
          changes_file.open("w", encoding="utf-8") as changes):
        for attr in focus:
            idx = spec["attr2idx"][attr]
            rows = []
            category_count = Counter()
            group_status = Counter()
            for i, path in enumerate(paths):
                y = int(results["none"]["truth"][i, idx])
                row = {"image": path, "attribute": attr, "truth": y}
                for mode, prefix in (("none", "e0"), ("se_group", "group"),
                                     ("se_group_center", "center")):
                    row[prefix + "_pred"] = int(results[mode]["pred"][i, idx])
                    row[prefix + "_prob"] = round(
                        float(results[mode]["prob"][i, idx]), 6)
                row["case"] = case_type(y, row["e0_pred"], row["center_pred"])
                row["group_correct"] = int(row["group_pred"] == y)
                line = json.dumps(row, ensure_ascii=False)
                full.write(line + "\n")
                if row["case"] in CHANGES:
                    changes.write(line + "\n")
                category_count[row["case"]] += 1
                if row["case"] in ("new_fn", "new_fp"):
                    group_status["group_correct" if row["group_correct"]
                                 else "group_wrong"] += 1
                rows.append(row)
            counts[attr] = {"cases": dict(category_count),
                            "new_errors_group_status": dict(group_status),
                            "metrics": {
                                mode: {k: existing_metrics[mode]["attributes"][attr][k]
                                       for k in ("precision", "recall", "F1", "mA")}
                                for mode in results}}
            for case in CHANGES:
                indices = [i for i, row in enumerate(rows) if row["case"] == case]
                sampled = choose_examples(
                    indices, [r["center_prob"] for r in rows], image_limit,
                    seed=42 + idx * 10 + CHANGES.index(case))
                for i in sampled:
                    row = rows[i].copy()
                    row["related"] = {
                        other: {
                            "truth": int(results["none"]["truth"][i, spec["attr2idx"][other]]),
                            **{mode: round(float(results[mode]["prob"][i, spec["attr2idx"][other]]), 4)
                               for mode in results},
                        }
                        for other in RELATED[attr]
                    }
                    rel = PurePosixPath(row["image"])
                    source = (root / Path(*rel.parts)).resolve()
                    if not source.is_relative_to(root) or not source.is_file():
                        raise FileNotFoundError(f"找不到安全的案例图片：{source}")
                    if row["image"] not in copied:
                        target = images_dir / (str(len(copied)).zfill(4) +
                                               "_" + source.name)
                        shutil.copy2(source, target)
                        copied[row["image"]] = "images/" + target.name
                    row["local_image"] = copied[row["image"]]
                    selected_rows.append(row)
                    gallery.append(row)
    with selected_file.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {"dataset": "PA-100K validation", "samples": len(paths),
               "focus_attributes": list(focus), "selected_images": len(copied),
               "selected_rows": len(selected_rows), "per_attribute": counts,
               "notes": ["new_fn/new_fp：E0 正确而中心门控 SE 错误",
                         "fixed_fn/fixed_fp：E0 错误而中心门控 SE 正确",
                         "group_correct 仅反映九组 SE 在新版新增错误图片上的表现",
                         "图片样例用于定性分析，不能代表整体验证集分布"]}
    (output / "paired_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# E0 与新版分组 SE：同图成对错误分析", "",
             "验证集 10,000 张；按图片文件名对齐，三个模型共用相同标签、尺寸和二值阈值 0.5。",
             "年龄/朝向仍按原本的 softmax argmax；本报告聚焦六个二值属性。", "",
             "| 属性 | 新增漏报 | 新增误报 | 修正漏报 | 修正误报 | 新增错误中九组 SE 判对 | E0 F1 | 九组 F1 | 新版 F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for attr in focus:
        entry = counts[attr]
        c, m = entry["cases"], entry["metrics"]
        lines.append(f"| {attr} | {c.get('new_fn', 0)} | {c.get('new_fp', 0)} | "
                     f"{c.get('fixed_fn', 0)} | {c.get('fixed_fp', 0)} | "
                     f"{entry['new_errors_group_status'].get('group_correct', 0)} | "
                     f"{m['none']['F1'] * 100:.2f}% | "
                     f"{m['se_group']['F1'] * 100:.2f}% | "
                     f"{m['se_group_center']['F1'] * 100:.2f}% |")
    lines += ["", "新增/修正数量按 *E0 与新版 E2* 的逐图判断差异统计，"
              "不能直接当作 F1 的百分点变化；九组 SE 判对数只用于初步区分门控幅度与分组问题。",
              "全部逐图数据在 `paired_predictions.jsonl`，发生翻转的样本在 `case_changes.jsonl`。",
              "打开 `case_gallery.html` 浏览抽样图片；请人工记录遮挡、目标大小和标签疑点。", "",
              "**实验边界：** 这些验证集案例可用于提出结构假设；"
              "若继续试 26 个属性各一套 SE，应保持其余设置一致，并用保留的测试集确认结论。", ""]
    (output / "paired_report.md").write_text("\n".join(lines), encoding="utf-8")

    html_parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
                  "<title>PA-100K 成对错误案例</title>",
                  "<style>body{font:16px system-ui;margin:24px;background:#f7f9fc;color:#172536}"
                  "h1,h2{margin-top:28px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}"
                  ".card{background:white;border:1px solid #d8e1eb;border-radius:10px;padding:12px}"
                  "img{display:block;max-height:250px;max-width:100%;margin:auto}"
                  "small{display:block;line-height:1.5;overflow-wrap:anywhere}</style><body>",
                  "<h1>E0 vs 新版 E2：同图案例</h1>",
                  "<p>每类混合抽取临界、高置信度和随机案例；仅供人工观察，不代表总体频率。"
                  "完整计数见 paired_report.md。</p>"]
    for attr in focus:
        for case in CHANGES:
            subset = [r for r in gallery if r["attribute"] == attr and r["case"] == case]
            total = counts[attr]["cases"].get(case, 0)
            html_parts.append(f"<h2>{html.escape(attr)}：{CASE_LABELS[case]}（共 {total}，展示 {len(subset)}）</h2>")
            html_parts.append("<div class='grid'>")
            for row in subset:
                path = html.escape(row["image"], quote=True)
                src = html.escape(row["local_image"], quote=True)
                detail = (f"真实={row['truth']} ｜ E0={row['e0_pred']} "
                          f"({row['e0_prob']:.3f}) ｜ 九组={row['group_pred']} "
                          f"({row['group_prob']:.3f}) ｜ 新版={row['center_pred']} "
                          f"({row['center_prob']:.3f})")
                related = "<br>".join(
                    html.escape(f"{name}: 真={values['truth']} E0={values['none']:.2f} "
                                f"九组={values['se_group']:.2f} "
                                f"新版={values['se_group_center']:.2f}")
                    for name, values in row["related"].items())
                html_parts.append(f"<div class='card'><img loading='lazy' src='{src}' alt='{path}'>"
                                  f"<small>{path}<br>{html.escape(detail)}<br>{related}</small></div>")
            html_parts.append("</div>")
    html_parts.append("</body></html>")
    (output / "case_gallery.html").write_text("\n".join(html_parts), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, default=Path("attributes.yaml"))
    parser.add_argument("--e0-pred", type=Path, required=True)
    parser.add_argument("--e0-ckpt", type=Path, required=True)
    parser.add_argument("--group-ckpt", type=Path, required=True)
    parser.add_argument("--center-ckpt", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--images-per-case", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"为保护原有分析结果，拒绝覆盖：{args.out}")
    if args.images_per_case < 1:
        raise ValueError("--images-per-case 至少为 1")

    import torch
    from dataset import PA100KDataset, build_transform, load_spec

    spec = load_spec(args.attributes)
    if spec["num_attrs"] != 26 or any(a not in spec["attr2idx"] for a in FOCUS):
        raise ValueError("属性词表与预期的 PA-100K 26 属性不符")
    e0_checkpoint = torch.load(args.e0_ckpt, map_location="cpu", weights_only=False)
    if e0_checkpoint["args"]["attention"] != "none":
        raise ValueError("E0 checkpoint 未关闭 SE")
    if spec_signature(e0_checkpoint["spec"]) != spec_signature(spec):
        raise ValueError("E0 checkpoint 与当前属性词表不一致")
    image_size = e0_checkpoint["args"]["image_size"]
    dataset = PA100KDataset(str(args.data_root), "val.csv", spec,
                            build_transform(False, image_size))
    paths, truth = dataset_truth(dataset, torch)
    if len(dataset) != 10000:
        raise ValueError(f"验证集应为 10000 张，实际 {len(dataset)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"验证集 {len(dataset)} 张；设备 {device}", flush=True)
    results = {}
    reused_e0 = False
    if args.e0_pred.is_file():
        try:
            results["none"] = load_e0_cache(args.e0_pred, paths, truth, torch)
            reused_e0 = True
            print("E0：复用已有 10000 张逐图预测", flush=True)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
            print(f"E0 逐图预测文件不兼容，将由 best.pt 补算：{exc}", flush=True)
    if not reused_e0:
        print("E0：由 best.pt 计算逐图预测", flush=True)
        results["none"] = predict_checkpoint(
            args.e0_ckpt, "none", dataset, paths, truth, spec, image_size,
            args.batch_size, args.workers, device, torch)
    for mode, path in (("se_group", args.group_ckpt),
                       ("se_group_center", args.center_ckpt)):
        results[mode] = predict_checkpoint(
            path, mode, dataset, paths, truth, spec, image_size,
            args.batch_size, args.workers, device, torch)
        print(f"{mode}：已获得 {len(paths)} 张逐图预测", flush=True)
    try:
        existing = verify_existing_metrics(
            results, spec, args.reference_metrics, torch)
    except ValueError:
        if not reused_e0:
            raise
        print("E0 旧预测与统一指标不一致，改由 E0 best.pt 复核", flush=True)
        results["none"] = predict_checkpoint(
            args.e0_ckpt, "none", dataset, paths, truth, spec, image_size,
            args.batch_size, args.workers, device, torch)
        existing = verify_existing_metrics(
            results, spec, args.reference_metrics, torch)

    args.out.mkdir(parents=True)
    for mode, result in results.items():
        torch.save(result, args.out / f"{mode}_val_predictions.pt")
    summary = make_reports(args.out, args.data_root, paths, results, spec,
                           FOCUS, existing, args.images_per_case)
    print(f"已保存分析到 {args.out}", flush=True)
    for attr, entry in summary["per_attribute"].items():
        c = entry["cases"]
        print(f"{attr:14s} 新增 FN={c.get('new_fn', 0):4d} "
              f"新增 FP={c.get('new_fp', 0):4d} "
              f"修正 FN={c.get('fixed_fn', 0):4d} "
              f"修正 FP={c.get('fixed_fp', 0):4d}", flush=True)
    print(f"已导出 {summary['selected_images']} 张不同案例图片", flush=True)


if __name__ == "__main__":
    main()
