# 数据放置说明

数据集下载地址：[Kaggle PA-100K](https://www.kaggle.com/datasets/yuulind/pa-100k)。

推荐直接使用完整的原始 PA-100K 数据集，不必复制或重排图片：

```
PA-100K/
├── data/                  # 100,000 张图片，文件名 000001.jpg～100000.jpg
├── train.csv              # 80,000 条训练标签
├── val.csv                # 10,000 条验证标签
└── test.csv               # 10,000 条测试标签
```

从仓库根目录运行自检：

```powershell
python starter_code/dataset.py --data-root "D:\ShuZiTuXiangChuLi_Lab\挑战项目2\PA-100K"
```

应看到 `训练集样本数: 80000` 以及各属性正例率统计。`dataset.py` 按 CSV 列名映射属性，尤其会把 `Female` 反转成模型的 `male`。课程旧版 `*_list.txt` 的列顺序与代码中的 `ATTR_ORDER` 不完全一致，因此也会按真实列顺序转换；旧版资源包中的零字节图片仍然不可用。
