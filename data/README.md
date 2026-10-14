# 数据放置说明

将教师分发的 PA100K 数据集解压后放到本目录：

```
data/
└── pa100k/
    ├── train/            # 80,000 张训练图
    ├── val/              # 10,000 张验证图
    ├── test/             # 10,000 张测试图
    ├── train_list.txt
    ├── val_list.txt
    └── test_list.txt
```

放置完成后运行以下命令自检：

```bash
cd ../starter_code
python dataset.py
```

应看到 `训练集样本数: 80000` 以及各属性正例率统计。
