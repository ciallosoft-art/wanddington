# Experiment Results

每次运行 `benchmark.py` 自动保存到此目录。

## 目录结构

```
results/
├── runs/
│   ├── run_20260611_154230_all_normal.json     # 完整原始结果
│   ├── run_20260611_160000_waddington_scrambled.json
│   └── ...
└── summary.csv     # 所有 run 的关键指标汇总（每 run 一行）
```

## summary.csv 列说明

| 列 | 说明 |
|----|------|
| `run_id` | 唯一标识，格式 `run_YYYYMMDD_HHMMSS` |
| `timestamp` | ISO 8601 时间戳 |
| `git_commit` | 运行时的 git commit hash |
| `ranker` | 使用的 ranker 名称 |
| `scramble_genes` | 是否启用基因名打乱消融 |
| `filter_essential` | 是否过滤 CEGv2 必需基因 |
| `rounds` / `trials` | 实验轮数 / 随机种子数 |
| `{DS}_r5_mean` | 各数据集第 5 轮 hit_ratio 均值 |
| `{DS}_r5_std` | 对应标准差 |
| `avg_r5` | 7 数据集平均 hit_ratio@R5 |
| `avg_auc` | 7 数据集平均 AUC |

## 快速查看

```bash
# 查看所有历史 run
python workspace/evaluation/results_summary.py

# 只看 waddington ranker 的 run
python workspace/evaluation/results_summary.py --ranker waddington

# 对比 normal vs scrambled
python workspace/evaluation/results_summary.py --compare-scramble
```
