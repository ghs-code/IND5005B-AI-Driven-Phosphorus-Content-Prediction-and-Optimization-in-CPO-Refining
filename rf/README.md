## 1. 目录结构

```text
rf/
├── random_forest.py           # RF 全量特征（45变量）
├── random_forest_4var.py      # RF 核心质量变量（5变量，与OLS对齐）
├── rf_combo_search.py         # 候选变量穷举组合搜索
├── README.md
└── output/
    ├── rf_45var/               # 45变量模型输出
    ├── rf_5var/                # 5变量模型输出
    └── rf_combo_search/        # 组合搜索输出
```

## 2. 三个模型说明

| 模型 | 脚本 | 特征数 | 目的 |
|------|------|--------|------|
| RF 45变量 | `random_forest.py` | 45 | 使用全部可用特征建模，评估整体预测能力 |
| RF 5变量 | `random_forest_4var.py` | 5 | 与OLS使用相同变量，公平对比两种模型 |
| 组合搜索 | `rf_combo_search.py` | 1~6 | 穷举候选变量组合，找最优子集 |

## 3. 运行方式

在 `rf/` 目录下执行：

```bash
# RF 45变量
python3 random_forest.py

# RF 5变量
python3 random_forest_4var.py

# 变量组合搜索
python3 rf_combo_search.py
```

默认输入：`../outputs/model_ready.csv`，输出目录：`output/`。

`random_forest.py` 支持命令行参数：

- `--input`：输入CSV路径（默认 `../outputs/model_ready.csv`）
- `--output-dir`：输出目录（默认 `output`）

## 4. 建模参数

- 训练/测试集划分：80/20，`random_state=42`
- 超参数搜索：GridSearchCV，5折交叉验证
- 评估指标：R²、RMSE、MAE
- 过拟合诊断：训练集 vs 测试集 R²/RMSE 差距

### 参数搜索网格

**45变量 & 5变量（完整网格，540组合）：**

| 参数 | 搜索范围 |
|------|----------|
| n_estimators | 100, 200, 300, 500 |
| max_depth | None, 5, 10, 15, 20 |
| min_samples_split | 2, 5, 10 |
| min_samples_leaf | 1, 2, 4 |
| max_features | sqrt, log2, None |

**组合搜索（简化网格，20组合 × 143种变量组合）：**

| 参数 | 搜索范围 |
|------|----------|
| n_estimators | 100, 200, 300, 500 |
| max_depth | None, 5, 10, 15, 20 |

## 5. 组合搜索候选变量

候选变量池（8个变量，互斥约束）：

| 变量 | 类型 | 约束 |
|------|------|------|
| feed_ffa_pct | 原始值 | 与 log_feed_ffa_pct 互斥 |
| log_feed_ffa_pct | 对数变换 | 与 feed_ffa_pct 互斥 |
| feed_p_ppm | 原始值 | 与 log_feed_p_ppm 互斥 |
| log_feed_p_ppm | 对数变换 | 与 feed_p_ppm 互斥 |
| feed_dobi | 原始值 | 独立 |
| feed_iv | 原始值 | 独立 |
| feed_car_pv | 原始值 | 独立 |
| feed_mi_pct | 原始值 | 独立 |

有效组合总数：3 × 3 × 2⁴ - 1 = **143种**

## 6. 输出文件说明

### rf_45var/

- `rf_results.json`：模型指标（R²、RMSE、MAE、最佳超参数、过拟合诊断）
- `rf_feature_importance.csv`：特征重要性排名
- `rf_feature_importance.png`：特征重要性柱状图
- `rf_gridsearch_cv_results.csv`：GridSearchCV全部540组合的详细结果
- `rf_actual_vs_predicted_*.png`：实际值 vs 预测值散点图（训练集/测试集）
- `rf_residuals_*.png`：残差图（训练集/测试集）

### rf_5var/

同上，文件名前缀为 `rf_5var_`。

### rf_combo_search/

- `rf_combo_search_results.json`：143种组合的完整结果（含排名、最佳参数、CV RMSE）
- `rf_combo_search_ranking.csv`：按CV RMSE排名的汇总表
- `rf_combo_search_ranking.png`：Top 20组合排名可视化

## 7. 环境依赖

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```
