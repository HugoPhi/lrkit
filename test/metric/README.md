## 分类模型评价指标测试 (`test_metrics.py`)

### 测试目的
验证自定义 `Metrics` 类的各项指标计算准确性，包括：
- 混淆矩阵、准确率、精确率、召回率、F1 值
- AUC 和 AP（平均精度）等概率相关指标
- 与 scikit-learn 官方计算结果的一致性

### 安装依赖
```bash
pip install pytest jax scikit-learn
```

### 运行测试
```bash
pytest test_metrics.py -v
```

### 测试用例说明
| 测试用例              | 验证场景                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| `test_confusion_matrix` | 混淆矩阵生成逻辑                                                         |
| `test_precision`        | 按类别精确率计算                                                         |
| `test_compare_with_sklearn` | 与 scikit-learn 的准确率/AUC/AP 等指标对比（允许 ±5% 误差）             |
| `test_edge_cases`       | 全正确/全错误预测等极端场景                                              |

### 注意事项
1. 需要 JAX 的 float64 支持，测试前请运行：
   ```python
   jax.config.update("jax_enable_x64", True)
   ```
2. 二分类测试数据需要显式构造两列概率矩阵
3. 当出现全正样本/全负样本时可能跳过部分测试
