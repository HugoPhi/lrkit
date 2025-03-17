## 执行器测试框架

这是一个用于测试执行器的框架，适配 `numpy` 实现。

### 功能
- 实现了一个简单的分类器 `SimpleClassifier`，支持训练和预测。
- 提供了多种执行器 (`Executer` 及其子类)，用于评估分类器的性能。
- 支持以下执行器：
  - `Executer`: 基本的训练和测试执行器。
  - `NonValidExecuter`: 无验证集的执行器。
  - `KFlodCrossExecuter`: K 折交叉验证执行器。
  - `LeaveOneCrossExecuter`: 留一法交叉验证执行器。
  - `BootstrapExecuter`: Bootstrap 采样执行器。

### 测试用例
- `test_executer`: 测试基本的训练和测试执行器。
- `test_non_valid_executer`: 测试无验证集的执行器。
- `test_kfold_cross_executer`: 测试 K 折交叉验证执行器。
- `test_leave_one_cross_executer`: 测试留一法交叉验证执行器。
- `test_bootstrap_executer`: 测试 Bootstrap 采样执行器。

### 关键约束
1. 分类器必须实现 `fit` 和 `predict_proba` 方法。
2. `fit` 方法需要包含 `load=False` 参数。
3. 自定义参数需要通过 `__init__` 显式声明。
