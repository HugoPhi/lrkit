## 2. 分类器接口测试 (`test_clfs.py`)

### 测试目的
验证 `Clfs` 抽象类及其子类的以下功能：
- 计时装饰器 (`@timing`) 的正确应用
- 训练/预测时间的精确记录（误差 < 50ms）
- 参数自动捕获功能 (`self.params`)
- 方法隔离性（不同方法不互相干扰计时）

### 安装依赖
```bash
pip install pytest jax
```

### 运行测试
```bash
pytest test_clfs.py -v
```

### 测试用例说明
| 测试用例                  | 验证场景                                                                 |
|--------------------------|--------------------------------------------------------------------------|
| `test_timing_decorator`  | 检查装饰器是否正确应用到 `fit/predict_proba` 方法                        |
| `test_params_recording`  | 验证 `__init__` 参数是否被正确记录到 `self.params`                       |
| `test_predict_chain`      | 测试 `predict` 方法是否触发 `predict_proba` 的计时逻辑                   |
| `test_time_initialization` | 检查 `training_time/testing_time` 初始值是否为 -1                       |

### 关键约束
1. 测试类必须实现 `predict_proba` 并返回 JAX 数组
2. `fit` 方法需要包含 `load=False` 参数
3. 自定义参数需要通过 `__init__` 显式声明

---

### 文档使用建议
1. 将两个 README 分别保存为 `TEST_METRICS_README.md` 和 `TEST_CLFS_README.md`
2. 与对应测试文件放在同一目录
3. 可通过以下命令快速查看测试说明：
   ```bash
   cat TEST_*_README.md | less
   ```
