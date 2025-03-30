import pytest
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from plugins.metric import Metrics  # 替换为您的模块路径

# 设置随机种子保证可重复性
SEED = 42
np.random.seed(SEED)
jax.config.update("jax_enable_x64", True)  # 启用高精度计算


# --------------------------
# 测试工具函数增强版
# --------------------------
class TestUtils:
    @staticmethod
    def generate_massive_data(samples=1e3, classes=10, proba=False):
        """生成大规模测试数据"""
        y_true = np.random.randint(0, classes, int(samples))

        if proba:
            # 生成符合概率分布的预测
            y_pred = np.random.rand(int(samples), classes)
            y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        else:
            y_pred = np.random.randint(0, classes, int(samples))

        return y_true, y_pred

    @staticmethod
    def assert_all_metrics(y_true, y_pred, metrics, decimal=5):
        """增强版指标对比"""
        # 基础指标对比
        np.testing.assert_array_almost_equal(
            metrics.precision(),
            precision_score(y_true, y_pred, average=None, zero_division=0),
            decimal=decimal
        )

        np.testing.assert_array_almost_equal(
            metrics.recall(),
            recall_score(y_true, y_pred, average=None, zero_division=0),
            decimal=decimal
        )

        np.testing.assert_array_almost_equal(
            metrics.f1(),
            f1_score(y_true, y_pred, average=None, zero_division=0),
            decimal=decimal
        )

        assert np.isclose(
            metrics.accuracy(),
            accuracy_score(y_true, y_pred),
            rtol=10**-decimal
        )

        # 混淆矩阵对比
        sklearn_matrix = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(metrics.matrix, sklearn_matrix)


# --------------------------
# 大规模测试用例
# --------------------------

# 测试用例 1: 10万样本二分类测试
def test_large_binary_classification():
    samples = 1e3
    y_true = np.random.randint(0, 2, int(samples))
    y_pred = np.random.randint(0, 2, int(samples))

    metrics = Metrics(jnp.array(y_true), jnp.array(y_pred))
    TestUtils.assert_all_metrics(y_true, y_pred, metrics)


# 测试用例 2: 高维多分类测试（100类）
def test_high_dimension_multiclass():
    classes = 100
    y_true = np.random.randint(0, classes, 1000)
    y_pred = np.random.randint(0, classes, 1000)

    metrics = Metrics(jnp.array(y_true), jnp.array(y_pred), classes=classes)
    TestUtils.assert_all_metrics(y_true, y_pred, metrics)


# 测试用例 3: 大规模概率预测测试
def test_large_proba_prediction():
    y_true, y_pred = TestUtils.generate_massive_data(samples=1e3, classes=5, proba=True)

    # 转换为硬标签
    y_pred_labels = y_pred.argmax(axis=1)

    # 初始化指标
    metrics = Metrics(jnp.array(y_true), jnp.array(y_pred))

    # 验证基础指标
    TestUtils.assert_all_metrics(y_true, y_pred_labels, metrics)

    # 验证概率指标
    for i in range(5):
        y_true_binary = (y_true == i).astype(int)

        # AUC对比
        sklearn_auc = roc_auc_score(y_true_binary, y_pred[:, i])
        assert np.isclose(metrics.auc()[i], sklearn_auc, rtol=1e-5)

        # AP对比
        sklearn_ap = average_precision_score(y_true_binary, y_pred[:, i])
        assert np.isclose(metrics.ap()[i], sklearn_ap, rtol=1e-5)


# 测试用例 4: 类别极度不均衡测试
def test_extreme_class_imbalance():
    # 99:1 不平衡数据
    y_true = jnp.array([0] * 99 + [1] * 1)
    y_pred = jnp.array([0] * 98 + [1] * 2)  # 预测2个正例（1个正确）

    metrics = Metrics(y_true, y_pred)

    # 验证混淆矩阵
    assert metrics.matrix[1, 1] == 1  # TP
    assert metrics.matrix[0, 1] == 1  # FP

    # 验证指标
    assert metrics.precision()[1] == 0.5  # 1/(1+1)
    assert metrics.recall()[1] == 1.0    # 1/1


# 测试用例 5: 数值稳定性测试
def test_numerical_stability():
    # 生成极端概率值
    y_true = jnp.array([0, 1, 0, 1])
    y_pred = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    metrics = Metrics(y_true, y_pred)

    # 验证AUC计算
    assert np.allclose(metrics.auc(), [1.0, 1.0])

    # 验证AP计算
    assert np.allclose(metrics.ap(), [1.0, 1.0])


# 测试用例 6: 随机压力测试
@pytest.mark.parametrize("seed", [42, 123, 789, 555])
def test_random_stress(seed):
    np.random.seed(seed)
    for _ in range(10):  # 每次运行10个随机测试
        classes = np.random.randint(2, 10)
        samples = np.random.randint(1000, 5000)

        y_true = np.random.randint(0, classes, samples)
        y_pred = np.random.randint(0, classes, samples)

        metrics = Metrics(jnp.array(y_true), jnp.array(y_pred))
        TestUtils.assert_all_metrics(y_true, y_pred, metrics)


# 测试用例 7: 边界值测试
def test_boundary_values():
    # 全零概率
    y_true = jnp.array([0, 1, 0, 1])
    y_pred = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    metrics = Metrics(y_true, y_pred)
    assert metrics.auc()[1] == 0.0  # 负例AUC

    # 全一概率
    y_pred = jnp.array([[0.0, 1.0]] * 4)
    metrics = Metrics(y_true, y_pred)
    assert metrics.auc()[0] == 0.0  # 正例AUC


# 测试用例 8: 并行计算测试
# def test_jit_compatibility():
#     # 测试JAX的JIT编译兼容性
#     @jax.jit
#     def jitted_metrics(y_true, y_pred):
#         metrics = Metrics(y_true, y_pred)
#         return (
#             metrics.precision(),
#             metrics.recall(),
#             metrics.accuracy()
#         )
#
#     y_true = jnp.array([0, 1, 0, 1])
#     y_pred = jnp.array([0, 1, 0, 1])
#     prec, rec, acc = jitted_metrics(y_true, y_pred)
#
#     np.testing.assert_allclose(prec, [1., 1.])
#     np.testing.assert_allclose(rec, [1., 1.])
#     assert acc == 1.0


# 测试用例 9: 内存压力测试
# @pytest.mark.large_memory
# def test_memory_pressure():
#     """需要至少16GB内存运行的测试"""
#     y_true, y_pred = TestUtils.generate_massive_data(samples=1e7, classes=100)
#
#     # 测试初始化不崩溃
#     metrics = Metrics(jnp.array(y_true), jnp.array(y_pred))
#
#     # 采样验证部分指标
#     sample_idx = np.random.choice(len(y_true), 1000)
#     partial_true = y_true[sample_idx]
#     partial_pred = y_pred[sample_idx]
#
#     # 验证子集指标一致性
#     partial_metrics = Metrics(
#         jnp.array(partial_true),
#         jnp.array(partial_pred)
#     )
#     TestUtils.assert_all_metrics(partial_true, partial_pred, partial_metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--durations=10"])
