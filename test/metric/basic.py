import pytest
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from plugins.metric import Metrics  # 替换为您的模块路径


# --------------------------
# 测试工具函数
# --------------------------
def assert_metrics_match_sklearn(y_true, y_pred, average=None, classes=None):
    """对比自定义指标与 sklearn 指标"""
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # 初始化自定义指标
    metrics = Metrics(jnp.array(y_true), jnp.array(y_pred), classes=classes)

    # 对比基础指标
    np.testing.assert_allclose(
        metrics.precision(),
        precision_score(y_true_np, y_pred_np, average=None, zero_division=0.0),
        rtol=1e-5
    )

    np.testing.assert_allclose(
        metrics.recall(),
        recall_score(y_true_np, y_pred_np, average=None, zero_division=0.0),
        rtol=1e-5
    )

    np.testing.assert_allclose(
        metrics.f1(),
        f1_score(y_true_np, y_pred_np, average=None, zero_division=0.0),
        rtol=1e-5
    )

    assert np.isclose(
        metrics.accuracy(),
        accuracy_score(y_true_np, y_pred_np),
        rtol=1e-5
    )

    # 对比混淆矩阵
    sklearn_matrix = confusion_matrix(y_true_np, y_pred_np)
    np.testing.assert_array_equal(metrics.matrix, sklearn_matrix)

# --------------------------
# 测试用例
# --------------------------


# 测试用例 1: 二分类硬标签
def test_binary_hard_labels():
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 1]
    assert_metrics_match_sklearn(y_true, y_pred)


# 测试用例 2: 多分类硬标签
def test_multiclass_hard_labels():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    assert_metrics_match_sklearn(y_true, y_pred)


# 测试用例 3: 二分类概率预测
def test_binary_proba():
    # 生成模拟概率（使用 sigmoid 函数）
    y_true = jnp.array([0, 1, 0, 1])
    y_pred_proba = jnp.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])

    # 初始化指标
    metrics = Metrics(y_true, y_pred_proba)

    # 对比 AUC
    sklearn_auc = roc_auc_score(y_true, y_pred_proba[:, 1], multi_class='ovr')
    assert np.isclose(metrics.avg_auc(), sklearn_auc, rtol=1e-5)

    # 对比 AP
    sklearn_ap = average_precision_score(y_true, y_pred_proba[:, 1])
    assert np.isclose(metrics.avg_ap(), sklearn_ap, rtol=1e-5)


# 测试用例 4: 多分类概率预测
def test_multiclass_proba():
    y_true = jnp.array([0, 1, 2, 0])
    y_pred_proba = jnp.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
        [0.6, 0.3, 0.1]
    ])

    metrics = Metrics(y_true, y_pred_proba)

    # 对比每个类别的 AUC
    for i in range(3):
        y_true_binary = (y_true == i).astype(int)
        sklearn_auc = roc_auc_score(y_true_binary, y_pred_proba[:, i])
        assert np.isclose(metrics.auc()[i], sklearn_auc, rtol=1e-5)


# 测试用例 5: 全正确预测
def test_perfect_predictions():
    y_true = [0, 1, 0, 1, 2, 2]
    y_pred = y_true.copy()
    assert_metrics_match_sklearn(y_true, y_pred)


# 测试用例 6: 全错误预测
def test_all_wrong():
    y_true = [0, 1, 0, 1]
    y_pred = [1, 0, 1, 0]
    assert_metrics_match_sklearn(y_true, y_pred)


# 测试用例 7: 输入形状测试
def test_input_shapes():
    # 测试 (N, 1) 形状输入
    y_true = jnp.array([[0], [1], [0], [1]])
    y_pred = jnp.array([[0], [1], [1], [1]])
    assert_metrics_match_sklearn(y_true.flatten(), y_pred.flatten())


# 测试用例 8: 边缘情况测试
def test_edge_cases():
    # 测试全零预测
    y_true = [0, 0, 0, 0, 0]
    y_pred = [0, 0, 0, 0, 0]
    assert_metrics_match_sklearn(y_true, y_pred)


# 测试用例 9: 异常输入处理
def test_invalid_inputs():
    # 类别数量不一致
    with pytest.raises(AssertionError):
        y_true = jnp.array([0, 1, 2])
        y_pred = jnp.array([0, 1, 1])
        Metrics(y_true, y_pred, classes=2)

    # 无效的概率输入
    with pytest.raises(ValueError):
        y_true = jnp.array([0, 1])
        y_pred = jnp.array([0, 1])  # 硬标签
        Metrics(y_true, y_pred).auc()


if __name__ == "__main__":
    pytest.main([__file__])
