import pytest
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from plugins.metric import Metrics

# 固定随机种子保证可重复性
jax.config.update("jax_enable_x64", True)
np.random.seed(42)
jax.random.PRNGKey(42)


def generate_random_data(n_samples=1000, n_classes=3):
    """生成随机测试数据（概率输入）"""
    # 生成真实标签（确保至少包含所有类别）
    y_true = np.random.randint(0, n_classes, n_samples)

    # 生成随机 logits 并通过 softmax 转换为概率
    logits = jax.random.normal(jax.random.PRNGKey(0), (n_samples, n_classes))
    y_pred = jax.nn.softmax(logits, axis=1)

    return y_true, y_pred


def test_compare_with_sklearn():
    """与 scikit-learn 的标准实现对比"""
    n_classes = np.random.randint(2, 5)  # 随机生成 2~4 个类别
    y_true, y_pred = generate_random_data(n_samples=1000, n_classes=n_classes)

    # 计算预测标签（argmax）
    y_pred_labels = jnp.argmax(y_pred, axis=1)

    # 初始化 Metrics 对象
    metrics = Metrics(y_true, y_pred)

    # ----------------- 验证基础指标 -----------------
    # 准确率
    sk_accuracy = accuracy_score(y_true, y_pred_labels)
    assert jnp.allclose(metrics.accuracy(), sk_accuracy, atol=1e-4)

    # 精确率（按类）
    sk_precision = precision_score(y_true, y_pred_labels, average=None, zero_division=0)
    assert jnp.allclose(metrics.precision(), sk_precision, atol=1e-4)

    # 召回率（按类）
    sk_recall = recall_score(y_true, y_pred_labels, average=None, zero_division=0)
    assert jnp.allclose(metrics.recall(), sk_recall, atol=1e-4)

    # F1（按类）
    sk_f1 = f1_score(y_true, y_pred_labels, average=None, zero_division=0)
    assert jnp.allclose(metrics.f1(), sk_f1, atol=1e-4)

    # ----------------- 验证 AUC/AP（概率相关指标） -----------------
    # AUC（One-vs-Rest）
    # sk_auc = roc_auc_score(
    #     y_true,
    #     y_pred,
    #     multi_class="ovr",
    #     average=None,
    #     labels=np.arange(n_classes),
    # )
    # assert jnp.allclose(metrics.auc(), sk_auc, atol=0.05)  # 允许较大误差
    #
    # # AP（按类）
    # sk_ap = []
    # for cls in range(n_classes):
    #     y_true_bin = (y_true == cls).astype(int)
    #     sk_ap.append(average_precision_score(y_true_bin, y_pred[:, cls]))
    # sk_ap = np.array(sk_ap)
    # assert jnp.allclose(metrics.ap(), sk_ap, atol=0.05)
    # 处理 scikit-learn 对极端情况的限制（如全正/负样本）
    # pytest.skip(f"Scikit-learn 限制: {str(e)}")


def test_edge_cases():
    """极端情况测试（如全正确/全错误预测）"""
    # 全正确预测
    y_true = jnp.array([0, 1, 2, 0, 1])
    y_pred = jnp.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.1, 0.9],
        [0.8, 0.2, 0.0],
        [0.2, 0.7, 0.1],
    ])
    metrics = Metrics(y_true, y_pred)
    assert jnp.allclose(metrics.accuracy(), 1.0)

    # 全错误预测
    y_pred_wrong = jnp.roll(y_pred, shift=1, axis=1)  # 错位预测
    metrics_wrong = Metrics(y_true, y_pred_wrong)
    assert jnp.allclose(metrics_wrong.accuracy(), 0.0)
