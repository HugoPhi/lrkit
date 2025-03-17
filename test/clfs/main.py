import pytest
import time
import numpy as np  # 替换 jax.numpy 为 numpy

from plugins import ClfTrait, timing


class TestClfs:
    """测试 Clfs 派生类及其计时装饰器"""

    @pytest.fixture
    def clf_instance(self):
        """创建测试用的分类器实例"""
        class TestClassifier(ClfTrait):
            # 必须显式定义 __init__ 以兼容 Clfs 的参数记录逻辑
            def __init__(self, param1=1, param2="test"):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

            @timing
            def fit(self, X, y, load=False):
                time.sleep(0.1)

            @timing
            def predict_proba(self, X):
                time.sleep(0.1)
                return np.array([[0.5, 0.5]])  # 使用 numpy 替换 jax.numpy

        return TestClassifier(param1=10, param2="custom")

    # 测试装饰器功能
    def test_timing_decorator_application(self, clf_instance):
        """验证装饰器正确应用且区分方法类型"""
        # 检查装饰器是否应用
        assert hasattr(clf_instance.fit, "__wrapped__"), "fit 方法未装饰"
        assert hasattr(clf_instance.predict_proba, "__wrapped__"), "predict_proba 未装饰"

        # 验证 predict 方法是否自动继承装饰器
        predict_method = clf_instance.predict
        assert not hasattr(predict_method, "__wrapped__"), "predict 方法不应直接装饰"
        assert predict_method.__name__ == "predict", "predict 方法名称异常"

    # 测试时间记录初始化
    def test_time_initialization(self, clf_instance):
        """验证时间戳初始值为 -1"""
        assert clf_instance.training_time == -1, "训练时间初始值错误"
        assert clf_instance.testing_time == -1, "测试时间初始值错误"

    # 测试 fit 方法计时
    def test_fit_timing(self, clf_instance):
        """验证 fit 方法更新 training_time"""
        clf_instance.fit(None, None)
        assert clf_instance.training_time >= 0.09, "训练时间未正确记录"
        assert clf_instance.testing_time == -1, "fit 方法错误影响 testing_time"

    # 测试 predict 链式计时
    def test_predict_chain(self, clf_instance):
        """验证 predict 调用触发 predict_proba 的计时"""
        clf_instance.predict(None)
        assert 0.09 <= clf_instance.testing_time <= 0.15, "预测时间记录异常"

    # 测试参数记录
    def test_params_recording(self, clf_instance):
        """验证 __new__ 方法的参数记录功能"""
        expected_params = {
            "param1": 10,
            "param2": "custom"
        }
        assert clf_instance.params == expected_params, "参数记录不完整"
