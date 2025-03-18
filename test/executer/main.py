import jax.numpy as jnp
from jax import random
import time

from plugins import ClfTrait, timing
from plugins.executer import Executer, NonValidExecuter, KFlodCrossExecuter, LeaveOneCrossExecuter, BootstrapExecuter


class SimpleClassifier(ClfTrait):
    def __init__(self):
        super().__init__()
        self.coef_ = None

    @timing
    def fit(self, X_train, y_train, load=False):
        # 模拟训练过程
        self.coef_ = jnp.zeros(X_train.shape[1])
        time.sleep(X_train.shape[0] / 200)  # 模拟训练时间

    @timing
    def predict_proba(self, x_test):
        # 模拟预测概率
        return jnp.ones((x_test.shape[0], 2)) * 0.5  # 返回均匀分布的概率

    def get_params(self):
        return {'model': 'SimpleClassifier'}


# 生成一些随机数据
key = random.PRNGKey(0)
X_train = random.normal(key, (100, 10))
y_train = random.randint(key, (100,), 0, 2)  # 确保标签从 0 开始
X_test = random.normal(key, (20, 10))
y_test = random.randint(key, (20,), 0, 2)    # 确保标签从 0 开始

# 创建一个简单的分类器字典
clf_dict = {
    'clf1': SimpleClassifier(),
    'clf2': SimpleClassifier(),
}

open_list = [
    True,
    True,
    True,
    True,
    True,
]


# 测试 Executer 类
if open_list[0]:
    def test_executer():
        executer = Executer(
            X_train, y_train, X_test, y_test,
            clf_dict=clf_dict,
            metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
            log=True,
            log_dir='./test_log/'
        )
        executer.run_all()
        print("Executer test passed.")


# 测试 NonValidExecuter 类
if open_list[1]:
    def test_non_valid_executer():
        executer = NonValidExecuter(
            X_train, y_train, X_test, y_test,
            clf_dict=clf_dict,
            metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
            log=True,
            log_dir='./test_log/'
        )
        executer.run_all()
        print("NonValidExecuter test passed.")


# 测试 KFlodCrossExecuter 类
if open_list[2]:
    def test_kfold_cross_executer():
        executer = KFlodCrossExecuter(
            X_train, y_train, X_test, y_test,
            clf_dict=clf_dict,
            metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
            k=5,
            log=True,
            log_dir='./test_log/'
        )
        executer.run_all()
        print("KFlodCrossExecuter test passed.")


# 测试 LeaveOneCrossExecuter 类
if open_list[3]:
    def test_leave_one_cross_executer():
        executer = LeaveOneCrossExecuter(
            X_train, y_train, X_test, y_test,
            clf_dict=clf_dict,
            metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
            log=True,
            n_class=2,
            log_dir='./test_log/'
        )
        executer.run_all()
        print("LeaveOneCrossExecuter test passed.")


# 测试 BootstrapExecuter 类
if open_list[4]:
    def test_bootstrap_executer():
        executer = BootstrapExecuter(
            X_train, y_train, X_test, y_test,
            clf_dict=clf_dict,
            metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
            n_bootstraps=10,
            log=True,
            log_dir='./test_log/'
        )
        executer.run_all()
        print("BootstrapExecuter test passed.")
