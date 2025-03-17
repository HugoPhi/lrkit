import numpy as np  # 替换 jax.numpy 为 numpy
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
        self.coef_ = np.zeros(X_train.shape[1])
        time.sleep(X_train.shape[0] / 200)  # 模拟训练时间

    @timing
    def predict_proba(self, x_test):
        # 模拟预测概率
        return np.ones((x_test.shape[0], 2)) * 0.5  # 返回均匀分布的概率

    def get_params(self):
        return {'model': 'SimpleClassifier'}


# 生成一些随机数据
np.random.seed(0)  # 设置随机种子
X_train = np.random.normal(size=(100, 10))
y_train = np.random.randint(0, 2, size=(100,))  # 确保标签从 0 开始
X_test = np.random.normal(size=(20, 10))
y_test = np.random.randint(0, 2, size=(20,))    # 确保标签从 0 开始

# 创建一个简单的分类器字典
clf_dict = {
    'clf1': SimpleClassifier(),
    'clf2': SimpleClassifier(),
}


# 测试 Executer 类
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
def test_leave_one_cross_executer():
    executer = LeaveOneCrossExecuter(
        X_train, y_train, X_test, y_test,
        clf_dict=clf_dict,
        metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
        log=True,
        log_dir='./test_log/'
    )
    executer.run_all()
    print("LeaveOneCrossExecuter test passed.")


# 测试 BootstrapExecuter 类
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


# 运行所有测试
if __name__ == "__main__":
    test_executer()
    test_non_valid_executer()
    test_kfold_cross_executer()
    test_leave_one_cross_executer()
    test_bootstrap_executer()
