import atexit
from datetime import datetime
import toml
import os
import traceback
from tabulate import tabulate
import pandas as pd
import jax.numpy as jnp
from jax import random

from .metric import Metrics


def combine_mean_std(df, precision=4):
    """
    by DeepSeek.
    合并 _mean 和 _std 列，格式为 mean ± std，并限制浮点数的精度。

    参数:
        df (pd.DataFrame): 包含 _mean 和 _std 列的 DataFrame。
        precision (int): 浮点数的精度（小数位数），默认为 4。

    返回:
        pd.DataFrame: 合并后的 DataFrame。
    """
    # 获取所有包含 _mean 和 _std 的列
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    std_cols = [col for col in df.columns if col.endswith('_std')]

    # 创建一个新的 DataFrame 来存储合并后的结果
    combined_df = pd.DataFrame()

    # 合并 mean 和 std 列
    for mean_col, std_col in zip(mean_cols, std_cols):
        # 提取指标名称（去掉 _mean 和 _std）
        metric_name = mean_col.replace('_mean', '')

        # 合并 mean 和 std 列，并限制浮点数精度
        combined_df[metric_name] = (
            df[mean_col].round(precision).astype(str) + " ± " + df[std_col].round(precision).astype(str)
        )

    # 添加 model 列
    combined_df['model'] = df['model']

    # 重新排列列顺序，确保 model 在第一列
    combined_df = combined_df[['model'] + [col for col in combined_df.columns if col != 'model']]

    return combined_df


class Executer:
    '''
    执行器基类，只进行训练和测试。
    ==========
      - 快捷管理训练，测试，日志全过程，并灵活调试Classifier数组里面的各个模型。
      - 开启Log，支持中途运行出错，结果不丢失。
      - 在使用的时候根据需要重写execute(self)方法。

    Parameters
    ----------
    X_train : jnp.ndarray
        训练集的X。
    y_train : jnp.ndarray
        训练集的y。
    X_test : jnp.ndarray
        测试集的X。
    y_test : jnp.ndarray
        测试集的y。
    clf_dict : dict
        Clf字典。包含多个实验的{name : Clf}
    metric_list : list
        测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
    log : bool
        是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将测试结果写入到同一文件夹的test.csv。
    log_dir : str
        存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
    '''

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 log_dir='./log/'):
        '''
        初始化。

        Parameters
        ----------
        X_train : jnp.ndarray
            训练集的X。
        y_train : jnp.ndarray
            训练集的y。
        X_test : jnp.ndarray
            测试集的X。
        y_test : jnp.ndarray
            测试集的y。
        clf_dict : dict
            Clf字典。包含多个实验的{name : Clf}
        metric_list : list
            测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
        log : bool
            是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将结果写入到同一文件夹的result.csv。
        log_dir : str
            存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
        '''

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf_dict = clf_dict
        self.metric_list = metric_list
        self.log = log

        self.df = pd.DataFrame(columns=['model'] + self.metric_list + ['training time'] + ['testing time'])

        # log
        if log:
            self.log_dir = log_dir
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.log_path = os.path.join(self.log_dir, f'{datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}/')
            os.mkdir(self.log_path)

            hyper_config = dict()
            for name, clf in clf_dict.items():
                hyper_config[name] = clf.get_params()

            toml.dump(hyper_config, open(os.path.join(self.log_path, 'hyper.toml'), 'w'))  # 保存超参数和模型参数

            atexit.register(self.save_df)  # 保证退出的时候能保存已经生成的df

    def save_df(self):
        '''
        保存df到日志
        '''
        self.df.to_csv(os.path.join(self.log_path, 'result.csv'), index=False)

    def execute(self, name, clf):
        '''
        执行实验。

        Notes
        -----
          - 这里必须返回一个测试器和一个训练好的分类器，因为写入日志要用。

        Parameters
        ----------
        name : str
            实验的名字。
        clf : Clfs
            实验获取的分类器，继承自接口Clfs。

        Returns
        -------
        clf : Clfs
            训练好的分类器。
        metric: Metrics
            有记录的Metric实例。

        Examples
        --------

        可以这么重写：
        ```python
        class MyExecuter(Executer):
            def execute(self, name, clf):
                print(f'>> {name}')

                clf.fit(self.X_train, self.y_train)
                print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

                y_pred = clf.predict(self.X_test)

                mtc = Metrics(self.y_test, y_pred)

                return mtc, clf
        ```
        '''
        print(f'>> {name}')

        clf.fit(self.X_train, self.y_train)  # 训练分类器
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)

        mtc = Metrics(self.y_test, y_pred)  # 构建测试器
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')

        time = [clf.get_training_time(), clf.get_testing_time()]

        return mtc, clf, time  # 返回测试器和分类器

    def logline(self, name, mtc, clf, time):
        '''
        将某次实验的结果写入日志df。
        '''

        func_list = []
        for metric in self.metric_list:
            func = getattr(mtc, metric, None)
            if callable(func):
                func_list.append(func)
            else:
                raise ValueError(f'{metric} is not in Metric.')

        self.df.loc[len(self.df)] = [name] + [func() for func in func_list] + time

    def run(self, key):
        '''
        运行单个实验。不会消耗clf_dict。同时会写入日志。

        Parameters
        ----------
        key : str
            实验的名字。
        '''
        if key in self.clf_dict.keys():
            mtc, clf, time = self.execute(key, self.clf_dict[key])

            self.logline(key, mtc, clf, time)
        else:
            raise KeyError(f'{key} is not in clf_dict')

    def step(self):
        '''
        迭代运行实验。采用迭代器模式。会逐个消耗实验，直到clf_dict为空。过程中会返回对应的名字和Clf对象，如果是最后一个，返回None。同时会写入日志。

        Returns
        -------
        name : str
            实验的名字。
        clf : Clfs
            实验获取的分类器，继承自接口Clfs。
        '''
        if len(self.clf_dict) == 0:
            return None

        try:
            name, clf = self.clf_dict.popitem()

            mtc, clf, time = self.execute(name, clf)

            self.logline(name, mtc, clf, time)

            return name, clf
        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

    def format_print(self, sort_by='accuracy', ascending=False, precision=4, time=False):
        '''
        表格的格式化输出。

        Parameters
        ----------
        sort_by : str
            按照哪个指标进行排序。比如：'accuracy'，表示测试集按照accuracy指标进行排序。
        ascending : bool
            是否升序。
        precision : int
            保留几位小数。
        time: bool
            是否显示训练和测试时间。
        '''

        if sort_by is not None:
            print(f'\n>> Test Result, sort by \'{sort_by}\'.')
            if not time:
                temp_table = self.df.sort_values(sort_by, ascending=ascending).drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.df.sort_values(sort_by, ascending=ascending)

            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

        else:
            print('\n>> Test Result.')
            if not time:
                temp_table = self.df.drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.df
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

    def run_all(self, sort_by=None, ascending=False, precision=4, time=False):
        '''
        运行所有实验。

        Parameters
        ----------
        sort_by : str
            按照哪个指标进行排序。比如：'accuracy'，表示测试集按照accuracy指标进行排序。
        ascending : bool
            是否升序。
        precision : int
            保留几位小数。
        time: bool
            是否显示训练和测试时间。
        '''

        for name, clf in self.clf_dict.items():
            mtc, clf, time = self.execute(name, clf)

            self.logline(name, mtc, clf, time)

        self.format_print(sort_by, ascending, precision, time)

    def get_result(self):
        '''
        返回实验结果对应的表格。

        Returns
        -------
        self.pd : pd.DataFrame
        '''
        return self.df


class NonValidExecuter(Executer):
    '''
    不进行Validation的执行器，只进行训练和测试。
    ==========
      - 快捷管理训练，测试，日志全过程，并灵活调试Classifier数组里面的各个模型。
      - 开启Log，支持中途运行出错，结果不丢失。
      - 在使用的时候根据需要重写execute(self)方法。

    Parameters
    ----------
    X_train : jnp.ndarray
        训练集的X。
    y_train : jnp.ndarray
        训练集的y。
    X_test : jnp.ndarray
        测试集的X。
    y_test : jnp.ndarray
        测试集的y。
    clf_dict : dict
        Clf字典。包含多个实验的{name : Clf}
    metric_list : list
        测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
    log : bool
        是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将测试结果写入到同一文件夹的test.csv。
    log_dir : str
        存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
    '''

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 log_dir='./log/'):

        super(NonValidExecuter, self).__init__(X_train, y_train, X_test, y_test,
                                               clf_dict=clf_dict, metric_list=metric_list, log=log, log_dir=log_dir)


class KFlodCrossExecuter(Executer):
    '''
    使用K折交叉验证作为Validation的执行器。
    ==========
      - 快捷管理训练，测试，日志全过程，并灵活调试Classifier数组里面的各个模型。
      - 开启Log，支持中途运行出错，结果不丢失。
      - 在使用的时候根据需要重写execute(self)方法。

    Parameters
    ----------
    X_train : jnp.ndarray
        训练集的X。
    y_train : jnp.ndarray
        训练集的y。
    X_test : jnp.ndarray
        测试集的X。
    y_test : jnp.ndarray
        测试集的y。
    clf_dict : dict
        Clf字典。包含多个实验的{name : Clf}
    metric_list : list
        测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
    k : int
        K折验证的k的大小，k >= 1 。
    log : bool
        是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将测试结果写入到同一文件夹的test.csv，将Validation结果写入到同一文件夹的valid.csv。
    log_dir : str
        存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
    '''

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 k=10,
                 log=False,
                 log_dir='./log/'):

        super(KFlodCrossExecuter, self).__init__(X_train, y_train, X_test, y_test,
                                                 clf_dict, metric_list, log, log_dir)

        self.k = k
        if k < 1:
            raise ValueError(f'k should >= 1, but get {self.k}')

        metrics = self.metric_list + ['training time', 'testing time']

        self.test = pd.DataFrame(columns=['model'] + metrics)
        self.valid = pd.DataFrame(columns=['model'] + [f'{x}_{suffix}' for x in metrics for suffix in ['mean', 'std']])

    def execute(self, name, clf):
        '''
        执行实验，不记录日志。

        Notes
        -----
          - 这里必须返回一个测试器和一个训练好的分类器，因为写入日志要用。

        Parameters
        ----------
        name : str
            实验的名字。
        clf : Clfs
            实验获取的分类器，继承自接口Clfs。

        Returns
        -------
        clf : Clfs
            训练好的分类器。
        metric: Metrics
            有记录的Metric实例。

        Examples
        --------

        可以这么重写：
        ```python
        class MyExcuter(Excuter):
            def execute(self, name, clf):
                print(f'>> {name}')

                clf.fit(self.X_train, self.y_train)
                print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

                y_pred = clf.predict(self.X_test)

                mtc = Metrics(self.y_test, y_pred)

                return mtc, clf
        ```
        '''
        print(f'>> {name}')

        # k折交叉验证
        k_fold_x_train = jnp.array_split(self.X_train, self.k)
        k_fold_y_train = jnp.array_split(self.y_train, self.k)
        mtcs = []
        times = []  # validation training, teting time + test training, testing time
        for i in range(self.k):
            print(f'>>>> Validate: {i + 1}')
            x_train = jnp.concatenate(k_fold_x_train[:i] + k_fold_x_train[i + 1:])
            y_train = jnp.concatenate(k_fold_y_train[:i] + k_fold_y_train[i + 1:])
            x_test = k_fold_x_train[i]
            y_test = k_fold_y_train[i]
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            mtc = Metrics(y_test, y_pred)
            times.append([clf.get_training_time(), clf.get_testing_time()])
            mtcs.append(mtc)

        # real train & test
        print('>>>> Test:')
        clf.fit(self.X_train, self.y_train)  # 训练分类器
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)

        mtc = Metrics(self.y_test, y_pred)  # 构建测试器
        mtcs.append(mtc)
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')
        times.append([clf.get_training_time(), clf.get_testing_time()])

        return mtcs, clf, times  # 返回所有测试器和分类器和验证时间

    def logline(self, name, mtcs: list, clf, times):
        '''
        将某次实验的结果写入日志df。
        '''

        test_mtc = mtcs.pop()
        test_times = times.pop()

        def getline(mtc):
            func_list = []
            for metric in self.metric_list:
                func = getattr(mtc, metric, None)
                if callable(func):
                    func_list.append(func)
                else:
                    raise ValueError(f'{metric} is not in Metric.')

            return [func() for func in func_list]

        self.test.loc[len(self.test)] = [name] + getline(test_mtc) + test_times  # 获取测试的结果

        valid_rows = [getline(mtc) + times[ix] for ix, mtc in enumerate(mtcs)]
        valids_array = jnp.array(valid_rows)

        mean_vals = jnp.mean(valids_array, axis=0).tolist()
        std_vals = jnp.std(valids_array, axis=0).tolist()

        valid_result = []
        for mean, std in zip(mean_vals, std_vals):
            valid_result.append(mean)
            valid_result.append(std)

        self.valid.loc[len(self.valid)] = [name] + valid_result

    def save_df(self):
        '''
        保存df到日志
        '''
        self.test.to_csv(os.path.join(self.log_path, 'test.csv'), index=False)
        self.valid.to_csv(os.path.join(self.log_path, 'valid.csv'), index=False)

    def format_print(self, sort_by=('accuracy', 'accuracy_mean'), ascending=False, precision=4, time=False):
        '''
        表格的格式化输出。

        Parameters
        ----------
        sort_by : str
            按照哪个指标进行排序。接受两个位置，分别是测试和验证的指标。比如：('accuracy', 'accuracy_mean')，表示测试集按照accuracy指标进行排序，验证集按照accuracy_mean指标进行排序。
        ascending : bool
            是否升序。
        '''

        if sort_by is not None:
            print(f'\n>> Test Result, sort by \'{sort_by[0]}\'.')
            if not time:
                temp_table = self.test.sort_values(sort_by[0], ascending=ascending).drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.test.sort_values(sort_by[0], ascending=ascending)
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

            print(f'\n>> Validation Result(Mean ± Std), sort by \'{sort_by[1]}\'.')
            if not time:
                temp_table = self.valid.sort_values(sort_by[1], ascending=ascending)
                temp_table = combine_mean_std(temp_table, precision=precision)
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.valid.sort_values(sort_by[1], ascending=ascending)
                temp_table = combine_mean_std(temp_table, precision=precision)
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                showindex=False
            ))
        else:
            print('\n>> Test Result.')
            if not time:
                temp_table = self.test.drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.test
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

            print('\n>> Validation Result(Mean ± Std).')
            if not time:
                temp_table = self.valid
                temp_table = combine_mean_std(temp_table, precision=precision)
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.valid
                temp_table = combine_mean_std(temp_table, precision=precision)
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                showindex=False
            ))

    def run_all(self, sort_by=['accuracy', 'accuracy_mean'], ascending=False, precision=4, time=False):
        '''
        运行所有实验。

        Parameters
        ----------
        sort_by : str
            按照哪个指标进行排序。接受两个位置，分别是测试和验证的指标。比如：('accuracy', 'accuracy_mean')，表示测试集按照accuracy指标进行排序，验证集按照accuracy_mean指标进行排序。
        ascending : bool
            是否升序。
        precision : int
            保留几位小数。
        time: bool
            是否显示训练和测试时间。
        '''

        for name, clf in self.clf_dict.items():
            mtc, clf, times = self.execute(name, clf)

            self.logline(name, mtc, clf, times)

        self.format_print(sort_by, ascending, precision, time)


class LeaveOneCrossExecuter(KFlodCrossExecuter):
    '''
    使用留一法交叉验证作为Validation的执行器。
    ==========
      - 快捷管理训练，测试，日志全过程，并灵活调试Classifier数组里面的各个模型。
      - 开启Log，支持中途运行出错，结果不丢失。
      - 在使用的时候根据需要重写execute(self)方法。

    Parameters
    ----------
    X_train : jnp.ndarray
        训练集的X。
    y_train : jnp.ndarray
        训练集的y。
    X_test : jnp.ndarray
        测试集的X。
    y_test : jnp.ndarray
        测试集的y。
    clf_dict : dict
        Clf字典。包含多个实验的{name : Clf}
    metric_list : list
        测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
    log : bool
        是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将测试结果写入到同一文件夹的test.csv，将Validation结果写入到同一文件夹的valid.csv。
    log_dir : str
        存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
    n_class: int
        分类任务的类别数。
    '''

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 n_class=None,
                 log_dir='./log/'):

        super(LeaveOneCrossExecuter, self).__init__(X_train, y_train, X_test, y_test,
                                                    clf_dict=clf_dict,
                                                    metric_list=metric_list,
                                                    k=X_train.shape[0],  # 留一法就是N折验证，N是训练集的大小。
                                                    log=False,
                                                    log_dir='./log/')

        if n_class is None:
            raise ValueError('n_class for LeaveOneCrossExecuter can not be None.')
        else:
            self.n_class = n_class

    def execute(self, name, clf):
        '''
        执行实验，不记录日志。

        Notes
        -----
          - 这里必须返回一个测试器和一个训练好的分类器，因为写入日志要用。

        Parameters
        ----------
        name : str
            实验的名字。
        clf : Clfs
            实验获取的分类器，继承自接口Clfs。

        Returns
        -------
        clf : Clfs
            训练好的分类器。
        metric: Metrics
            有记录的Metric实例。

        Examples
        --------

        可以这么重写：
        ```python
        class MyExcuter(Excuter):
            def execute(self, name, clf):
                print(f'>> {name}')

                clf.fit(self.X_train, self.y_train)
                print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

                y_pred = clf.predict(self.X_test)

                mtc = Metrics(self.y_test, y_pred)

                return mtc, clf
        ```
        '''
        print(f'>> {name}')

        # k折交叉验证
        k_fold_x_train = jnp.array_split(self.X_train, self.k)
        k_fold_y_train = jnp.array_split(self.y_train, self.k)
        mtcs = []
        times = []  # validation training, teting time + test training, testing time
        for i in range(self.k):
            print(f'>>>> Validate: {i + 1}')
            x_train = jnp.concatenate(k_fold_x_train[:i] + k_fold_x_train[i + 1:])
            y_train = jnp.concatenate(k_fold_y_train[:i] + k_fold_y_train[i + 1:])
            x_test = k_fold_x_train[i]
            y_test = k_fold_y_train[i]
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            mtc = Metrics(y_test, y_pred, self.n_class)
            times.append([clf.get_training_time(), clf.get_testing_time()])
            mtcs.append(mtc)

        # real train & test
        print('>>>> Test:')
        clf.fit(self.X_train, self.y_train)  # 训练分类器
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)

        mtc = Metrics(self.y_test, y_pred, self.n_class)  # 构建测试器
        mtcs.append(mtc)
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')
        times.append([clf.get_training_time(), clf.get_testing_time()])

        return mtcs, clf, times  # 返回所有测试器和分类器和验证时间


class BootstrapExecuter(Executer):
    '''
    使用Bootstrap方法作为Validation的执行器。
    ==========
      - 快捷管理训练，测试，日志全过程，并灵活调试Classifier数组里面的各个模型。
      - 开启Log，支持中途运行出错，结果不丢失。
      - 在使用的时候根据需要重写execute(self)方法。

    Parameters
    ----------
    X_train : jnp.ndarray
        训练集的X。
    y_train : jnp.ndarray
        训练集的y。
    X_test : jnp.ndarray
        测试集的X。
    y_test : jnp.ndarray
        测试集的y。
    clf_dict : dict
        Clf字典。包含多个实验的{name : Clf}
    metric_list : list
        测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
    n_bootstraps : int
        Bootstrap重采样的次数。
    log : bool
        是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将测试结果写入到同一文件夹的test.csv，将Validation结果写入到同一文件夹的valid.csv。
    log_dir : str
        存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
    '''

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 n_bootstraps=100,
                 log=False,
                 random_state=42,
                 log_dir='./log/'):

        super(BootstrapExecuter, self).__init__(X_train, y_train, X_test, y_test,
                                                clf_dict, metric_list, log, log_dir)

        self.n_bootstraps = n_bootstraps
        self.random_state = random_state

        metrics = self.metric_list + ['training time', 'testing time']

        self.test = pd.DataFrame(columns=['model'] + metrics)
        self.valid = pd.DataFrame(columns=['model'] + [f'{x}_{suffix}' for x in metrics for suffix in ['mean', 'std']])

    def execute(self, name, clf):

        def __resample(key, X, y):
            """
            执行Bootstrap采样。

            参数:
            - key: JAX随机数生成器的键
            - X: 特征矩阵（JAX数组）
            - y: 标签向量（JAX数组）

            返回:
            - 新的key: 更新后的随机数生成器键
            - X_resampled: 经过Bootstrap采样后的特征矩阵
            - y_resampled: 对应的标签向量
            """
            # 分裂当前的key以获得新的子key，并更新key
            key, subkey = random.split(key)

            # 获取样本数量
            n_samples = len(X)

            # 使用random.choice从原始索引中有放回地抽取样本
            indices = random.choice(subkey, jnp.arange(n_samples), shape=(n_samples,), replace=True)

            # 根据抽样得到的索引获取重采样的X和y
            X_resampled = X[indices]
            y_resampled = y[indices]

            return key, X_resampled, y_resampled

        print(f'>> {name}')

        mtcs = []
        times = []
        key = random.PRNGKey(self.random_state)
        for i in range(self.n_bootstraps):
            # Bootstrap 采样
            print(f'>>>> Validate: {i + 1}')
            key, X_resampled, y_resampled = __resample(key, self.X_train, self.y_train)
            clf.fit(X_resampled, y_resampled)

            y_pred = clf.predict(X_resampled)
            mtc = Metrics(y_resampled, y_pred)
            mtcs.append(mtc)
            times.append([clf.get_training_time(), clf.get_testing_time()])

        # 真实的训练和测试
        print('>>>> Test:')
        clf.fit(self.X_train, self.y_train)
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)
        mtc = Metrics(self.y_test, y_pred)
        mtcs.append(mtc)
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')
        times.append([clf.get_training_time(), clf.get_testing_time()])

        return mtcs, clf, times

    def logline(self, name, mtcs: list, clf, times):
        '''
        将某次实验的结果写入日志df。
        '''

        test_mtc = mtcs.pop()
        test_times = times.pop()

        def getline(mtc):
            func_list = []
            for metric in self.metric_list:
                func = getattr(mtc, metric, None)
                if callable(func):
                    func_list.append(func)
                else:
                    raise ValueError(f'{metric} is not in Metric.')

            return [func() for func in func_list]

        self.test.loc[len(self.test)] = [name] + getline(test_mtc) + test_times  # 获取测试的结果

        valid_rows = [getline(mtc) + times[ix] for ix, mtc in enumerate(mtcs)]
        valids_array = jnp.array(valid_rows)

        mean_vals = jnp.mean(valids_array, axis=0).tolist()
        std_vals = jnp.std(valids_array, axis=0).tolist()

        valid_result = []
        for mean, std in zip(mean_vals, std_vals):
            valid_result.append(mean)
            valid_result.append(std)

        self.valid.loc[len(self.valid)] = [name] + valid_result

    def save_df(self):
        super().save_df()
        self.valid.to_csv(os.path.join(self.log_path, 'valid.csv'), index=False)

    def format_print(self, sort_by=('accuracy', 'accuracy_mean'), ascending=False, precision=4, time=False):
        '''
        表格的格式化输出。

        Parameters
        ----------
        sort_by : str
            按照哪个指标进行排序。接受两个位置，分别是测试和验证的指标。比如：('accuracy', 'accuracy_mean')，表示测试集按照accuracy指标进行排序，验证集按照accuracy_mean指标进行排序。
        ascending : bool
            是否升序。
        precision : int
            保留几位小数。
        time: bool
            是否显示训练和测试时间。
        '''

        if sort_by is not None:
            print(f'\n>> Test Result, sort by \'{sort_by[0]}\'.')
            if not time:
                temp_table = self.test.sort_values(sort_by[0], ascending=ascending).drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.test.sort_values(sort_by[0], ascending=ascending)
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

            print(f'\n>> Validation Result(Mean ± Std), sort by \'{sort_by[1]}\'.')
            if not time:
                temp_table = self.valid.sort_values(sort_by[1], ascending=ascending)
                temp_table = combine_mean_std(temp_table, precision=precision)
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.valid.sort_values(sort_by[1], ascending=ascending)
                temp_table = combine_mean_std(temp_table, precision=precision)
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                showindex=False
            ))
        else:
            print('\n>> Test Result.')
            if not time:
                temp_table = self.test.drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.test
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

            print('\n>> Validation Result(Mean ± Std).')
            if not time:
                temp_table = self.valid
                temp_table = combine_mean_std(temp_table, precision=precision)
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.valid
                temp_table = combine_mean_std(temp_table, precision=precision)
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                showindex=False
            ))

    def run_all(self, sort_by=['accuracy', 'accuracy_mean'], ascending=False, precision=4, time=False):
        '''
        运行所有实验。

        Parameters
        ----------
        sort_by : str
            按照哪个指标进行排序。接受两个位置，分别是测试和验证的指标。比如：('accuracy', 'accuracy_mean')，表示测试集按照accuracy指标进行排序，验证集按照accuracy_mean指标进行排序。
        ascending : bool
            是否升序。
        precision : int
            保留几位小数。
        time: bool
            是否显示训练和测试时间。
        '''

        for name, clf in self.clf_dict.items():
            mtc, clf, times = self.execute(name, clf)

            self.logline(name, mtc, clf, times)

        self.format_print(sort_by, ascending, precision, time)
