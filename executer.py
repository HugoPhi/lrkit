import atexit
from datetime import datetime
import toml
import os
import traceback
from tabulate import tabulate
import pandas as pd
import numpy as np
import numpy.random as random

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
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    std_cols = [col for col in df.columns if col.endswith('_std')]

    combined_df = pd.DataFrame()

    for mean_col, std_col in zip(mean_cols, std_cols):
        metric_name = mean_col.replace('_mean', '')
        combined_df[metric_name] = (
            df[mean_col].round(precision).astype(str) + " ± " + df[std_col].round(precision).astype(str)
        )

    combined_df['model'] = df['model']
    combined_df = combined_df[['model'] + [col for col in combined_df.columns if col != 'model']]

    return combined_df


class Executer:
    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 log_dir='./log/'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf_dict = clf_dict
        self.metric_list = metric_list
        self.log = log

        self.df = pd.DataFrame(columns=['model'] + self.metric_list + ['training time', 'testing time'])

        if log:
            self.log_dir = log_dir
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.log_path = os.path.join(self.log_dir, f'{datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}/')
            os.mkdir(self.log_path)

            hyper_config = dict()
            for name, clf in clf_dict.items():
                hyper_config[name] = clf.get_params()

            toml.dump(hyper_config, open(os.path.join(self.log_path, 'hyper.toml'), 'w'))
            atexit.register(self.save_df)

    def save_df(self):
        self.df.to_csv(os.path.join(self.log_path, 'result.csv'), index=False)

    def execute(self, name, clf):
        print(f'>> {name}')

        clf.fit(self.X_train, self.y_train)
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)
        mtc = Metrics(self.y_test, y_pred)
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')

        time = [clf.get_training_time(), clf.get_testing_time()]
        return mtc, clf, time

    def logline(self, name, mtc, clf, time):
        func_list = []
        for metric in self.metric_list:
            func = getattr(mtc, metric, None)
            if callable(func):
                func_list.append(func)
            else:
                raise ValueError(f'{metric} is not in Metric.')

        self.df.loc[len(self.df)] = [name] + [func() for func in func_list] + time

    def run(self, key):
        if key in self.clf_dict.keys():
            mtc, clf, time = self.execute(key, self.clf_dict[key])
            self.logline(key, mtc, clf, time)
        else:
            raise KeyError(f'{key} is not in clf_dict')

    def step(self):
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
        if sort_by is not None:
            print(f'\n>> Test Result, sort by \'{sort_by}\'.')
            temp_table = self.df.sort_values(sort_by, ascending=ascending)
            if not time:
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))
        else:
            print('\n>> Test Result.')
            temp_table = self.df if time else self.df.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

    def run_all(self, sort_by=None, ascending=False, precision=4, time=False):
        for name, clf in self.clf_dict.items():
            mtc, clf, time_vals = self.execute(name, clf)
            self.logline(name, mtc, clf, time_vals)
        self.format_print(sort_by, ascending, precision, time)

    def get_result(self):
        return self.df


class NonValidExecuter(Executer):
    pass


class KFlodCrossExecuter(Executer):
    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 k=10,
                 log=False,
                 log_dir='./log/'):
        super().__init__(X_train, y_train, X_test, y_test, clf_dict, metric_list, log, log_dir)
        self.k = k
        if k < 1:
            raise ValueError(f'k should >= 1, but get {self.k}')

        metrics = self.metric_list + ['training time', 'testing time']
        self.test = pd.DataFrame(columns=['model'] + metrics)
        self.valid = pd.DataFrame(columns=['model'] + [f'{x}_{suffix}' for x in metrics for suffix in ['mean', 'std']])

    def execute(self, name, clf):
        print(f'>> {name}')

        k_fold_x_train = np.array_split(self.X_train, self.k)
        k_fold_y_train = np.array_split(self.y_train, self.k)
        mtcs = []
        times = []
        for i in range(self.k):
            print(f'>>>> Validate: {i + 1}')
            x_train = np.concatenate(k_fold_x_train[:i] + k_fold_x_train[i + 1:])
            y_train = np.concatenate(k_fold_y_train[:i] + k_fold_y_train[i + 1:])
            x_test = k_fold_x_train[i]
            y_test = k_fold_y_train[i]
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            mtc = Metrics(y_test, y_pred)
            times.append([clf.get_training_time(), clf.get_testing_time()])
            mtcs.append(mtc)

        print('>>>> Test:')
        clf.fit(self.X_train, self.y_train)
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)
        mtc = Metrics(self.y_test, y_pred)
        mtcs.append(mtc)
        times.append([clf.get_training_time(), clf.get_testing_time()])
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')

        return mtcs, clf, times

    def logline(self, name, mtcs, clf, times):
        test_mtc = mtcs.pop()
        test_times = times.pop()

        def getline(mtc):
            return [getattr(mtc, metric)() for metric in self.metric_list]

        self.test.loc[len(self.test)] = [name] + getline(test_mtc) + test_times

        valid_rows = [getline(mtc) + times[ix] for ix, mtc in enumerate(mtcs)]
        valids_array = np.array(valid_rows)

        mean_vals = np.mean(valids_array, axis=0).tolist()
        std_vals = np.std(valids_array, axis=0).tolist()

        valid_result = []
        for mean, std in zip(mean_vals, std_vals):
            valid_result.append(mean)
            valid_result.append(std)

        self.valid.loc[len(self.valid)] = [name] + valid_result

    def save_df(self):
        self.test.to_csv(os.path.join(self.log_path, 'test.csv'), index=False)
        self.valid.to_csv(os.path.join(self.log_path, 'valid.csv'), index=False)

    def format_print(self, sort_by=('accuracy', 'accuracy_mean'), ascending=False, precision=4, time=False):
        if sort_by is not None:
            print(f'\n>> Test Result, sort by \'{sort_by[0]}\'.')
            temp_table = self.test.sort_values(sort_by[0], ascending=ascending)
            if not time:
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

            print(f'\n>> Validation Result(Mean ± Std), sort by \'{sort_by[1]}\'.')
            temp_table = self.valid.sort_values(sort_by[1], ascending=ascending)
            temp_table = combine_mean_std(temp_table, precision=precision)
            if not time:
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                showindex=False
            ))
        else:
            print('\n>> Test Result.')
            temp_table = self.test if time else self.test.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

            print('\n>> Validation Result(Mean ± Std).')
            temp_table = combine_mean_std(self.valid, precision=precision)
            if not time:
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                showindex=False
            ))

    def run_all(self, sort_by=('accuracy', 'accuracy_mean'), ascending=False, precision=4, time=False):
        for name, clf in self.clf_dict.items():
            mtcs, clf, times = self.execute(name, clf)
            self.logline(name, mtcs, clf, times)
        self.format_print(sort_by, ascending, precision, time)


class LeaveOneCrossExecuter(KFlodCrossExecuter):
    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 n_class=None,
                 log_dir='./log/'):
        super().__init__(X_train, y_train, X_test, y_test,
                         clf_dict=clf_dict,
                         metric_list=metric_list,
                         k=X_train.shape[0],
                         log=log,
                         log_dir=log_dir)

        if n_class is None:
            raise ValueError('n_class can not be None in LeaveOneCrossExecuter.')
        else:
            self.n_class = n_class

    def execute(self, name, clf):
        print(f'>> {name}')

        k_fold_x_train = np.array_split(self.X_train, self.k)
        k_fold_y_train = np.array_split(self.y_train, self.k)
        mtcs = []
        times = []
        for i in range(self.k):
            print(f'>>>> Validate: {i + 1}')
            x_train = np.concatenate(k_fold_x_train[:i] + k_fold_x_train[i + 1:])
            y_train = np.concatenate(k_fold_y_train[:i] + k_fold_y_train[i + 1:])
            x_test = k_fold_x_train[i]
            y_test = k_fold_y_train[i]
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            mtc = Metrics(y_test, y_pred, self.n_class)
            times.append([clf.get_training_time(), clf.get_testing_time()])
            mtcs.append(mtc)

        print('>>>> Test:')
        clf.fit(self.X_train, self.y_train)
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)
        mtc = Metrics(self.y_test, y_pred, self.n_class)
        mtcs.append(mtc)
        times.append([clf.get_training_time(), clf.get_testing_time()])
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')

        return mtcs, clf, times


class BootstrapExecuter(Executer):
    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 n_bootstraps=100,
                 log=False,
                 random_state=42,
                 log_dir='./log/'):
        super().__init__(X_train, y_train, X_test, y_test, clf_dict, metric_list, log, log_dir)
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
        self.rng = random.default_rng(random_state)

        metrics = self.metric_list + ['training time', 'testing time']
        self.test = pd.DataFrame(columns=['model'] + metrics)
        self.valid = pd.DataFrame(columns=['model'] + [f'{x}_{suffix}' for x in metrics for suffix in ['mean', 'std']])

    def __resample(self, X, y):
        n_samples = X.shape[0]
        indices = self.rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def execute(self, name, clf):
        print(f'>> {name}')

        mtcs = []
        times = []
        for i in range(self.n_bootstraps):
            print(f'>>>> Validate: {i + 1}')
            X_resampled, y_resampled = self.__resample(self.X_train, self.y_train)
            clf.fit(X_resampled, y_resampled)
            y_pred = clf.predict(X_resampled)
            mtc = Metrics(y_resampled, y_pred)
            times.append([clf.get_training_time(), clf.get_testing_time()])
            mtcs.append(mtc)

        print('>>>> Test:')
        clf.fit(self.X_train, self.y_train)
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)
        mtc = Metrics(self.y_test, y_pred)
        mtcs.append(mtc)
        times.append([clf.get_training_time(), clf.get_testing_time()])
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')

        return mtcs, clf, times

    def logline(self, name, mtcs, clf, times):
        test_mtc = mtcs.pop()
        test_times = times.pop()

        def getline(mtc):
            return [getattr(mtc, metric)() for metric in self.metric_list]

        self.test.loc[len(self.test)] = [name] + getline(test_mtc) + test_times

        valid_rows = [getline(mtc) + times[ix] for ix, mtc in enumerate(mtcs)]
        valids_array = np.array(valid_rows)

        mean_vals = np.mean(valids_array, axis=0).tolist()
        std_vals = np.std(valids_array, axis=0).tolist()

        valid_result = []
        for mean, std in zip(mean_vals, std_vals):
            valid_result.append(mean)
            valid_result.append(std)

        self.valid.loc[len(self.valid)] = [name] + valid_result

    def save_df(self):
        super().save_df()
        self.valid.to_csv(os.path.join(self.log_path, 'valid.csv'), index=False)

    def format_print(self, sort_by=('accuracy', 'accuracy_mean'), ascending=False, precision=4, time=False):
        if sort_by is not None:
            print(f'\n>> Test Result, sort by \'{sort_by[0]}\'.')
            temp_table = self.test.sort_values(sort_by[0], ascending=ascending)
            if not time:
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

            print(f'\n>> Validation Result(Mean ± Std), sort by \'{sort_by[1]}\'.')
            temp_table = self.valid.sort_values(sort_by[1], ascending=ascending)
            temp_table = combine_mean_std(temp_table, precision=precision)
            if not time:
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                showindex=False
            ))
        else:
            print('\n>> Test Result.')
            temp_table = self.test if time else self.test.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                floatfmt='.4f',
                showindex=False
            ))

            print('\n>> Validation Result(Mean ± Std).')
            temp_table = combine_mean_std(self.valid, precision=precision)
            if not time:
                temp_table = temp_table.drop(columns=['training time', 'testing time'])
            print(tabulate(
                temp_table,
                headers='keys',
                tablefmt='fancy_grid',
                showindex=False
            ))

    def run_all(self, sort_by=('accuracy', 'accuracy_mean'), ascending=False, precision=4, time=False):
        for name, clf in self.clf_dict.items():
            mtcs, clf, times = self.execute(name, clf)
            self.logline(name, mtcs, clf, times)
        self.format_print(sort_by, ascending, precision, time)
