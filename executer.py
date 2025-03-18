"""
Experiment Executor Module

This module defines the base functionality for managing and executing machine learning experiments.
It handles the training, testing, logging, and evaluation of classifiers across multiple experiments.
The module also supports different validation techniques (e.g., K-fold, Leave-One-Out, Bootstrap) and
provides a flexible framework for experiment tracking and result analysis.

Key Components:
---------------
- Executer: The base class for managing training and testing workflows.
- NonValidExecuter: A subclass that skips validation during the experiment.
- KFlodCrossExecuter: A subclass that uses K-fold cross-validation for validation.
- LeaveOneCrossExecuter: A subclass that uses Leave-One-Out Cross-Validation for validation.
- BootstrapExecuter: A subclass that uses Bootstrap resampling for validation.
- Log management: Supports logging of experiment parameters, results, and execution time.

Common Workflow:
----------------
1. Initialize the desired executor class (e.g., KFlodCrossExecuter, BootstrapExecuter).
2. Define the classifier models in `clf_dict` and specify evaluation metrics.
3. Run experiments using methods like `run_all()` or `step()`.
4. Access experiment results through the `get_result()` method.

Example Usage:
--------------
1. K-Fold Cross-Validation Example:
    executer = KFlodCrossExecuter(X_train, y_train, X_test, y_test, clf_dict, k=10, log=True)
    executer.run_all()

2. Bootstrap Resampling Example:
    executer = BootstrapExecuter(X_train, y_train, X_test, y_test, clf_dict, n_bootstraps=100, log=True)
    executer.run_all()

3. Leave-One-Out Cross-Validation Example:
    executer = LeaveOneCrossExecuter(X_train, y_train, X_test, y_test, clf_dict, n_class=3, log=True)
    executer.run_all()

Attributes:
-----------
- X_train, y_train, X_test, y_test: Training and testing data used for model evaluation.
- clf_dict: A dictionary of classifier models, where each key is the experiment name and value is the classifier.
- metric_list: A list of evaluation metrics (e.g., accuracy, F1-score) for model evaluation.
- log_dir: Directory to store log files with experiment parameters and results.
"""

import os
import atexit
import traceback
from datetime import datetime

import toml
import pandas as pd
from jax import random
import jax.numpy as jnp
from tabulate import tabulate

from .metric import Metrics


def combine_mean_std(df, precision=4):
    """
    Merges the '_mean' and '_std' columns of a DataFrame into a single column
    with the format 'mean ± std' and limits the precision of the floating-point numbers.

    This function is particularly useful when you want to represent the mean
    and standard deviation together in a more readable format for reporting purposes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns that end with '_mean' and '_std'.
    precision : int, optional
        The precision of the floating-point numbers, by default 4.

    Returns
    -------
    pd.DataFrame
        A DataFrame with combined 'mean ± std' columns and rounded to the specified precision.

    Examples
    --------
    Given a DataFrame:
    ```
    df = pd.DataFrame({
        'model': ['Model1', 'Model2'],
        'accuracy_mean': [0.95, 0.92],
        'accuracy_std': [0.01, 0.02]
    })
    ```
    The function will return:
    ```
    model    accuracy
    0  Model1   0.9500 ± 0.0100
    1  Model2   0.9200 ± 0.0200
    ```
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
    """
    Base class for executing training and testing experiments.
    ========================================================
    This class provides convenient methods for managing training, testing,
    and logging throughout the entire experiment process. It is designed to be
    flexible so you can easily modify the execution flow for different models
    by overriding the `execute()` method.

    Parameters
    ----------
    X_train : jnp.ndarray
        Feature matrix for the training data.
    y_train : jnp.ndarray
        Labels for the training data.
    X_test : jnp.ndarray
        Feature matrix for the test data.
    y_test : jnp.ndarray
        Labels for the test data.
    clf_dict : dict
        Dictionary containing classifiers with their corresponding names.
    metric_list : list
        List of metrics to be evaluated, e.g., ['accuracy', 'macro_f1', 'micro_f1'].
    log : bool
        If True, logs will be created and saved in the specified directory.
    log_dir : str
        Directory where logs will be stored. Each experiment run will be saved
        in a subdirectory named with the current date and time.

    Example
    -------
    # Example usage:
    executer = Executer(X_train, y_train, X_test, y_test, clf_dict, metric_list)
    executer.run_all()
    """

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 log_dir='./log/'):
        """
        Initializes the Executer class with necessary data and settings.

        Parameters are the same as those described in the class-level docstring.

        Example
        -------
        executer = Executer(X_train, y_train, X_test, y_test, clf_dict)
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf_dict = clf_dict
        self.metric_list = metric_list
        self.log = log

        self.test = pd.DataFrame(columns=['model'] + self.metric_list + ['training time'] + ['testing time'])

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
        """
        Save the experiment results (DataFrame) to a CSV file in the log directory.

        Example:
        After running experiments, the DataFrame with all results is saved to
        'result.csv' in the appropriate log folder.
        """

        self.test.to_csv(os.path.join(self.log_path, 'result.csv'), index=False)

    def execute(self, name, clf):
        """
        Run a single experiment.

        Parameters
        ----------
        name : str
            The name of the experiment.
        clf : Clfs
            The classifier object to train and test.

        Returns
        -------
        clf : Clfs
            The trained classifier.
        metric : Metrics
            A Metrics instance containing the evaluation results.

        Example:
        To override the `execute` method, use the following structure:
        ```python
        class MyExecuter(Executer):
            def execute(self, name, clf):
                print(f'Running {name}')
                clf.fit(self.X_train, self.y_train)
                print(f'Train {name} Cost: {clf.get_training_time()} seconds')
                y_pred = clf.predict(self.X_test)
                metrics = Metrics(self.y_test, y_pred)
                return metrics, clf
        ```
        """

        print(f'>> {name}')

        clf.fit(self.X_train, self.y_train)  # 训练分类器
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)

        mtc = Metrics(self.y_test, y_pred)  # 构建测试器
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')

        time = [clf.get_training_time(), clf.get_testing_time()]

        return mtc, clf, time  # 返回测试器和分类器

    def logline(self, name, mtc, clf, time):
        """
        Log the results of a single experiment into the DataFrame.

        Parameters
        ----------
        name : str
            The name of the experiment.
        metrics : Metrics
            The metrics object containing evaluation metrics.
        clf : Clfs
            The classifier object.
        time : list
            The list containing training and testing times.

        Example:
        After an experiment, the result is logged into the DataFrame for future analysis.
        """

        func_list = []
        for metric in self.metric_list:
            func = getattr(mtc, metric, None)
            if callable(func):
                func_list.append(func)
            else:
                raise ValueError(f'{metric} is not in Metric.')

        self.test.loc[len(self.test)] = [name] + [func() for func in func_list] + time

    def run(self, key):
        """
        Run a single experiment and log the results without consumption clf in clf_dict.

        Parameters
        ----------
        key : str
            The name of the experiment.

        Example:
        executer.run('experiment_name')  # Runs the specified experiment
        """

        if key in self.clf_dict.keys():
            mtc, clf, time = self.execute(key, self.clf_dict[key])

            self.logline(key, mtc, clf, time)
        else:
            raise KeyError(f'{key} is not in clf_dict')

    def step(self):
        """
        Run experiments iteratively until all classifiers have been processed. This will consume clf in clf_dict until it is empty.

        Returns
        -------
        name : str
            The name of the experiment.
        clf : Clfs
            The classifier object for the experiment.

        Example:
        for name, clf in executer.step():
            print(f'Running {name} using classifier {clf}')
        """

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
        """
        Format and print the results as a table.

        Parameters
        ----------
        sort_by : str
            Metric to sort by, e.g., 'accuracy'.
        ascending : bool
            Whether to sort in ascending order.
        precision : int
            The number of decimal places to display.
        time : bool
            Whether to display training and testing time.

        Example:
        executer.format_print(sort_by='accuracy', ascending=True)
        """

        if sort_by is not None:
            print(f'\n>> Test Result, sort by \'{sort_by}\'.')
            if not time:
                temp_table = self.test.sort_values(sort_by, ascending=ascending).drop(columns=['training time', 'testing time'])
            else:
                temp_table = self.test.sort_values(sort_by, ascending=ascending)

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

    def run_all(self, sort_by=None, ascending=False, precision=4, time=False):
        """
        Run all experiments and log the results.

        Parameters
        ----------
        sort_by : str, optional
            Metric to sort by, e.g., 'accuracy'.
        ascending : bool
            Whether to sort in ascending order.
        precision : int
            The number of decimal places to display.
        time : bool
            Whether to display training and testing time.

        Example:
        executer.run_all(sort_by='accuracy', ascending=False)
        """

        for name, clf in self.clf_dict.items():
            mtc, clf, time = self.execute(name, clf)

            self.logline(name, mtc, clf, time)

        self.format_print(sort_by, ascending, precision, time)

    def get_result(self):
        """
        Return the experiment results as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the experiment results.

        Example:
        test = executer.get_result()  # Get the result DataFrame
        """

        return self.test


class NonValidExecuter(Executer):
    """
    Executor class for training and testing without validation.
    ========================================================
    This class extends the base `Executer` class and is designed to manage the entire
    training, testing, and logging process without performing validation. It allows
    for easy management of experiments with classifiers and metrics, and it supports
    logging, ensuring that experiment results are saved even if errors occur during execution.

    Key Features:
    -------------
    - Manages the entire lifecycle of training, testing, and logging.
    - Enables flexible adjustments to classifier models within the `clf_dict` array.
    - Supports logging, saving parameters and results into the specified directory.
    - You can override the `execute(self)` method as needed for different classifier behaviors.

    Parameters:
    ----------
    X_train : jnp.ndarray
        Feature matrix for the training data.
    y_train : jnp.ndarray
        Labels for the training data.
    X_test : jnp.ndarray
        Feature matrix for the test data.
    y_test : jnp.ndarray
        Labels for the test data.
    clf_dict : dict
        A dictionary of classifiers where each key is an experiment name and each value is the classifier instance.
    metric_list : list, optional
        A list of evaluation metrics, such as ['accuracy', 'macro_f1', 'micro_f1', 'avg_recall']. Default is `['accuracy', 'macro_f1', 'micro_f1', 'avg_recall']`.
    log : bool, optional
        If True, enables logging of hyperparameters and results. The logs will be saved in the `log_dir` directory. Default is False.
    log_dir : str, optional
        Directory where logs will be saved. If logging is enabled, logs will be saved in a subfolder named with the current timestamp. Default is `'./log/'`.

    Example:
    --------
    # Example usage for running an experiment without validation:
    executer = NonValidExecuter(X_train, y_train, X_test, y_test, clf_dict, log=True)
    executer.run_all()
    """

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 log_dir='./log/'):
        """
        Initializes the NonValidExecuter class for training and testing without validation.

        This constructor calls the parent `Executer` class and initializes necessary
        parameters like the training data, testing data, classifiers, and logging options.

        Parameters are the same as described in the class-level docstring.

        Example:
        --------
        executer = NonValidExecuter(X_train, y_train, X_test, y_test, clf_dict)
        """

        super(NonValidExecuter, self).__init__(X_train, y_train, X_test, y_test,
                                               clf_dict=clf_dict, metric_list=metric_list, log=log, log_dir=log_dir)


class KFlodCrossExecuter(Executer):
    """
    Executor class using K-fold cross-validation for model validation.
    =================================================================
    This class extends the `Executer` base class and integrates K-fold cross-validation
    to assess classifier performance. It manages the entire training, validation, testing,
    and logging process, allowing flexibility in experiment execution.

    Key Features:
    -------------
    - Simplifies the management of training, testing, and logging workflows.
    - Supports K-fold cross-validation for more robust evaluation.
    - Allows for model adjustments within the `clf_dict` array for various experiments.
    - Logs results, ensuring data persistence even if errors occur during execution.
    - You can override the `execute(self)` method to customize behavior for different classifiers.

    Parameters:
    ----------
    X_train : jnp.ndarray
        Feature matrix for the training data.
    y_train : jnp.ndarray
        Labels for the training data.
    X_test : jnp.ndarray
        Feature matrix for the test data.
    y_test : jnp.ndarray
        Labels for the test data.
    clf_dict : dict
        Dictionary containing classifier models, where each key is an experiment name and the value is the classifier.
    metric_list : list, optional
        List of evaluation metrics (e.g., ['accuracy', 'macro_f1', 'micro_f1']). Default is `['accuracy', 'macro_f1', 'micro_f1', 'avg_recall']`.
    k : int, optional
        The number of folds in cross-validation (k >= 1). Default is 10.
    log : bool, optional
        If True, logging is enabled and results are saved in `log_dir`. Default is False.
    log_dir : str, optional
        Directory where logs will be stored. Default is './log/'.

    Example:
    --------
    # Example usage for running K-fold cross-validation experiments:
    executer = KFlodCrossExecuter(X_train, y_train, X_test, y_test, clf_dict, k=5, log=True)
    executer.run_all()
    """

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 k=10,
                 log=False,
                 log_dir='./log/'):
        """
        Initializes the KFlodCrossExecuter with the necessary data and settings.

        Parameters:
        -----------
        X_train, y_train, X_test, y_test : jnp.ndarray
            Training and testing data features and labels.
        clf_dict : dict
            Dictionary containing classifier models for experimentation.
        metric_list : list, optional
            List of evaluation metrics for performance measurement. Default includes ['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'].
        k : int, optional
            The number of folds for cross-validation (k >= 1). Default is 10.
        log : bool, optional
            Whether to enable logging. Default is False.
        log_dir : str, optional
            Directory where logs are saved. Default is './log/'.

        Example:
        --------
        executer = KFlodCrossExecuter(X_train, y_train, X_test, y_test, clf_dict, k=5, log=True)
        """

        super(KFlodCrossExecuter, self).__init__(X_train, y_train, X_test, y_test,
                                                 clf_dict, metric_list, log, log_dir)

        self.k = k
        if k < 1:
            raise ValueError(f'k should >= 1, but get {self.k}')

        metrics = self.metric_list + ['training time', 'testing time']

        self.test = pd.DataFrame(columns=['model'] + metrics)
        self.valid = pd.DataFrame(columns=['model'] + [f'{x}_{suffix}' for x in metrics for suffix in ['mean', 'std']])

    def execute(self, name, clf):
        """
        Executes an experiment using K-fold cross-validation and returns evaluation metrics.

        Parameters:
        ----------
        name : str
            Name of the experiment.
        clf : Clfs
            Classifier model used for the experiment.

        Returns:
        -------
        clf : Clfs
            Trained classifier model.
        metric : Metrics
            Recorded metrics for the experiment.

        Example:
        --------
        You can override this method for custom behavior like so:
        ```python
        class MyExecuter(KFlodCrossExecuter):
            def execute(self, name, clf):
                print(f'Running {name}')
                clf.fit(self.X_train, self.y_train)
                y_pred = clf.predict(self.X_test)
                metrics = Metrics(self.y_test, y_pred)
                return metrics, clf
        ```
        """

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
        """
        Logs the results of an experiment into the DataFrame for both testing and validation.

        Parameters:
        ----------
        name : str
            Name of the experiment.
        mtcs : list
            List of recorded metrics from K-fold validation and final testing.
        clf : Clfs
            Classifier used in the experiment.
        times : list
            List containing training and testing times.

        Example:
        --------
        After an experiment, this method stores the results into the DataFrame for later analysis.
        """

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
        """
        Saves the results DataFrame to CSV files in the log directory.

        Example:
        --------
        This method is invoked to save both test and validation results after all experiments.
        """

        self.test.to_csv(os.path.join(self.log_path, 'test.csv'), index=False)
        self.valid.to_csv(os.path.join(self.log_path, 'valid.csv'), index=False)

    def format_print(self, sort_by=('accuracy', 'accuracy_mean'), ascending=False, precision=4, time=False):
        """
        Formats and prints the results as a table, with optional sorting and time display.

        Parameters:
        ----------
        sort_by : tuple
            A tuple of two strings specifying the metrics to sort by for test and validation sets (e.g., ('accuracy', 'accuracy_mean')).
        ascending : bool
            Whether to sort in ascending order. Default is False.
        precision : int
            Number of decimal places to display. Default is 4.
        time : bool
            Whether to display training and testing times.

        Example:
        --------
        executer.format_print(sort_by=('accuracy', 'accuracy_mean'), ascending=True)
        """

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
        """
        Runs all experiments in the `clf_dict` and prints the results.

        Parameters:
        ----------
        sort_by : list
            A list of two strings specifying the metrics to sort by for test and validation sets.
        ascending : bool
            Whether to sort the results in ascending order.
        precision : int
            Number of decimal places to display.
        time : bool
            Whether to include training and testing times in the output.

        Example:
        --------
        executer.run_all(sort_by=['accuracy', 'accuracy_mean'], ascending=True)
        """

        for name, clf in self.clf_dict.items():
            mtc, clf, times = self.execute(name, clf)

            self.logline(name, mtc, clf, times)

        self.format_print(sort_by, ascending, precision, time)

    def get_result(self):
        """
        Return the experiment results as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the experiment results.

        Example:
        test, valid = executer.get_result()  # Get the result DataFrame
        """

        return self.test, self.valid


class LeaveOneCrossExecuter(KFlodCrossExecuter):
    """
    Executor class using Leave-One-Out Cross-Validation (LOO-CV) for model validation.
    ===========================================================================
    This class extends the `KFlodCrossExecuter` class and uses Leave-One-Out Cross-Validation (LOO-CV)
    to evaluate classifier performance. In this validation method, for each iteration, one sample is used
    as the test set while the remaining samples are used as the training set.

    Key Features:
    -------------
    - Manages training, testing, and logging workflows.
    - Implements Leave-One-Out Cross-Validation (LOO-CV) for more thorough model evaluation.
    - Supports logging of results, ensuring data persistence even if errors occur.
    - Flexibly adjusts classifiers within the `clf_dict` array for multiple experiments.
    - Customizable `execute(self)` method to fit specific classifier needs.

    Parameters:
    ----------
    X_train : jnp.ndarray
        Feature matrix for the training data.
    y_train : jnp.ndarray
        Labels for the training data.
    X_test : jnp.ndarray
        Feature matrix for the test data.
    y_test : jnp.ndarray
        Labels for the test data.
    clf_dict : dict
        Dictionary of classifiers, where each key is an experiment name and the value is the classifier instance.
    metric_list : list, optional
        List of evaluation metrics, such as ['accuracy', 'macro_f1', 'micro_f1'], default is `['accuracy', 'macro_f1', 'micro_f1', 'avg_recall']`.
    log : bool, optional
        If True, logging is enabled and results are saved in the `log_dir` directory. Default is False.
    log_dir : str, optional
        Directory where logs will be stored. Default is `'./log/'`.
    n_class : int
        The number of classes in the classification task.

    Example:
    --------
    # Example usage for running LOO-CV experiments:
    executer = LeaveOneCrossExecuter(X_train, y_train, X_test, y_test, clf_dict, n_class=3, log=True)
    executer.run_all()
    """

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 n_class=None,
                 log_dir='./log/'):
        """
        Initializes the LeaveOneCrossExecuter class for Leave-One-Out Cross-Validation (LOO-CV).

        Parameters:
        -----------
        X_train, y_train, X_test, y_test : jnp.ndarray
            Training and testing data (features and labels).
        clf_dict : dict
            Dictionary containing classifier models for experimentation.
        metric_list : list, optional
            List of evaluation metrics for performance measurement. Default is `['accuracy', 'macro_f1', 'micro_f1', 'avg_recall']`.
        log : bool, optional
            Whether to enable logging. Default is False.
        n_class : int
            The number of classes in the classification task.
        log_dir : str, optional
            Directory where logs will be saved. Default is `'./log/'`.

        Example:
        --------
        executer = LeaveOneCrossExecuter(X_train, y_train, X_test, y_test, clf_dict, n_class=3, log=True)
        """

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
        """
        Executes an experiment using Leave-One-Out Cross-Validation (LOO-CV) and returns evaluation metrics.

        Parameters:
        ----------
        name : str
            Name of the experiment.
        clf : Clfs
            Classifier used for the experiment.

        Returns:
        -------
        clf : Clfs
            Trained classifier.
        metric : Metrics
            Recorded metrics for the experiment.

        Example:
        --------
        You can override this method for custom behavior like so:
        ```python
        class MyExecuter(LeaveOneCrossExecuter):
            def execute(self, name, clf):
                print(f'Running {name}')
                clf.fit(self.X_train, self.y_train)
                y_pred = clf.predict(self.X_test)
                metrics = Metrics(self.y_test, y_pred)
                return metrics, clf
        ```
        """

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
    """
    Executor class using Bootstrap resampling for model validation.
    =============================================================
    This class extends the `Executer` base class and implements Bootstrap resampling
    for model validation. Bootstrap involves generating multiple random resamples of
    the data with replacement and evaluating model performance across these samples.

    Key Features:
    -------------
    - Manages the full cycle of training, testing, and logging for experiments.
    - Implements Bootstrap resampling for robust model evaluation.
    - Supports logging of results, even if the process encounters errors.
    - Allows for flexible adjustments to classifiers within the `clf_dict` array for multiple experiments.
    - You can override the `execute(self)` method to customize the model fitting process.

    Parameters:
    ----------
    X_train : jnp.ndarray
        Feature matrix for the training data.
    y_train : jnp.ndarray
        Labels for the training data.
    X_test : jnp.ndarray
        Feature matrix for the test data.
    y_test : jnp.ndarray
        Labels for the test data.
    clf_dict : dict
        Dictionary containing classifiers, where each key is an experiment name and the value is the classifier instance.
    metric_list : list, optional
        List of evaluation metrics, such as ['accuracy', 'macro_f1', 'micro_f1']. Default is `['accuracy', 'macro_f1', 'micro_f1', 'avg_recall']`.
    n_bootstraps : int, optional
        The number of bootstrap resamples to perform. Default is 100.
    log : bool, optional
        If True, logging is enabled and results are saved in the `log_dir` directory. Default is False.
    random_state : int, optional
        The random seed for reproducibility. Default is 42.
    log_dir : str, optional
        Directory where logs will be saved. Default is `'./log/'`.

    Example:
    --------
    # Example usage for running Bootstrap resampling experiments:
    executer = BootstrapExecuter(X_train, y_train, X_test, y_test, clf_dict, n_bootstraps=50, log=True)
    executer.run_all()
    """

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 n_bootstraps=100,
                 log=False,
                 random_state=42,
                 log_dir='./log/'):
        """
        Initializes the BootstrapExecuter class for model validation with Bootstrap resampling.

        Parameters are the same as described in the class-level docstring.

        Example:
        --------
        executer = BootstrapExecuter(X_train, y_train, X_test, y_test, clf_dict)
        """

        super(BootstrapExecuter, self).__init__(X_train, y_train, X_test, y_test,
                                                clf_dict, metric_list, log, log_dir)

        self.n_bootstraps = n_bootstraps
        self.random_state = random_state

        metrics = self.metric_list + ['training time', 'testing time']

        self.test = pd.DataFrame(columns=['model'] + metrics)
        self.valid = pd.DataFrame(columns=['model'] + [f'{x}_{suffix}' for x in metrics for suffix in ['mean', 'std']])

    def execute(self, name, clf):
        """
        Executes an experiment using Bootstrap resampling and returns evaluation metrics.

        Parameters:
        ----------
        name : str
            Name of the experiment.
        clf : Clfs
            Classifier used for the experiment.

        Returns:
        -------
        clf : Clfs
            Trained classifier.
        metric : Metrics
            Recorded metrics for the experiment.

        Example:
        --------
        You can override this method for custom behavior like so:
        ```python
        class MyExecuter(BootstrapExecuter):
            def execute(self, name, clf):
                print(f'Running {name}')
                clf.fit(self.X_train, self.y_train)
                y_pred = clf.predict(self.X_test)
                metrics = Metrics(self.y_test, y_pred)
                return metrics, clf
        ```
        """

        # Bootstrap resampling procedure
        def __resample(key, X, y):
            """
            Perform Bootstrap resampling on the given data.

            Parameters:
            ----------
            key : JAX PRNGKey
                Random number generator key.
            X : jnp.ndarray
                Feature matrix.
            y : jnp.ndarray
                Label vector.

            Returns:
            -------
            key : JAX PRNGKey
                Updated random number generator key.
            X_resampled : jnp.ndarray
                Resampled feature matrix.
            y_resampled : jnp.ndarray
                Resampled label vector.
            """

            key, subkey = random.split(key)
            n_samples = len(X)
            indices = random.choice(subkey, jnp.arange(n_samples), shape=(n_samples,), replace=True)
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
        """
        Logs the results of an experiment into the DataFrame for both testing and validation.

        Parameters:
        ----------
        name : str
            Name of the experiment.
        mtcs : list
            List of recorded metrics from Bootstrap resampling and final testing.
        clf : Clfs
            Classifier used in the experiment.
        times : list
            List containing training and testing times for each resample and final test.

        Example:
        --------
        After the experiment execution, the results are logged into the DataFrame for later analysis.
        """

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
        """
        Saves the results DataFrame to CSV files in the log directory.

        Example:
        --------
        After running experiments, the results are saved in `test.csv` and `valid.csv` under the log directory.
        """

        super().save_df()
        self.valid.to_csv(os.path.join(self.log_path, 'valid.csv'), index=False)

    def format_print(self, sort_by=('accuracy', 'accuracy_mean'), ascending=False, precision=4, time=False):
        """
        Formats and prints the results as a table, with optional sorting and time display.

        Parameters:
        ----------
        sort_by : tuple
            A tuple of two strings specifying the metrics to sort by for test and validation sets (e.g., ('accuracy', 'accuracy_mean')).
        ascending : bool
            Whether to sort in ascending order. Default is False.
        precision : int
            Number of decimal places to display. Default is 4.
        time : bool
            Whether to display training and testing times.

        Example:
        --------
        executer.format_print(sort_by=('accuracy', 'accuracy_mean'), ascending=True)
        """

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
        """
        Runs all experiments in the `clf_dict` and prints the results.

        Parameters:
        ----------
        sort_by : list
            A list of two strings specifying the metrics to sort by for test and validation sets.
        ascending : bool
            Whether to sort the results in ascending order.
        precision : int
            Number of decimal places to display.
        time : bool
            Whether to include training and testing times in the output.

        Example:
        --------
        executer.run_all(sort_by=['accuracy', 'accuracy_mean'], ascending=True)
        """

        for name, clf in self.clf_dict.items():
            mtc, clf, times = self.execute(name, clf)

            self.logline(name, mtc, clf, times)

        self.format_print(sort_by, ascending, precision, time)

    def get_result(self):
        """
        Return the experiment results as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the experiment results.

        Example:
        test, valid = executer.get_result()  # Get the result DataFrame
        """

        return self.test, self.valid
