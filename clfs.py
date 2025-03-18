"""
Classifier Interface Module

This module provides an abstract base class for creating classifier implementations with
consistent APIs, along with utility decorators for timing measurements. The design enables:
- Unified interface for different classifier implementations
- Separation of concerns between predictors, data handling, and model architecture
- Built-in training and inference timing metrics
- Parameter serialization for experiment reproducibility

Key Components:
- timing: Decorator for measuring method execution time
- Clfs: Abstract Base Class defining classifier interface

Example Usage:

1. Basic Implementation Example:
    class LogisticRegressionClassifier(Clfs):
        def __init__(self, learning_rate=0.01, iterations=100):
            super().__init__()
            self.learning_rate = learning_rate
            self.iterations = iterations

        @timing
        def fit(self, X_train, y_train, load=False):
            # Implementation here
            pass

        @timing
        def predict_proba(self, x_test):
            # Implementation here
            return probabilities

    # Instantiate and use classifier
    clf = LogisticRegressionClassifier(learning_rate=0.1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(f"Training time: {clf.get_training_time():.2f}s")
    print(f"Model parameters: {clf.get_params()}")

2. scikit-learn Compatibility Example:
    class SklearnWrapper(Clfs):
        def __init__(self, sklearn_model):
            super().__init__()
            self.model = sklearn_model

        @timing
        def fit(self, X_train, y_train, load=False):
            self.model.fit(X_train, y_train)
            return self

        @timing
        def predict_proba(self, x_test):
            return self.model.predict_proba(x_test)

    # Usage with scikit-learn model
    from sklearn.ensemble import RandomForestClassifier
    sklearn_model = RandomForestClassifier(n_estimators=100)
    clf = SklearnWrapper(sklearn_model)
    clf.fit(X_train, y_train)
    print(f"Training time: {clf.get_training_time():.2f}s")

3. PyTorch Compatibility Example:
    class TorchClassifier(Clfs):
        def __init__(self, model, optimizer, criterion, epochs=10):
            super().__init__()
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.epochs = epochs

        @timing
        def fit(self, X_train, y_train, load=False):
            self.model.train()
            for epoch in range(self.epochs):
                # Training loop implementation
                pass
            return self

        @timing
        def predict_proba(self, x_test):
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x_test)
                return torch.softmax(outputs, dim=1).numpy()

    # Usage with PyTorch model
    import torch
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    clf = TorchClassifier(model, optimizer, criterion, epochs=20)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
"""

import time
from functools import wraps
from abc import ABC, abstractmethod

import inspect
import jax.numpy as jnp


def timing(func):
    """
    Decorator for measuring and storing execution time of training/test methods.

    The measured time is stored in the object's:
    - training_time property for 'fit' method
    - testing_time property for 'predict' or 'predict_proba' methods

    Args:
        func (callable): Method to be timed. Must be one of ['fit', 'predict', 'predict_proba']

    Returns:
        callable: Wrapped method with timing functionality

    Raises:
        ValueError: If applied to unsupported method names

    Example:
        @timing
        def fit(self, X_train, y_train):
            # Training implementation
    """

    if func.__name__ in ['predict', 'predict_proba']:
        @wraps(func)
        def wrapper(self_obj, *args, **kwargs):
            start_time = time.time()
            result = func(self_obj, *args, **kwargs)
            elapsed_time = time.time() - start_time
            self_obj.testing_time = elapsed_time
            return result
        return wrapper
    elif func.__name__ in ['fit']:
        @wraps(func)
        def wrapper(self_obj, *args, **kwargs):
            start_time = time.time()
            result = func(self_obj, *args, **kwargs)
            elapsed_time = time.time() - start_time
            self_obj.training_time = elapsed_time
            return result
        return wrapper
    else:
        raise ValueError(f'{func.__name__} is not timable. Only fit, predict and predict_proba are timable.')


class Clfs(ABC):
    """
    Abstract Base Class defining standardized classifier interface.

    Subclasses must implement:
    - __init__: Constructor for initial parameters
    - fit: Training method implementation
    - predict_proba: Probability prediction implementation

    Provides default implementations for:
    - predict: Converts probability output to class labels
    - parameter management
    - timing metrics access

    Attributes:
        training_time (float): Time in seconds for last fit() call (-1 if not trained)
        testing_time (float): Time in seconds for last predict/predict_proba call (-1 if not tested)
        params (dict): Dictionary of constructor parameters for reproducibility
    """

    def __new__(cls, *args, **kwargs):
        """
        Instance factory that captures initialization parameters.

        Automatically populates the `params` attribute by analyzing __init__ arguments.
        Handles nested dictionary parameters through recursive merging.
        """

        instance = super().__new__(cls)
        original_init = cls.__init__
        sig = inspect.signature(original_init)
        bound_args = sig.bind(instance, *args, **kwargs)
        bound_args.apply_defaults()
        bound_args.arguments.pop('self', None)

        instance.params = {}

        def merge_params(params_dict, arguments):
            for key, value in arguments.items():
                if isinstance(value, dict):
                    merge_params(params_dict, value)
                else:
                    params_dict[key] = value

        merge_params(instance.params, bound_args.arguments)

        return instance

    @abstractmethod
    def __init__(self):
        """
        Initialize classifier instance.

        Note: Subclasses must call super().__init__() to properly initialize
        timing attributes.
        """

        self.training_time = -1
        self.testing_time = -1

    def predict(self, x_test) -> jnp.ndarray:
        """
        Predict class labels for input samples.

        Args:
            x_test (jnp.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            jnp.ndarray: Predicted class labels of shape (n_samples,)

        Note:
            - Calls predict_proba internally and applies argmax
            - Updates testing_time attribute via timing decorator
        """

        proba = self.predict_proba(x_test)
        return jnp.argmax(proba, axis=1)

    @abstractmethod
    def predict_proba(self, x_test) -> jnp.ndarray:
        """
        Compute class probability estimates for input samples.

        Args:
            x_test (jnp.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            jnp.ndarray: Probability estimates of shape (n_samples, n_classes)

        Note:
            - Should return values after softmax transformation
            - Updates testing_time attribute via timing decorator
        """

        pass

    @abstractmethod
    def fit(self, X_train, y_train, load=False):
        """
        Train classifier model.

        Args:
            X_train (jnp.ndarray): Training data of shape (n_samples, n_features)
            y_train (jnp.ndarray): Target values of shape (n_samples,)
            load (bool, optional): If True, skip training (for loading pretrained models)

        Note:
            - Updates training_time attribute via timing decorator
        """

        pass

    def get_params(self) -> dict:
        """
        Retrieve constructor parameters for reproducibility.

        Returns:
            dict: Dictionary of parameters that fully define the classifier state.
                  Can be used with __init__ to create an equivalent instance.
        """

        return self.params

    def get_training_time(self):
        """
        Retrieve duration of last training operation.

        Returns:
            float: Training time in seconds. Returns -1 if model hasn't been trained.
        """

        return self.training_time

    def get_testing_time(self):
        """
        Retrieve duration of last inference operation.

        Returns:
            float: Testing time in seconds. Returns -1 if no inference performed.
        """

        return self.testing_time
