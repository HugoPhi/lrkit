"""
Metrics Evaluation Module

This module provides the `Metrics` class to compute and store various classification performance metrics.
The `Metrics` class computes key evaluation metrics like Precision, Recall, F1 Score, Accuracy, and ROC/AUC.
It supports both binary and multi-class classification problems, with additional support for probability-based predictions
when classification probabilities are available.

Key Features:
-------------
- Precision, Recall, F1 Score: For each class, computes these fundamental classification metrics.
- Accuracy: Computes overall accuracy as the ratio of correct predictions to total predictions.
- ROC Curve & AUC: Calculates the Receiver Operating Characteristic curve and Area Under the Curve (AUC) for each class.
- Average Precision (AP): Computes the average precision score using the one-vs-rest approach.
- Confusion Matrix: Provides the confusion matrix for the classification problem, showing true positives, false positives, etc.

Usage:
------
1. Basic Example:
    metrics = Metrics(y_true, y_pred)
    print("Precision per class:", metrics.precision())
    print("Recall per class:", metrics.recall())
    print("F1 Score per class:", metrics.f1())
    print("Overall Accuracy:", metrics.accuracy())

2. Using ROC and AUC:
    if metrics.proba:
        print("ROC:", metrics.roc())
        print("AUC:", metrics.auc())

3. Using Average Precision:
    if metrics.proba:
        print("Average Precision:", metrics.avg_ap())

Attributes:
-----------
- y: True class labels for the dataset.
- y_pred: Predicted class labels or probabilities.
- classes: Number of classes in the classification task.
- matrix: Confusion matrix showing the true vs predicted label counts.
- proba: Boolean indicating if `y_pred` contains probabilities or hard class labels.
"""

import jax.numpy as jnp


class Metrics:
    '''
    Classification Model Evaluation Metrics.

    This class computes various evaluation metrics to assess the performance of classification models.
    It includes common metrics such as Precision, Recall, F1 Score, Accuracy, ROC, AUC, and Average Precision.
    The class supports both single-label and multi-class classification problems, with options for
    handling probability-based predictions.

    Key Metrics:
    -------------
    - Precision: Measures the ratio of true positives to predicted positives for each class.
    - Recall: Measures the ratio of true positives to actual positives for each class.
    - F1 Score: The harmonic mean of Precision and Recall.
    - Accuracy: The proportion of correctly classified samples.
    - ROC and AUC: Evaluates classifier performance in distinguishing between classes based on true positive and false positive rates.
    - Average Precision (AP): Calculates the average precision across different classification thresholds.

    Attributes:
    -----------
    y : jnp.ndarray
        True class labels.
    y_pred : jnp.ndarray
        Predicted class labels or probabilities.
    classes : int
        Number of classes in the classification problem.
    matrix : jnp.ndarray
        Confusion matrix for true vs predicted labels.
    proba : bool
        Whether the model provides probabilities (True) or hard predictions (False).
    '''

    def __init__(self, y, y_pred, classes=None):
        '''
        Initialize the Metrics class to compute evaluation metrics. y & y_pred should be 1D or 2D & labeled start from '0'.

        Parameters:
        ----------
        y : jnp.ndarray
            True class labels.
        y_pred : jnp.ndarray
            Predicted class labels or probabilities.
        classes : int, optional
            The number of classes. If None, the number of unique labels in `y` will be used.
        '''
        self.y

        self.y = y
        self.y_pred = y_pred

        if classes is not None:
            self.classes = classes
        else:
            uni = jnp.unique(self.y)
            self.classes = uni.shape[0]  # get classes num

        self.matrix = jnp.zeros((self.classes, self.classes))  # get confusion matrix

        if len(self.y.shape) == 1:
            temp_y = self.y
        elif len(self.y.shape) == 2:
            temp_y = jnp.argmax(self.y, axis=1)
        else:
            raise ValueError('Input y must be 1D or 2D.')

        if len(self.y_pred.shape) == 1:
            temp_y_pred = self.y_pred
        if len(self.y_pred.shape) == 2:
            temp_y_pred = jnp.argmax(self.y_pred, axis=1)
        elif len(self.y_pred.shape) > 2:
            raise ValueError('Input y_pred must be 1D or 2D.')

        if len(self.y.shape) == 2 and len(self.y_pred.shape) == 2:
            self.proba = True
        else:
            self.proba = False

        for i, j in zip(temp_y, temp_y_pred):
            self.matrix = self.matrix.at[i, j].set(
                self.matrix[i, j] + 1
            )

    def precision(self):
        '''
        Compute the precision for each class.

        Precision is the ratio of true positives to the total predicted positives.

        Returns
        -------
        numpy.ndarray
            The precision of each class.
        '''
        return jnp.diag(self.matrix) / self.matrix.sum(axis=0)

    def recall(self):
        '''
        Compute the recall for each class.

        Recall is the ratio of true positives to the total actual positives.

        Returns
        -------
        numpy.ndarray
            The recall of each class.
        '''

        return jnp.diag(self.matrix) / self.matrix.sum(axis=1)

    def f1(self):
        '''
        Compute the F1 score for each class.

        The F1 score is the harmonic mean of precision and recall.

        Returns
        -------
        numpy.ndarray
            The F1 score of each class.
        '''

        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def accuracy(self):
        '''
        Compute the overall accuracy.

        Accuracy is the ratio of correctly predicted instances to the total instances.

        Returns
        -------
        float
            The accuracy of the model.
        '''

        return jnp.diag(self.matrix).sum() / self.matrix.sum()

    def roc(self):
        '''
        Compute the ROC curve for each class. Only callable when 'proba == Ture'

        Uses the one-vs-rest ('ovr') approach and returns the AUC for each class.

        Returns
        -------
        list of tuple
            A list where each element is a tuple containing true positive rates (TPR)
            and false positive rates (FPR) for each class.
        '''

        if not self.proba:
            raise ValueError('roc() can only be called when y & y_pred are proba matrix')

        def calculate_tpr_fpr(y_true, y_pred):
            tp = jnp.sum((y_pred == 1) & (y_true == 1))
            tn = jnp.sum((y_pred == 0) & (y_true == 0))
            fp = jnp.sum((y_pred == 1) & (y_true == 0))
            fn = jnp.sum((y_pred == 0) & (y_true == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            return tpr, fpr

        rocs = []
        for class_idx in range(self.classes):
            tprs = []
            fprs = []
            thresholds = self.y_pred[:, class_idx].reshape(-1)
            for threshold in jnp.sort(thresholds)[::-1]:
                idx_pred = (self.y_pred[:, class_idx] >= threshold).astype(int)  # '=' here is important
                idx_true = (self.y == class_idx).astype(int).reshape(-1)
                tpr, fpr = calculate_tpr_fpr(idx_true, idx_pred)
                tprs.append(tpr)
                fprs.append(fpr)

            rocs.append((tprs, fprs))

        return rocs

    def auc(self):
        '''
        Compute the AUC for each class. Only callable when 'proba == Ture'

        Uses the one-vs-rest ('ovr') approach to calculate the AUC for each class.

        Returns
        -------
        numpy.ndarray
            The AUC of each class.
        '''

        if not self.proba:
            raise ValueError('auc() can only be called when y & y_pred are proba matrix')

        rocs = self.roc()
        aucs = []
        for (tprs, fprs) in rocs:
            auc = 0
            for i in range(1, len(fprs)):
                auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2
            aucs.append(auc)

        return jnp.array(aucs)

    def ap(self):
        '''
        Compute the Average Precision (AP) for each class. Only callable when 'proba == Ture'

        Uses the one-vs-rest ('ovr') approach to calculate the AP for each class.

        Returns
        -------
        numpy.ndarray
            The average precision (AP) of each class.
        '''

        if not self.proba:
            raise ValueError('ap() can only be called when y & y_pred are proba matrix')

        def calculate_prec_rec(y_true, y_pred):
            tp = jnp.sum((y_pred == 1) & (y_true == 1))
            fp = jnp.sum((y_pred == 1) & (y_true == 0))
            fn = jnp.sum((y_pred == 0) & (y_true == 1))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (fp + fn) > 0 else 0
            return prec, rec

        aps = []
        for class_idx in range(self.classes):
            precs = []
            recs = []
            thresholds = self.y_pred[:, class_idx].reshape(-1)
            for threshold in jnp.sort(thresholds)[::-1]:
                idx_pred = (self.y_pred[:, class_idx] >= threshold).astype(int)  # '=' here is important
                idx_true = (self.y == class_idx).astype(int).reshape(-1)
                prec, rec = calculate_prec_rec(idx_true, idx_pred)
                precs.append(prec)
                recs.append(rec)

            ap = 0
            for i in range(1, len(recs)):
                ap += (recs[i] - recs[i - 1]) * (precs[i] + precs[i - 1]) / 2
            aps.append(ap)

        return jnp.array(aps)

    def avg_ap(self):
        '''
        Compute the average of average precision (AP) scores. Only callable when 'proba == Ture'

        Returns
        -------
        float
            The mean average precision score across all classes.
        '''

        if not self.proba:
            raise ValueError('avg_ap() can only be called when y & y_pred are proba matrix')

        return self.ap().mean()

    def avg_pre(self):
        '''
        Compute the average precision score.

        Returns
        -------
        float
            The mean precision score across all classes.
        '''

        return self.precision().mean()

    def avg_recall(self):
        '''
        Compute the average recall score.

        Returns
        -------
        float
            The mean recall score across all classes.
        '''

        return self.recall().mean()

    def avg_auc(self):
        '''
        Compute the average AUC score.

        Returns
        -------
        float
            The mean AUC score across all classes.
        '''

        return self.auc().mean()

    def macro_f1(self):
        '''
        Compute the macro F1 score.

        The macro F1 score is the average F1 score across all classes.

        Returns
        -------
        float
            The macro F1 score.
        '''

        return self.f1().mean()

    def micro_f1(self):
        '''
        Compute the micro F1 score.

        The micro F1 score is computed using the global counts of true positives,
        false positives, and false negatives across all classes.

        Returns
        -------
        float
            The micro F1 score.
        '''

        return 2 * self.precision().mean() * self.recall().mean() / (self.precision().mean() + self.recall().mean())

    def confusion_matrix(self):
        '''
        Get the confusion matrix.

        Returns
        -------
        numpy.ndarray
            The confusion matrix.
        '''

        return self.matrix

    def __repr__(self) -> str:
        '''
        Provide a string representation of the performance metrics.

        Returns
        -------
        str
            A formatted string showing precision, recall, F1, accuracy,
            macro average, and micro average.
        '''

        table = ' ' * 6
        print(f'        {table}Precision{table}Recall{table}  F1')
        for i in range(len(self.precision())):
            print(f'Class {i} {table}{self.precision()[i]:.6f} {table}{self.recall()[i]:.6f}{table}{self.f1()[i]:.6f}')
        print()
        print(f'Accuracy      {self.accuracy():.6f}')
        print(f'Macro F1      {self.macro_f1():.6f}')
        print(f'Micro F1      {self.micro_f1():.6f}')

        return ''
