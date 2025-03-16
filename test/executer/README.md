## Executer 是一个基类，用于管理机器学习模型的训练、测试和日志记录过程。其子类包括：

- NonValidExecuter: 不进行交叉验证的执行器。
- KFlodCrossExecuter: 使用 K 折交叉验证的执行器。
- LeaveOneCrossExecuter: 使用留一法交叉验证的执行器。
- BootstrapExecuter: 使用 Bootstrap 方法的执行器。

测试脚本通过模拟数据和简单的分类器，验证这些执行器的功能是否正常工作。
