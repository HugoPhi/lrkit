# Learn Kit

This is a task-oriented test frame work for learning algorithms. We provide two API version: `numpy` & `jax`. You can import them as submodules. For example, if you want to use `jax` version:  

```bash
git submodule add -b jax https://github.com/tibless/lrkit.git ./plugins/lrkit
git submodule sync
```

then, you should make sure there is a `__init__.py` under the `plugins/` folder & write `setup.py` for current project. For example, put this file under your project folder:  

```python
from setuptools import setup, find_packages

setup(
    name='myproject',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'toml',
        'tabulate',
    ],
)
```

then install by pip under Debug: 

```bash
pip install -e .
```

## Dependencies

- numpy
- toml
- tabulate
