from setuptools import setup, find_packages

setup(
    name='lrkit',
    version='v0.1',
    author='Yunming Hu',
    author_email='hugonelson07@outlook.com',
    url='https://github.com/HugoPhi/lrkit.git',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'toml',
    ],
)
