from setuptools import setup, find_packages

setup(
    name="variational_partitioning",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["variational_partitioning"],
    install_requires=[
        "Click",
    ],
    entry_points={
        "console_scripts": [
            "variational_partitioning = variational_partitioning:main",
        ],
    },
)
