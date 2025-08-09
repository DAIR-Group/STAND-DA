from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="stand_da",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'mpmath==1.3.0',
        'numpy==2.1.0',
        'scipy==1.15.3',
        'torch',
        'numba'
    ],
    long_description=long_description,
    long_description_content_type="text/markdown"
)