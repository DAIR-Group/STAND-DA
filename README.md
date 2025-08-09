# Statistical Inference for Autoencoder-based Anomaly Detection after Representation Learning-based Domain Adaptation

[![PyPI version](https://badge.fury.io/py/stand-da.svg)](https://badge.fury.io/py/stand-da)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package offers a Statistical Inference (SI) framework tailored for Autoencoder-based anomaly detection following representation learning–driven domain adaptation. It utilizes a divide-and-conquer approach, enhanced by GPU-accelerated parallel processing, to efficiently compute the $p$-values of identified anomalies. The proposed methods deliver dependable detection by keeping the false positive rate (FPR) under control, while also maximizing the true positive rate (TPR) and effectively lowering the false negative rate (FNR).

## Requirements
This package has the following requirements:

    numpy
    mpmath
    scipy
    torch
    numba

## Installation

### Prerequisites
This package requires a compatible NVIDIA GPU and the NVIDIA CUDA Toolkit to be installed on your system. Your GPU driver must also be up-to-date.

This package was built and tested with CUDA Toolkit 11.8. We recommend you install this version to ensure full compatibility. You can find the downloads on the [NVIDIA Developer website](https://developer.nvidia.com/cuda-toolkit-archive).

### Package Installation
This package can be installed using pip:
```bash
$ pip install stand_da
```

## Usage

We provide several Jupyter notebooks demonstrating how to use the stand-da package in action.

- Example for computing $p$-value for Autoencoder-based AD after Representation Learning-based DA
```
>> ex0_compute_p_value.ipynb
```
- Check the uniformity of the pivot
```
>> ex2_validity_of_p_value.ipynb
```