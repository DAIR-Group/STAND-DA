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

- Example for computing $p$-value for Autoencoder-based
AD after Representation
Learning-based DA
```
>> ex0_compute_p_value.ipynb
```
- Check the uniformity of the pivot
```
>> ex2_validity_of_p_value.ipynb
```