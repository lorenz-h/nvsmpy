# nvsmpy
This package automatically manages your `CUDA_VISIBLE_DEVICES` environment variable to avoid using GPUs that are currently being used by other users on a multi-user, multi-gpu system. A RuntimeError will be raised if all GPUs are busy. If you pass the `max_n_processes` argument to `available_devices()` you may run multiple processes under your system username on a given GPU at the same time.

## Installation
```shell
pip install nvsmpy
```
## Usage
```python
import os
from nvsmpy import CudaCluster

cluster = CudaCluster()
print(cluster)

# To limit access to any two unused GPUs:
with cluster.available_devices(n_devices=2):
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # your code goes here

# Alternatively limit access to GPUs 0 and 7, regardless of availability:
with cluster.visible_devices(0, 3, 7):
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # your code goes here

```
