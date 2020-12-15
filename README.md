# nvsmpy

This package parses information from nvidia-smi into python. You can use it to display usage information about your GPUs, or to select an unoccupied GPU to run your experiments on. It can set the CUDA_VISIBLE_DEVICES environment variable to limit your access to a single or multiple available GPUs. A RuntimeError will be raised if all GPUs are busy.

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
