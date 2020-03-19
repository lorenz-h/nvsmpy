# nvsmpy

This package parses information from nvidia-smi into python. You can use it to display usage information about your GPUs, or to select an unoccupied GPU to run your experiments on. It can set the CUDA_VISIBLE_DEVICES environment variable to limit your access to a single or multiple available GPUs. A RuntimeError will be raised if all GPUs are busy.
## Usage
```python
import os
from nvsmpy import CudaCluster

cluster = CudaCluster()
print(cluster)
with cluster.limit_visible_devices():
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # your code goes here
```