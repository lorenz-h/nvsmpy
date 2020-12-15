import os
import logging
import time
from typing import List, Dict, Optional, Sequence

import psutil
from pynvml import nvmlDeviceGetComputeRunningProcesses
from pynvml.smi import nvidia_smi

from .cuda_gpu import CudaGPU, parse_gpu_uuid, parse_device_index


class ClusterModeError(RuntimeError):
    def __init__(self, current_mode: Optional[str]):
        msg: str = f"Cuda Cluster device filtering was in mode {current_mode}." \
                   f"You may only use either cuda_cluster.visible_devices() or cuda_cluster.available_devices()"
        super().__init__(msg)


class CudaCluster:
    def __init__(self):
        self.logger = logging.getLogger(__file__)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self.n_visible_devices = None
        self.visible_device_indices: Optional[List[int]] = None
        self.max_n_processes = None
        self.last_update: float = time.time()
        self.min_update_interval: int = 5
        self.mode: Optional[str] = None
        self.devices = self.create_devices()
        self.update_device_occupation()

    def __getitem__(self, item):
        return [device for device in self.devices.values() if device.index == item][0]

    def get_gpu_infos(self):
        nvsmi = nvidia_smi.getInstance()
        gpu_infos = nvsmi.DeviceQuery("index, uuid, name")
        self.logger.debug(f"Got device info from nvidia-smi: {gpu_infos}")
        return gpu_infos

    def create_devices(self):
        gpu_infos = self.get_gpu_infos()

        devices = {}

        for info in gpu_infos["gpu"]:
            gpu_idx: int = parse_device_index(info["minor_number"])
            gpu_uuid: int = parse_gpu_uuid(info["uuid"])
            devices[gpu_uuid] = CudaGPU(gpu_idx, gpu_uuid, info["product_name"])

        return devices

    def update(self, force: bool = True):
        if time.time() - self.last_update > self.min_update_interval or force:
            gpu_infos = self.get_gpu_infos()
            for gpu_info in gpu_infos:
                uuid = parse_gpu_uuid(gpu_info["uuid"])
                self.devices[uuid].update(**gpu_info)
            self.update_device_occupation()
            self.last_update = time.time()
        else:
            self.logger.debug(f"will not update states because last state "
                              f"update was less than {self.min_update_interval} seconds old.")

    def update_device_occupation(self):
        for device in self.devices.values():
            device.reset_processes()
            procs = nvmlDeviceGetComputeRunningProcesses(device.handle)
            for p in procs:
                proc = psutil.Process(p.pid)
                device.add_process(proc)

    def available_devices(self, n_devices=1, max_n_processes=1):
        self.max_n_processes = max_n_processes
        self.n_visible_devices = n_devices
        if self.mode is not None:
            raise ClusterModeError(self.mode)
        self.mode = "available_devices"
        return self

    def visible_devices(self, *device_ids: Sequence[int]):
        if self.mode is not None:
            raise ClusterModeError(self.mode)
        self.mode = "visible_devices"
        self.n_visible_devices = len(device_ids)
        self.visible_device_indices = list(device_ids)
        return self

    @staticmethod
    def query_gpu(*fields) -> List[Dict]:
        nvsmi = nvidia_smi.getInstance()
        gpu_infos = nvsmi.DeviceQuery(','.join(fields))
        return gpu_infos["gpu"]

    def __str__(self):
        self.update(force=False)
        devices_str = "\n".join([str(device) for device in self.devices.values()])
        line = "-------------------------------------"
        return f"{line}\nCuda Cluster:\n{devices_str}\n{line}"

    def __enter__(self):
        if self.mode is None or self.n_visible_devices is None:
            raise ClusterModeError(self.mode)
        elif self.mode == "available_devices":
            self.update(force=False)
            available_devices: List[int] = [device.index for device in
                                            self.devices.values() if device.is_available(self.max_n_processes)]
            if len(available_devices) >= self.n_visible_devices:
                # set CUDA_VISIBLE_DEVICES to <self.n_visible_devices> available devices.
                self.visible_device_indices = available_devices[:self.n_visible_devices]
                self._set_visible_devices()
            else:
                raise RuntimeError(f"Could not find {self.n_visible_devices} available devices.")

        elif self.mode == "visible_devices":
            self._set_visible_devices()
        else:
            raise ValueError(f"Unexpected mode {self.mode}")

    def _set_visible_devices(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in self.visible_device_indices])
        self.logger.warning(f"Set visible devices to: {[f'gpu:{idx}' for idx in self.visible_device_indices]}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.n_visible_devices = None
        self.visible_device_indices = None
        self.mode = None
        # reset CUDA_VISIBLE_DEVICES to be all devices on the system:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(device.index) for device in self.devices.values()])
