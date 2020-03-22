import subprocess
import csv
import os
import logging
import time
from io import StringIO
from typing import List, Tuple, Dict, Optional

import psutil

from .cuda_gpu import CudaGPU, parse_gpu_uuid


logger = logging.getLogger(__file__)


class CudaCluster:
    def __init__(self):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self.n_visible_devices = None
        self.visible_device_indices: Optional[List[int]] = None
        self.max_n_processes = None
        self.last_update: float = time.time()
        self.min_update_interval: int = 5

        self.devices = self.create_devices()
        self.update_device_occupation()

    def __getitem__(self, item):
        return [device for device in self.devices.values() if device.index == item][0]

    def get_gpu_infos(self):
        fields = ("index", "uuid", "name")
        dev_info_cmd = ["nvidia-smi", "--format=csv", f"--query-gpu={','.join(fields)}"]
        gpu_infos = self.parse_smi_command(dev_info_cmd, fields=fields)
        logger.debug(f"Parsed device info from nvidia-smi: {gpu_infos}")
        return gpu_infos

    def create_devices(self):
        gpu_infos = self.get_gpu_infos()
        return {parse_gpu_uuid(gpu_info["uuid"]): CudaGPU(**gpu_info) for gpu_info in gpu_infos}

    def update(self, force: bool = True):
        if time.time() - self.last_update > self.min_update_interval or force:
            gpu_infos = self.get_gpu_infos()
            for gpu_info in gpu_infos:
                uuid = parse_gpu_uuid(gpu_info["uuid"])
                self.devices[uuid].update(**gpu_info)
            self.update_device_occupation()
            self.last_update = time.time()
        else:
            logger.debug(f"will not update states because last state "
                         f"update was less than {self.min_update_interval} seconds old.")

    def update_device_occupation(self):
        for device in self.devices.values():
            device.reset_processes()
        apps_info = self.get_compute_apps_information()
        for app_info in apps_info:
            uuid = parse_gpu_uuid(app_info["gpu_uuid"])
            proc = psutil.Process(int(app_info["pid"]))
            self.devices[uuid].add_process(proc)

    def limit_visible_devices(self, n_devices=1, max_n_processes=1):
        self.max_n_processes = max_n_processes
        self.n_visible_devices = n_devices
        return self

    def get_compute_apps_information(self) -> List[Dict]:
        fields = ("pid", "process_name", "gpu_uuid")
        proc_info_cmd = ["nvidia-smi", "--format=csv", f"--query-compute-apps={','.join(fields)}"]
        apps_info = self.parse_smi_command(proc_info_cmd, fields=fields)
        logger.debug(f"Parsed compute apps info from nvidia-smi: {apps_info}")
        return apps_info

    def query_gpu(self, *fields) -> List[Dict]:
        dev_info_cmd = ["nvidia-smi", "--format=csv", f"--query-gpu={','.join(fields)}"]
        return self.parse_smi_command(dev_info_cmd, fields=fields)

    @staticmethod
    def parse_smi_command(command: List[str], fields: Tuple) -> List:
        smi_output = subprocess.check_output(command).strip().decode("utf-8")
        f = StringIO(smi_output)
        reader = csv.DictReader(f, fieldnames=fields, delimiter=',', skipinitialspace=True)
        next(reader)  # skips the headers
        return list(reader)

    def __str__(self):
        self.update(force=False)
        devices_str = "\n".join([str(device) for device in self.devices.values()])
        line = "-------------------------------------"
        return f"{line}\nCuda Cluster:\n{devices_str}\n{line}"

    def __enter__(self):
        self.update(force=False)
        if self.n_visible_devices is None:
            raise AttributeError("Could not enter Cluster object directly. Please use with "
                                 "cluster.limit_visible_devices() instead.")

        available_devices = [str(device.index) for device in
                             self.devices.values() if device.is_available(self.max_n_processes)]
        if len(available_devices) >= self.n_visible_devices:
            # set CUDA_VISIBLE_DEVICES to <self.n_visible_devices> available devices.
            self.visible_device_indices = available_devices[:self.n_visible_devices]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.visible_device_indices)
            logger.warning(f"Set visible devices to: "
                           f"{['gpu:'+idx for idx in self.visible_device_indices]}")
        else:
            raise RuntimeError(f"Could not find {self.n_visible_devices} available devices.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.n_visible_devices = None
        self.visible_device_indices = None
        # reset CUDA_VISIBLE_DEVICES to be all devices on the system:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(device.index) for device in self.devices.values()])
