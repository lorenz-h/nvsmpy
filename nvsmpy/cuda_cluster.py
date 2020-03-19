import subprocess
import csv
import os
import logging
from io import StringIO
from typing import List, Tuple, Dict

import psutil

from .cuda_gpu import CudaGPU, parse_gpu_uuid


logger = logging.getLogger(__file__)


class CudaCluster:
    def __init__(self):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        self.n_visible_devices = None
        self.max_n_processes = None

        self.devices = self.get_devices()
        self.update_device_occupation()

    def __getitem__(self, item):
        return self.devices[item]

    def get_devices(self):
        fields = ("index", "uuid", "name")
        dev_info_cmd = ["nvidia-smi", "--format=csv", f"--query-gpu={','.join(fields)}"]
        device_infos = self.parse_smi_command(dev_info_cmd, fields=fields)
        logger.debug(f"Parsed device info from nvidia-smi: {device_infos}")
        return {parse_gpu_uuid(device_info["uuid"]): CudaGPU(**device_info) for device_info in device_infos}

    def update_device_state(self):
        # load new information from query-gpu
        pass

    def update_device_occupation(self):
        for device in self.devices.values():
            device.reset_processes()
        apps_info = self.get_compute_apps_information()
        logger.debug(f"Parsed compute apps info from nvidia-smi: {apps_info}")
        for app_info in apps_info:
            uuid = parse_gpu_uuid(app_info["gpu_uuid"])
            proc = psutil.Process(int(app_info["pid"]))
            self.devices[uuid].add_process(proc)

    def limit_visible_devices(self, n_devices=1, max_n_processes=0):
        self.max_n_processes = max_n_processes
        self.n_visible_devices = n_devices
        return self

    def get_compute_apps_information(self) -> List[Dict]:
        fields = ("pid", "process_name", "gpu_uuid")
        proc_info_cmd = ["nvidia-smi", "--format=csv", f"--query-compute-apps={','.join(fields)}"]
        return self.parse_smi_command(proc_info_cmd, fields=fields)

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
        self.update_device_occupation()
        devices_str = "\n".join([str(device) for device in self.devices.values()])
        return f"Cluster:\n {devices_str}"

    def __enter__(self):
        self.update_device_occupation()
        if self.n_visible_devices is None:
            raise AttributeError("Could not enter Cluster object directly. Please use with "
                                 "cluster.limit_visible_devices() instead.")

        available_devices = [str(device.index) for device in
                             self.devices.values() if device.is_available(self.max_n_processes)]
        if len(available_devices) >= self.n_visible_devices:
            # set CUDA_VISIBLE_DEVICES to <self.n_visible_devices> available devices.
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available_devices[:self.n_visible_devices])
            logger.warning(f"Set visible devices: {available_devices[:self.n_visible_devices]}")
        else:
            raise RuntimeError(f"Could not find {self.n_visible_devices} available devices.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.n_visible_devices = None
        # reset CUDA_VISIBLE_DEVICES to be all devices on the system:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(device.index) for device in self.devices.values()])
