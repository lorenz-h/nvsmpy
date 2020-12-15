import getpass

import psutil
from pynvml import nvmlDeviceGetHandleByIndex


def parse_gpu_uuid(uuid_str: str) -> int:
    # removes the dashes from the string and the leading GPU
    return int(uuid_str.replace("-", "")[3:], 16)


def parse_device_index(minor_number: str) -> int:
    if minor_number == "N/A":
        #  if there is only one GPU in the system nvidia-smi returns minor_number N/A
        return 0
    else:
        try:
            return int(minor_number)
        except ValueError:
            raise ValueError(f"Could not parse device index. "
                             f"Got unexpected minor number {minor_number} from nvidia-smi.")


class CudaGPU:
    def __init__(self, index: int, uuid: int, name: str):
        assert isinstance(index, int), f"GPU index must be integer."
        assert isinstance(uuid, int), f"GPU uuid must be hexadecimal integer. See parse_gpu_uuid."

        self.index: int = index
        self.uuid: int = uuid
        self.name: str = name
        self.handle = nvmlDeviceGetHandleByIndex(self.index)

        self.user = getpass.getuser()
        self.processes = []

    def update(self, minor_number: str, uuid: str, product_name: str, **kwargs) -> None:
        self.index = parse_device_index(minor_number)
        self.uuid = parse_gpu_uuid(uuid)
        self.name = product_name

    def __str__(self):

        procs_str = " - ".join([f"(user:{proc.username()}, pid: {proc.pid})"for proc in self.processes])

        return f"name: {self.name}, index: {self.index}, available: {self.is_available()}\n" \
               f"processes ({len(self.processes)}):{procs_str}"

    def __hash__(self):
        return self.uuid

    def reset_processes(self) -> None:
        self.processes = []

    def add_process(self, proc: psutil.Process) -> None:
        self.processes.append(proc)

    def is_available(self, max_n_processes: int = 1):

        if any([proc.username != self.user for proc in self.processes]):
            # if any of the processes on this GPU is owned by another user mark GPU as unavailable
            return False
        else:
            # if more than max_n_processes are already running on the GPU mark the GPU as unavailable.
            return len(self.processes) < max_n_processes
