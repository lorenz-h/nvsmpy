import getpass

import psutil


def parse_gpu_uuid(uuid_str: str) -> int:
    # removes the dashes from the string and the leading GPU
    return int(uuid_str.replace("-", "")[3:], 16)


class CudaGPU:
    def __init__(self, index: str, uuid: str, name: str):
        self.index: int = int(index)
        self.uuid: int = parse_gpu_uuid(uuid)
        self.name: str = name

        self.user = getpass.getuser()
        self.processes = []

    def update(self, index: str, uuid: str, name: str) -> None:
        self.index = int(index)
        self.uuid = parse_gpu_uuid(uuid)
        self.name = name

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
