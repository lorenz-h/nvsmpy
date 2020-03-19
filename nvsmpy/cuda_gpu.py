import getpass

import psutil


def parse_gpu_uuid(uuid_str: str) -> int:
    # removes the dashes from the string and the leading GPU
    return int(uuid_str.replace("-", "")[3:], 16)


class CudaGPU:
    def __init__(self, index, uuid: str, name: str):
        self.index: int = int(index)
        self.uuid: int = parse_gpu_uuid(uuid)
        self.name: str = name
        self.visible: bool = True

        self.user = getpass.getuser()
        self.processes = []

    def __str__(self):

        procs_str = "\n".join([f"User:{proc.username()}, PID: {proc.pid}"for proc in self.processes])

        return f"<{self.name}> <{self.index}> <{self.uuid}> <{self.is_available()}>\n" \
               f"<running processes>:\n {procs_str}"

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
