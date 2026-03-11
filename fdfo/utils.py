# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

import os
import re
import json
import hashlib
import socket
import torch
import torch.distributed
import numpy as np

_sync_device = None

def init():
    global _sync_device

    if not torch.distributed.is_initialized():
        # Setup some reasonable defaults for env-based distributed init if
        # not set by the running environment.
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            os.environ['MASTER_PORT'] = str(s.getsockname()[1])
            s.close()
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = '1'
        backend = 'gloo' if os.name == 'nt' else 'nccl'
        init_method = 'env://?use_libuv=False' if os.name == 'nt' else 'env://'
        torch.distributed.init_process_group(backend=backend, init_method=init_method)
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    _sync_device = torch.device('cuda') if get_world_size() > 1 else None

def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get('RANK', '0'))

def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get('WORLD_SIZE', '1'))

def barrier():
    if get_world_size() > 1:
        torch.distributed.barrier()

def deinit():
    barrier()
    if torch.distributed.is_initialized() and hasattr(torch.distributed, 'destroy_process_group'):
        torch.distributed.destroy_process_group()

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


class VRAMProfiler:
    """GPU memory profiler that logs checkpoints to a JSONL file."""

    def __init__(self, log_path: str, device=None):
        self.device = device or torch.device(f'cuda:{torch.cuda.current_device()}')
        self.log_path = log_path
        torch.cuda.reset_peak_memory_stats(self.device)

    def _gb(self, nbytes):
        return round(nbytes / (1 << 30), 3)

    def checkpoint(self, label: str):
        torch.cuda.synchronize(self.device)
        entry = dict(
            label=label,
            allocated_gb=self._gb(torch.cuda.memory_allocated(self.device)),
            peak_allocated_gb=self._gb(torch.cuda.max_memory_allocated(self.device)),
            peak_reserved_gb=self._gb(torch.cuda.max_memory_reserved(self.device)),
        )
        if get_rank() == 0:
            import json
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')

    def reset_peak(self):
        torch.cuda.reset_peak_memory_stats(self.device)


class CheckpointIO:
    def __init__(self, **kwargs):
        self._state_objs = kwargs

    def save(self, pt_path, verbose=True):
        if verbose:
            print0(f'Saving {pt_path} ... ', end='', flush=True)
        data = dict()
        for name, obj in self._state_objs.items():
            if obj is None:
                data[name] = None
            elif isinstance(obj, dict):
                data[name] = obj
            elif hasattr(obj, 'state_dict'):
                data[name] = obj.state_dict()
            elif hasattr(obj, '__getstate__'):
                data[name] = obj.__getstate__()
            elif hasattr(obj, '__dict__'):
                data[name] = obj.__dict__
            else:
                raise ValueError(f'Invalid state object of type {type(obj).__name__}')
        if get_rank() == 0:
            torch.save(data, pt_path)
        if verbose:
            print0('done')

    def load(self, pt_path, verbose=True):
        if verbose:
            print0(f'Loading {pt_path} ... ', end='', flush=True)
        data = torch.load(pt_path, map_location=torch.device('cpu'))
        for name, obj in self._state_objs.items():
            if obj is None:
                pass
            elif isinstance(obj, dict):
                obj.clear()
                obj.update(data[name])
            elif hasattr(obj, 'load_state_dict'):
                obj.load_state_dict(data[name])
            elif hasattr(obj, '__setstate__'):
                obj.__setstate__(data[name])
            elif hasattr(obj, '__dict__'):
                obj.__dict__.clear()
                obj.__dict__.update(data[name])
            else:
                raise ValueError(f'Invalid state object of type {type(obj).__name__}')
        if verbose:
            print0('done')

    def load_latest(self, run_dir, pattern=r'training-state-(\d+).pt', verbose=True):
        fnames = [entry.name for entry in os.scandir(run_dir) if entry.is_file() and re.fullmatch(pattern, entry.name)]
        if len(fnames) == 0:
            return None
        pt_path = os.path.join(run_dir, max(fnames, key=lambda x: float(re.fullmatch(pattern, x).group(1))))
        self.load(pt_path, verbose=verbose)
        return pt_path


def select_random_seed(*args):
    binary = json.dumps(args).encode('utf-8')
    digest = hashlib.md5(binary).hexdigest()[:8]
    return int(digest, 16) % (1 << 31)

def set_random_seed(*args):
    seed = select_random_seed(*args)
    torch.manual_seed(seed)
    np.random.seed(seed)
