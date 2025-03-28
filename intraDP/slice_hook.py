import time
from typing import List, Dict, Callable, Tuple, Any
from functools import partial

import numpy as np
import torch
from .hook_tensor import OffloadProfile, TorchOPProfile
from ._utils import iterate_tensor, log_dur, iterate_all_close
from .bm_socket import BiDirectionalSocket
from torch.overrides import TorchFunctionMode


class SliceModule(torch.nn.Module):
    mode = None
    sock = None
    log = print
    communicated_bytes = 0
    def __init__(self, profile: TorchOPProfile, send_slice: List[slice], keep_slice: List[slice], pre: bool=False):
        super().__init__()
        self.send_slice = send_slice
        self.keep_slice = keep_slice
        self.profile = profile
        self.pre = pre
        if pre:
            self.forward = self.pre_forward
        else:
            self.forward = self.post_forward
        self.info = str(self)

    def __repr__(self):
        return f"{'pre' if self.pre else 'post'}{self.__class__.__name__} for {self.profile.idx}:{self.profile.func_name}" + \
            (f"; send_slice: {self.send_slice}" if self.send_slice else "") + \
            (f"; keep_slice: {self.keep_slice}" if self.keep_slice else "")

    def _forward(self, x):
        raise NotImplementedError

    def post_forward(self, *args, **kwargs):
        num_args = len(self.keep_slice)
        x = self._forward(*args[:num_args])
        if kwargs:
            return (*x, *args[num_args:]), kwargs
        else:
            return (*x, *args[num_args:]) if len(args) > 1 else x[0]

    def pre_forward(self, *args, **kwargs):
        num_args = len(self.keep_slice)
        x = self._forward(*args[:num_args])
        return (*x, *args[num_args:]), kwargs

    @classmethod
    def set_mode(cls, mode: str, sock: BiDirectionalSocket, log=print):
        assert mode in ["client", "server"], f"Invalid mode {mode}"
        cls.mode = mode
        cls.sock = sock
        cls.log = log
        cls.communicated_bytes = 0

class SliceSend(SliceModule):
    def __init__(self, profile: TorchOPProfile, send_slice: List[slice], keep_slice: List[slice], pre: bool=False):
        super().__init__(profile, send_slice, keep_slice, pre)

    def _forward(self, *args):
        ret = []
        send = []
        for x, send_slice, keep_slice in zip(args, self.send_slice, self.keep_slice):
            if send_slice == ... or not isinstance(x, torch.Tensor):
                send.append(x)
            else:
                send.append(x[send_slice].clone().detach())
            if keep_slice == ... or not isinstance(x, torch.Tensor):
                ret.append(x)
            else:
                ret.append(x[keep_slice].clone().detach())
        len_bytes = self.sock.send_message(send)
        SliceModule.communicated_bytes += len_bytes
        return ret

class SliceRecv(SliceModule):
    def __init__(self, profile: TorchOPProfile, keep_slice: List[slice], empty: bool=False, pre: bool=False):
        super().__init__(profile, None, keep_slice, pre)
        self.empty = empty

    def _forward(self, *args):
        recvd, len_bytes = self.sock.receive_message_len()
        SliceModule.communicated_bytes += len_bytes
        local_dim = self.profile.local_dim
        if local_dim is None:
            local_dim = -1
        if self.empty or not isinstance(recvd[0], torch.Tensor):
            return recvd
        else:
            if self.mode == "client":
                ret = []
                for x, recvd_tensor, keep_slice in zip(args, recvd, self.keep_slice):
                    if keep_slice == ...:
                        ret.append(torch.cat([x, recvd_tensor], dim=local_dim))
                    else:
                        ret.append(torch.cat([x[keep_slice], recvd_tensor], dim=local_dim))
                return ret
            else:
                ret = []
                for x, recvd_tensor, keep_slice in zip(args, recvd, self.keep_slice):
                    if keep_slice == ...:
                        ret.append(torch.cat([recvd_tensor, x], dim=local_dim))
                    else:
                        ret.append(torch.cat([recvd_tensor, x[keep_slice]], dim=local_dim))
                return ret


class SliceRecover(SliceModule):
    def __init__(self, profile: TorchOPProfile, pre: bool=False):
        super().__init__(profile, None, [...], pre) # We are handling only one input case
        self.slices = None

    def _forward(self, *args):
        profile = self.profile
        ret = []
        all_slices = []
        for x, input_shape in zip(args, profile.input_shapes):
            _ret = torch.empty(input_shape, dtype=x.dtype, device=x.device)
            slices = [slice(None)] * len(_ret.shape)
            if self.mode == "client":
                slices[profile.local_dim] = slice(0, x.shape[profile.local_dim])
            else:
                slices[profile.local_dim] = slice(-x.shape[profile.local_dim], None)
            _ret[tuple(slices)] = x
            ret.append(_ret)
            all_slices.append(tuple(slices))
        self.slices = all_slices
        return ret

class SliceSlice(SliceModule):
    def __init__(self, profile: TorchOPProfile, recover_module: SliceRecover, pre: bool=False):
        super().__init__(profile, None, [...], pre)
        self.recover_module = recover_module

    def _forward(self, *args):
        ret = []
        for x, _slice in zip(args, self.recover_module.slices):
            ret.append(x[_slice])
        return ret

class AlignShape(SliceModule):
    def __init__(self, profile: TorchOPProfile, num_inputs: int, pre: bool=False):
        super().__init__(profile, None, [...] * num_inputs, pre)
        self.recover_module = SliceRecover(profile, pre=True)
        self.local_dim = profile.local_dim
        self.num_inputs = num_inputs

    def _forward(self, *args):
        ret = []
        dim_lens = [x.shape[self.local_dim] for x in args[0]]
        smallest_dim_len = min(dim_lens)
        if all(smallest_dim_len == _dim_len for _dim_len in dim_lens):
            return args
        if self.mode == "client":
            slices = [slice(None)] * self.local_dim + [slice(0, smallest_dim_len)]
        else:
            slices = [slice(None)] * self.local_dim + [slice(-smallest_dim_len, None)]
        for x in args[0]:
            ret.append(x[tuple(slices)])
        return [ret]

class CustomIdentity(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return args, kwargs

def profile_ops_time(forward_func: Callable,
                     args: List[torch.Tensor],
                     kwargs: Dict[str, Any],
                     warmup_runs: int = 5,
                     measure_runs: int = 10,
                     profile_result: OffloadProfile = None,
                     log=print) -> OffloadProfile:
    events = {}  # Store {operator_name: [execution_times]}
    num_valid_profiles = len(profile_result.profile) - 2

    class ProfileTimeMode(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.idx = 1    # Skip the first and last profile
            self.end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_valid_profiles)]

        def __torch_function__(self, func, types, args=(), kwargs=None):
            idx = self.idx
            self.idx += 1

            end_event = self.end_events[idx-1]

            # Execute the actual function
            ret = func(*args, **(kwargs or {}))

            # Stop timing
            end_event.record()
            if idx not in events:
                events[idx] = []

            # Record start & end events for later processing
            events[idx].append(end_event)
            return ret

    # **🔥 Step 1: WARMUP RUNS to stabilize kernel launch times 🔥**
    log(f"\nWarming up for {warmup_runs} runs...")
    for _ in range(warmup_runs):
        with ProfileTimeMode():
            forward_func(args, kwargs)

    # Clear events (only save the measured executions)
    events.clear()
    # **🔥 Step 2: MEASUREMENT RUNS 🔥**
    log(f"\nMeasuring per operator execution time for {measure_runs} runs...")
    for _ in range(measure_runs):
        with ProfileTimeMode():
            forward_func(args, kwargs)

    # **🔥 Step 3: Synchronize once & Process Execution Times 🔥**
    torch.cuda.synchronize()  # Ensure all recorded events are completed

    execution_times = {}
    for op_idx, event_list in events.items():
        total_time = 0.0
        num_samples = len(event_list)

        for i, end_event in enumerate(event_list):
            if op_idx == 1:
                elapsed_time = 0.
            else:
                elapsed_time = events[op_idx-1][i].elapsed_time(end_event)  # Get execution time (ms)
            total_time += elapsed_time

        # Compute the average execution time over runs
        avg_time = total_time / num_samples
        profile_result.profile[op_idx].ops_time = avg_time / 1000
        execution_times[op_idx] = avg_time

    # Compute the end-to-end execution time
    log(f"\nMeasuring end to end computation time for {measure_runs} runs...")
    stime = time.time()
    for _ in range(measure_runs):
        forward_func(args, kwargs)
    torch.cuda.synchronize()
    etime = time.time()
    profile_result.local_comp_time = (etime - stime) / measure_runs

    return profile_result  # Return for external analysis

def hook_model(profile_result: OffloadProfile, mode: str,
               offload_plan: list, recv_plan: list, skip_plan: list, align_shape_plan: list, recv_first_plan: list,
               send_slice_plan: list, send_keep_slice_plan: list, recv_keep_slice_plan: list,
                align_cat=False, align_cat_indices=[]):
    hook_funcs = {}
    end_idx = len(skip_plan) - 1
    empty_num = 0
    for idx, (skip, offload, recv, align_shape) in enumerate(zip(skip_plan, offload_plan, recv_plan, align_shape_plan)):
        pre_hooks = []
        post_hooks = []
        empty = False
        profile = profile_result.profile[idx]
        output_to = profile.output_to
        if end_idx in output_to and (offload or recv):
            _, _post_hook, _ = hook_funcs.get(end_idx, [None, None, True])
            if _post_hook is None:
                if recv:
                    post_hooks.append(SliceRecv(profile_result.profile[end_idx],
                        recv_keep_slice_plan[idx], empty=skip, pre=False))
                else:
                    post_hooks.append(SliceSend(profile_result.profile[end_idx],
                        send_slice_plan[idx], send_keep_slice_plan[idx], pre=False))
            else:
                pre_keep_slice = _post_hook.keep_slice
                if recv:
                    pre_keep_slice.extend(recv_keep_slice_plan[idx])
                    post_hooks.append(SliceRecv(profile_result.profile[end_idx],
                        pre_keep_slice, empty=skip, pre=False))
                else:
                    pre_send_slice = _post_hook.send_slice
                    pre_send_slice.extend(send_slice_plan[idx])
                    post_hooks.append(SliceSend(profile_result.profile[end_idx],
                        pre_send_slice, pre_keep_slice,  pre=False))
            idx = end_idx
        elif idx == 0:
            empty = True
            if offload:
                pre_hooks.append(SliceSend(profile, send_slice_plan[idx], send_keep_slice_plan[idx], pre=True))
            elif recv:
                pre_hooks.append(SliceRecv(profile, recv_keep_slice_plan[idx], empty=True, pre=True))
        elif idx == end_idx:
            if idx in hook_funcs:
                pass
            else:
                empty = True
                if offload:
                    post_hooks.append(SliceSend(profile, send_slice_plan[idx], send_keep_slice_plan[idx], pre=False))
                elif recv:
                    post_hooks.append(SliceRecv(profile, recv_keep_slice_plan[idx], empty=skip==True, pre=False))
        elif not skip and align_cat and idx in align_cat_indices:
            pre_hooks.append(AlignShape(profile, num_inputs=1, pre=True))
        else:
            if skip or profile.excluded:
                empty = True
                if offload:
                    pre_hooks.append(SliceSend(profile, send_slice_plan[idx], send_keep_slice_plan[idx], pre=True))
            if align_shape:
                pre_hooks.append(SliceRecover(profile, pre=True))
                post_hooks.append(SliceSlice(profile, pre_hooks[-1]))
            if recv:
                post_hooks.append(SliceRecv(profile, recv_keep_slice_plan[idx], empty=empty, pre=False))
            elif offload and not skip:
                post_hooks.append(SliceSend(profile, send_slice_plan[idx], send_keep_slice_plan[idx], pre=False))
        if pre_hooks or post_hooks or empty:
            hook_funcs[idx] = (torch.nn.Sequential(*pre_hooks) if len(pre_hooks) > 1 else (pre_hooks[0] if pre_hooks else None),
                               torch.nn.Sequential(*post_hooks) if len(post_hooks) > 1 else (post_hooks[0] if post_hooks else None),
                               empty)
    return hook_funcs

def slice_forward(args: List[torch.Tensor], kwargs: Dict[str, Any], 
                  hook_funcs: Dict[int, Tuple[Callable, Callable, bool]],
                  forward_func: Callable,
                  profile_result: OffloadProfile, mode: str, sock: BiDirectionalSocket, log=print):
    profiles = profile_result.profile
    SliceModule.set_mode(mode, sock, log)
    class SliceMode(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.idx = 1
            self.hook_funcs = hook_funcs

        def __torch_function__(self, func, types, args=(), kwargs=None):
            idx = self.idx
            self.idx += 1
            if kwargs is None:
                kwargs = {}
            if (hooks := self.hook_funcs.get(idx, None)) is not None:
                pre_hook, post_hook, empty = hooks
                args, kwargs = pre_hook(*args, **kwargs) if pre_hook is not None else (args, kwargs)
                if empty:
                    ret = profiles[idx].store_ret # TODO
                else:
                    ret = func(*args, **kwargs)
                ret = post_hook(ret) if post_hook is not None else ret
            else:
                ret = func(*args, **kwargs)
            return ret
    start_pre_hook, _, _ = hook_funcs.get(0, [None, None, False])
    args, kwargs = start_pre_hook(*args, **kwargs) if start_pre_hook is not None else (args, kwargs)
    stime = time.time()
    with SliceMode():
        ret = forward_func(*args, **kwargs)
    torch.cuda.synchronize()
    inner_forward_dur = time.time() - stime
    _, end_post_hook, _ = hook_funcs.get(profile_result.end_idx, [None, None, False])
    ret = end_post_hook(ret) if end_post_hook is not None else ret
    return ret, SliceModule.communicated_bytes, inner_forward_dur
