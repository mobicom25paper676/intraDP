import time
from typing import List, Dict, Callable, Tuple
import asyncio
from functools import partial

import numpy as np
from functools import partial
from threading import Thread
from queue import Queue
import torch
from .hook_tensor import OffloadProfile, TorchOPProfile, InputSlot
from ._utils import iterate_tensor, log_dur, iterate_all_close
from .test_asyncio import AsyncTCPMessageStream


def fill_slot(arg: torch.Tensor, slots_iter):
    for slot in next(slots_iter):
        slot.container[slot.index] = arg

def fill_slot_cuda(arg: torch.Tensor, slots_iter):
    for slot in next(slots_iter):
        slot.container[slot.index] = arg.cuda()

def fill_output_to_input_slots(tensors, output_dict: Dict[int, List[InputSlot]]):
    slots_iter = iter(output_dict.values())
    def fill_slot(arg: torch.Tensor):
        for slot in next(slots_iter):
            slot.fill(arg)
    iterate_tensor(tensors, fill_slot)

def empty_input_slots(input_slots: List[InputSlot]):
    for slot in input_slots:
        slot.empty()

def slice_output(tensors, send_slice: List[slice], keep_slice: List[slice]):
    send_tensor = []    # Always send a list of tensor
    send_slice_iter = iter(send_slice)
    keep_slice_iter = iter(keep_slice)
    def slice_tensor(arg: torch.Tensor):
        send_tensor.append(arg[next(send_slice_iter)].cpu())
        return arg[next(keep_slice_iter)].contiguous()
    keep_tensor = iterate_tensor(tensors, slice_tensor)
    return keep_tensor, send_tensor

def cat_output(tensors, recv_tensor: List[torch.Tensor], keep_slice: List[slice], order: int=0, dim=-1):
    recv_tensor_iter = iter(recv_tensor)
    keep_slice_iter = iter(keep_slice)
    if order == 0:
        def cat_tensor(arg: torch.Tensor):
            return torch.cat([arg[next(keep_slice_iter)], next(recv_tensor_iter).cuda()], dim)
    else:
        def cat_tensor(arg: torch.Tensor):
            return torch.cat([next(recv_tensor_iter).cuda(), arg[next(keep_slice_iter)]], dim)
    return iterate_tensor(tensors, cat_tensor)

def align_tensor_shapes(obj, local_dim, align_mode=1):
    tensors = []
    iterate_tensor(obj, tensors.append)
    shape_len = len(tensors[0].shape)
    align_dim_len = min(t.shape[local_dim] for t in tensors)
    if align_mode == 1:
        slices = [slice(None)] * local_dim + [slice(align_dim_len)] + [slice(None)] * (shape_len - local_dim - 1)
    else:   # 2 at the server side
        slices = [slice(None)] * local_dim + [slice(-align_dim_len, None)] + [slice(None)] * (shape_len - local_dim - 1)
    def _align_shapes(t: torch.Tensor):
        return t[slices]
    return iterate_tensor(obj, _align_shapes)

def compile_plan_to_static_exec(
    profile_result: OffloadProfile, plans: Dict[int, Dict[str, list]],
    sock: AsyncTCPMessageStream, log=print, sleep_step=10, merge=True):
    async def plain_skip(p: TorchOPProfile):
        for slot in p.input_slots:
            slot.container[slot.index] = None
    async def skip_with_recv(p: TorchOPProfile):
        with log_dur(sock.add_sufSPSO-GA_to_log, preSPSO-GA=f"op {p.idx} {p.func_name} recv"):
            recved_tensor = await sock.queued_recv()
        iterate_tensor(recved_tensor,
                    partial(fill_slot_cuda,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    async def plain_exec(p: TorchOPProfile, align_shape):
        if align_shape:
            args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shape)
        else:
            args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    async def merged_plain_exec(
        ps: List[TorchOPProfile], align_shapes: List[int], sleep_plan: List[bool]):
        with log_dur(sock.add_sufSPSO-GA_to_log, preSPSO-GA=f"op {ps[0].idx}-{ps[-1].idx} merged exec"):
            for p, align_shape, sleep in zip(ps, align_shapes, sleep_plan):
                if align_shape:
                    args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shapes)
                else:
                    args, kwargs = p.func_args
                intermediates: torch.Tensor = p.func(*args, **kwargs)
                iterate_tensor(intermediates,
                    partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
                for slot in p.input_slots:
                    slot.container[slot.index] = None
                if p.idx % 10 == 0:
                    await asyncio.sleep(0.)
            torch.cuda.synchronize()
    async def exec_with_recv(p: TorchOPProfile, keep_slice, cat_order, cat_dim, align_shape: int):
        if align_shape:
            args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shape)
        else:
            args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(sock.add_sufSPSO-GA_to_log, preSPSO-GA=f"op {p.idx} {p.func_name} recv"):
            recv_intermediates = await sock.queued_recv()
            intermediates = cat_output(
                intermediates, recv_intermediates,
                keep_slice=keep_slice,
                order=cat_order, dim=cat_dim)
        iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    async def exec_with_offload(p: TorchOPProfile, send_slice, keep_slice, align_shape: List[int]):
        if align_shape:
            args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shape)
        else:
            args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(sock.add_sufSPSO-GA_to_log, preSPSO-GA=f"op {p.idx} {p.func_name} send"):
            intermediates, send_intermediates = slice_output(
                intermediates, send_slice=send_slice, keep_slice=keep_slice)
            await sock.queued_send(send_intermediates)
        iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def _plan_to_static_exec(skip: list, offload: list, recv: list,
                send_slice: dict, send_keep_slice: dict, recv_keep_slice: dict,
                cat_dim: dict, cat_order: dict, recv_first: dict,
                align_shape: list, **kwargs):
        func_calls = []
        sleep_count = 0
        for p in profile_result.profile.values():
            idx = p.idx
            if skip[idx] and np.all(skip[p.input_from]) and not recv[idx]:
                continue    # Ignore completely skipped op

            if skip[idx]:
                if recv[idx]:
                    func_calls.append([skip_with_recv, [p]])
                else:
                    func_calls.append([plain_skip, [p]])
            elif offload[idx]:
                func_calls.append([
                    exec_with_offload, [p, send_slice[idx], send_keep_slice[idx], align_shape[idx]]])
            elif recv[idx]:
                func_calls.append([
                    exec_with_recv, [p, recv_keep_slice[idx], cat_order[idx], cat_dim[idx], align_shape[idx]]])
            else:
                func_calls.append([plain_exec, [p, align_shape[idx]]])
        merged_op = []
        merged_align_shape = []
        ret_func_calls = []
        in_plain_exec = False
        for func_call, args in func_calls:
            if func_call is plain_exec and merge:
                in_plain_exec = True
                merged_op.append(args[0])
                merged_align_shape.append(args[1])
            else:
                if in_plain_exec:
                    if len(merged_op) > 3:
                        sleep_plan = np.zeros(len(merged_op), dtype=bool)
                        sleep_plan[::sleep_step] = True
                        sleep_count += sum(sleep_plan)
                        ret_func_calls.append([merged_plain_exec, [merged_op, merged_align_shape, sleep_plan.tolist()]])
                    else:
                        for op, _align_shape in zip(merged_op, merged_align_shape):
                            ret_func_calls.append([plain_exec, [op, _align_shape]])
                    merged_op = []
                    merged_align_shape = []
                ret_func_calls.append([func_call, args])
                in_plain_exec = False
        if len(merged_op) > 0:
            if len(merged_op) > 3:
                sleep_plan = np.zeros(len(merged_op), dtype=bool)
                sleep_plan[::sleep_step] = True
                sleep_count += sum(sleep_plan)
                ret_func_calls.append([merged_plain_exec, [merged_op, merged_align_shape, sleep_plan.tolist()]])
            else:
                for op, _align_shape in zip(merged_op, merged_align_shape):
                    ret_func_calls.append([plain_exec, [op, _align_shape]])
        return ret_func_calls, sleep_count
    for bw, plan in plans.items():
        profile_result.exec_plan[bw], sleep_count = _plan_to_static_exec(**plan)
        to_offload = np.nonzero(plan["offload"])
        to_recv = np.nonzero(plan["recv"])
        est_time = plan["est_time"]
        log(f"bw {bw}MB/s offload at {to_offload[0].tolist()} recv at {to_recv[0].tolist()} sleep for {sleep_count} ops est time {est_time:.4f}s.")


@torch.no_grad()
async def random_exec_compiled(profile_result: OffloadProfile, bw: int):
    # list(map(lambda x: x[0](*x[1]), profile_result.exec_plan[bw]))
    for exec_func, args in profile_result.exec_plan[bw]:
        await exec_func(*args)
    torch.cuda.synchronize()
    return profile_result.ret_store


@torch.no_grad()
async def random_exec(profile_result: OffloadProfile,
                sock: AsyncTCPMessageStream, log: Callable[[str], None],
                skip: list, offload: list, recv: list,
                send_slice: dict, send_keep_slice: dict, recv_keep_slice: dict,
                cat_dim: dict, cat_order: dict, recv_first: dict,
                align_shape: list, **kwargs):
    """Execute each operator involved in the profile result sequentially.

    Args:
        profile_result (OffloadProfile): profile result of the forward of a model
        sock (AsyncTCPMessageStream)
        log (callable): function to log str
        skip (list): skip plan
        offload (list): offload plan
        recv (list): recv plan
        send_slice (dict): send_slice plan
        send_keep_slice (dict): send_keep_slice plan
        recv_keep_slice (dict): recv_keep_slice plan
        cat_dim (dict): cat_dim plan
        cat_order (dict): cat_order plan

    Returns:
        _type_: _description_
    """
    # Start exec
    for p, _skip, _offload, _recv, _align_shape in zip(
            profile_result.profile.values(), skip, offload, recv, align_shape):
        idx = p.idx
        if _skip:    # Server should always skip the input (first) op
            if _recv:
                with log_dur(log, preSPSO-GA=f"op {idx} {p.func_name} recv"):
                    recved_tensor = await sock.queued_recv()
                fill_output_to_input_slots(recved_tensor, p.output_idx_slots)
            empty_input_slots(p.input_slots)
        else:     # Client should never skip the input (first) op
            args, kwargs = p.func_args
            if _align_shape:
                local_dim = p.local_dim
                arg0, arg1 = args
                align_dim_len = min(arg0.shape[local_dim], arg1.shape[local_dim])
                if _align_shape == 1:
                    slices = [slice(None)] * local_dim + [slice(align_dim_len)] + [slice(None)] * (len(arg0.shape) - local_dim - 1)
                else:   # 2 at the server side
                    slices = [slice(None)] * local_dim + [slice(-align_dim_len, None)] + [slice(None)] * (len(arg0.shape) - local_dim - 1)
                if arg0.shape[local_dim] < arg1.shape[local_dim]:
                    intermediates: torch.Tensor = p.func(arg0, arg1[slices], **kwargs)
                else:
                    intermediates: torch.Tensor = p.func(arg0[slices], arg1, **kwargs)
            else:
                try:
                    intermediates: torch.Tensor = p.func(*args, **kwargs)
                except Exception as e:
                    print(p)

            if _recv or _offload:
                if _recv and recv_first[idx]:
                    with log_dur(log, preSPSO-GA=f"op {idx} {p.func_name} recv"):
                        recv_intermediates = await sock.queued_recv()
                    # merge received tensor with intermediates
                    intermediates = cat_output(
                        intermediates, recv_intermediates,
                        keep_slice=recv_keep_slice[idx],
                        order=cat_order[idx], dim=cat_dim[idx])
                if _offload:
                    intermediates, send_intermediates = slice_output(
                        intermediates,
                        send_slice=send_slice[idx],
                        keep_slice=send_keep_slice[idx])
                    with log_dur(log, preSPSO-GA=f"op {idx} {p.func_name} send"):
                        await sock.queued_send(send_intermediates)
                if _recv and not recv_first[idx]:
                    with log_dur(
                        log, preSPSO-GA=f"op {idx} {p.func_name} recv; keep slice {recv_keep_slice[idx]} cat order {cat_order[idx]}"):
                        recv_intermediates = await sock.queued_recv()
                    # merge received tensor with intermediates
                    intermediates = cat_output(
                        intermediates, recv_intermediates,
                        keep_slice=recv_keep_slice[idx],
                        order=cat_order[idx], dim=cat_dim[idx])
            # fill arguments for following ops
            iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
            # fill_output_to_input_slots(intermediates, p.output_idx_slots)
            for slot in p.input_slots:
                slot.container[slot.index] = None
            # empty_input_slots(p.input_slots)
        # await asyncio.sleep(0.)
        if idx % 10 == 0:
            await asyncio.sleep(0.)
    return profile_result.ret_store

PLAIN_EXEC=0
SKIP_RECV=1
EXEC_OFFLOAD=2
EXEC_RECV=3
PLAIN_SKIP=4

def filter_plan(profile_result: OffloadProfile, plans: Dict[int, Dict[str, list]],
                log=print, sleep_step=10):
    def _plan_to_static_exec(skip: list, offload: list, recv: list, **kwargs):
        exec_plan = []
        sleep_idx = []
        end_idx = len(profile_result.profile) - 1
        offload_ops = np.nonzero(offload)[0]
        recv_ops = np.nonzero(recv)[0]
        valid_op_idx = 0
        next_sleep_idx = 0
        no_sleep = False
        for p in profile_result.profile.values():
            idx = p.idx
            if skip[idx] and np.all(skip[p.input_from]) and not recv[idx]:
                continue    # Ignore completely skipped op

            if len(recv_ops) > 0:
                next_recv = np.where(idx + 1 <= recv_ops,
                                              recv_ops, end_idx).min()
                if (next_recv < end_idx and (np.all(skip[idx+1:next_recv]) or
                    idx+1==next_recv)) or next_recv == end_idx or idx < min(offload_ops):
                    # all skipped between idx and next recv
                    no_sleep = True
                else:
                    no_sleep = False

            if skip[idx]:
                if recv[idx]:
                    current_plan = [p, SKIP_RECV]
                else:
                    current_plan = [p, PLAIN_SKIP]
            elif offload[idx]:
                current_plan = [p, EXEC_OFFLOAD]
            elif recv[idx]:
                current_plan = [p, EXEC_RECV]
            else:
                current_plan = [p, PLAIN_EXEC]
            if not no_sleep and (len(offload_ops) > 0 and\
                np.any(idx >= offload_ops) and np.any(idx < recv_ops) or\
                    valid_op_idx >= next_sleep_idx):
                # For offloaded: Sleep at every valid op between offload and recv
                # For local comp: Sleep at a SPSO-GAed step size
                current_plan.append(True)   # Sleep bool at the last element
                next_sleep_idx = valid_op_idx + sleep_step
                sleep_idx.append(idx)
            else:
                current_plan.append(False)
            exec_plan.append(current_plan)
            valid_op_idx += 1
        return exec_plan, sleep_idx
    for bw, plan in plans.items():
        profile_result.exec_plan[bw], sleep_plan = _plan_to_static_exec(**plan)
        to_offload = np.nonzero(plan["offload"])
        to_recv = np.nonzero(plan["recv"])
        est_time = plan["est_time"]
        log(f"bw {bw}MB/s offload at {to_offload[0].tolist()} recv at {to_recv[0].tolist()} sleep at {sleep_plan} est time {est_time:.4f}s exec ops {len(profile_result.exec_plan[bw])}.")

async def random_exec_filtered(exec_plan: List[Tuple[TorchOPProfile, int, bool]],
                sock: AsyncTCPMessageStream, log=print,
                send_slice: dict=None, keep_slice: dict=None,
                cat_dim: dict=None, cat_order: dict=None, **kwargs):
    """Filter out unnecessary ops (e.g., skipped due to offloading) and reduce function calls

    Args:
        exec_plan (List[Tuple[TorchOPProfile, int, bool]]): _description_
        sock (AsyncTCPMessageStream): _description_
        log (Callable[[str], None]): _description_
        send_slice (dict): _description_
        keep_slice (dict): _description_
        cat_dim (dict): _description_
        cat_order (dict): _description_
    """
    for p, plan, sleep in exec_plan:
        if plan == PLAIN_EXEC:
            args, kwargs = p.func_args
            intermediates: torch.Tensor = p.func(*args, **kwargs)
        elif plan == SKIP_RECV:
            with log_dur(log, preSPSO-GA=f"op {p.idx} {p.func_name} recv"):
                intermediates = await sock.queued_recv()
        elif plan == EXEC_OFFLOAD:
            args, kwargs = p.func_args
            intermediates: torch.Tensor = p.func(*args, **kwargs)
            intermediates, send_intermediates = slice_output(
                intermediates,
                send_slice=send_slice[p.idx],
                keep_slice=...)
            with log_dur(log, preSPSO-GA=f"op {p.idx} {p.func_name} send"):
                await sock.queued_send(send_intermediates)
        elif plan == EXEC_RECV:
            idx = p.idx
            args, kwargs = p.func_args
            intermediates: torch.Tensor = p.func(*args, **kwargs)
            with log_dur(log, preSPSO-GA=f"op {idx} {p.func_name} recv"):
                recv_intermediates = await sock.queued_recv()
            # merge received tensor with intermediates
            intermediates = cat_output(
                intermediates, recv_intermediates,
                keep_slice=keep_slice[idx],
                order=cat_order[idx], dim=cat_dim[idx])
        elif plan == PLAIN_SKIP:
            for slot in p.input_slots:
                slot.container[slot.index] = None
            if sleep:
                asyncio.sleep(0.)
            continue
        else:
            raise RuntimeError(f"Not supported action {plan}")

        if sleep:
            asyncio.sleep(0.)
        # Empty input slots
        for slot in p.input_slots:
            slot.container[slot.index] = None

        # fill output to input slots of following ops
        iterate_tensor(intermediates,
                       partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
    # The results are stored in profile_result.ret_store

def local_random_exec_profile(profile_result: OffloadProfile, log=print):
    torch.cuda.synchronize()
    last_stamp = time.time()
    stime = last_stamp
    # Start exec
    for p in profile_result.profile.values():
        if p.func:
            args, kwargs = p.func_args
            intermediates: torch.Tensor = p.func(*args, **kwargs)
        else:
            intermediates = p.func_args

        fill_output_to_input_slots(intermediates, p.output_idx_slots)
        empty_input_slots(p.input_slots)
        torch.cuda.synchronize()
        c_stamp = time.time()
        p.ops_time = c_stamp - last_stamp
        last_stamp = c_stamp
    log(f"total time from exec step by step with torch.cuda.synchronize: {time.time() - stime:.4e}s.")
    return profile_result.ret_store

async def local_random_exec(profile_result: OffloadProfile):
    # Start exec
    for i, p in enumerate(profile_result.profile.values()):
        args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        # fill arguments for following ops
        fill_output_to_input_slots(intermediates, p.output_idx_slots)
        empty_input_slots(p.input_slots)
    torch.cuda.synchronize()
    return profile_result.ret_store
