#! /usr/bin/env python
import sys
import atexit
import time
import copy
from typing import List, Dict, Callable, Tuple, Any, Iterable
from packaging.version import Version
import os.path as osp
import hashlib
from typing import List
import pickle
from collections import OrderedDict
import socket
from threading import Thread, Event
from queue import Queue
import concurrent
import gc
import asyncio
from functools import partial
import numpy as np
import torch
import dill
import torch.backends
from .hook_tensor import (profile_tensor_factory,
                        OffloadProfile, TorchOPProfile,
                        keep_funcs, profile_model_ops)
from .slice_hook import hook_model, slice_forward, CustomIdentity, profile_ops_time
from .bm_socket import BiDirectionalSocket
from .recursive_pickle_obj import recur_dump_obj
from ._utils import iterate_tensor, iterate_all_close, log_dur
from .schedule import IDP_scheduler
from .profile_pickle import profile_pickle
from .estimate_bandwidth import init_bandwidth_monitor
from .estimate_power_consumption import init_power_monitor
from .limit_bandwidth import start_replay_bandwidth


# None zero GPU Memory occupancy will be observed due to cuda context
# https://discuss.pytorch.org/t/nvidia-smi-does-not-drop-after-empty-cache-although-cuda-memory-summary-says-there-is-no-current-usage/143741
# print(f"{torch.cuda.memory_allocated()} {torch.cuda.max_memory_allocated()}")

empty_preamble = b"0"*int(1024*1024*0.5) # 0.5MB
class intraDP:
    def __init__(self, offload=True,  parallel_approach = "select",
                ip="127.0.0.1", port=12345, ctrl_port=12346, debug=False,
                constraint_latency=False, log=print) -> None:
        self.offload = offload
        self.log = log
        self.idx = 0
        self.server_ip = ip
        self.server_port = port
        self.server_ctrl_port = port + 1
        self.debug = debug
        self.init_forward_count = 0
        self.parallel_approach = parallel_approach
        self.constraint_latency = constraint_latency
        self.SPSO-GAed_bw = None
        self.role = "client"
        self.model_name = None
        log("Configuring torch to use deterministic behaviors.")
        torch.manual_seed(0)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.deterministic = True

        log(f"parallel approach {parallel_approach}")
        log(f"constraint_latency {constraint_latency}")

    def offload_order(self, bw: int):
        self.SPSO-GAed_bw = bw

    def start_client(self, args, kwargs, model: torch.nn.Module,
                     init_forward_count=2, network_interface="eth0", log_dir="."):
        self.model_name = model.__class__.__name__
        assert init_forward_count >= 0
        self.role = "client"
        for p in model.parameters():
            p.requires_grad = False

        sock = BiDirectionalSocket(address=self.server_ip, port=self.server_port, mode="client", log=self.log)
        self.log(f"Connecting to server {self.server_ip}: {self.server_port}")
        while True:
            sock.send_message("Examinining connection...")
            if (_msg := sock.receive_message_timeout(3)) != "":
                assert _msg == "Connection established.", f"Wrong msg {_msg}"
                break
            else:
                self.log("Retrying...")
                continue
        self.log(f"Connected to server {self.server_ip}: {self.server_port}")

        model_msg = recur_dump_obj(model)
        sock.send_raw_message(model_msg)
        assert sock.receive_message() == "Model received."
        self.log(f"Send model to server {len(model_msg)/1024/1024:.4f}MB.")
        sock.send_message(init_forward_count)
        param_num = 0
        for p in model.parameters():
            param_num += p.numel()
        self.log(f"Model parameter number {param_num/1e6:.4f}M.")
        
        old_forward = model.forward
        wrapped_forward, ret = self.process_model(model.forward, args, kwargs, sock, init_forward_count, role="client",
                               preSPSO-GA="client: ", log=self.log)
        model.forward = wrapped_forward
        power_monitor_stop = init_power_monitor(log_file=osp.join(log_dir, "power_consumption.log"), interval=1000)
        atexit.register(power_monitor_stop)
        atexit.register(sock.stop)
        assert sock.receive_message() == "Init complete"
        sock.send_message("Init complete")
        self.log("All init complete.\n")
        return ret

    def start_server(self, bandwidth_file_path):
        self.role = "server"
        self.log(f"starting intraDP server on {self.server_ip}:{self.server_port}")
        torch.set_grad_enabled(False)
        sock = BiDirectionalSocket(address=self.server_ip, port=self.server_port, mode="server", log=self.log)
        assert sock.receive_message() == "Examinining connection..."
        time.sleep(2)
        while not sock.recv_queue.empty():
            sock.recv_queue.get()
        sock.send_message("Connection established.")
        try:
            model: torch.nn.Module = dill.loads(sock.receive_raw_message())
            sock.send_message("Model received.")
            peername = sock.peername if sock.peername else "client"
            preSPSO-GA = f"server for {peername}: "
            num_param = int(sum([p.numel() for p in model.parameters()])/1024/1024)
            model_name = f"{model.__class__.__name__}_{num_param}M"
            self.model_name = model_name
            self.log(preSPSO-GA + f"model {model_name} initial complete.")
            init_forward_count = sock.receive_message()
            server_forward = self.process_model(model.forward, None, None, sock, init_forward_count, role="server",
                               preSPSO-GA="server: ", log=self.log, model_hash=model_name)
        except EOFError:
            self.log("Connection closed by client.")
            sock.stop()
            self.log("Stopped server.")
            sys.exit(0)
        except Exception as e:
            self.log(str(e))
            raise e
        limit_bandwidth_stop = start_replay_bandwidth(bandwidth_file_path)
        self.bw_monitor, monitor_stop = init_bandwidth_monitor(interval=0.05)
        atexit.register(limit_bandwidth_stop)
        atexit.register(monitor_stop)
        atexit.register(sock.stop)
        time.sleep(1)
        print(f"bandwidth {self.bw_monitor()}MBps")
        time.sleep(4)
        print(f"bandwidth {self.bw_monitor()}MBps")
        sock.send_message("Init complete")
        assert sock.receive_message() == "Init complete"
        self.log("All init complete.\n")
        try:
            while True:
                if not server_forward():
                    break
        except (SystemExit, KeyboardInterrupt):
            sock.stop()
            self.log("Stopped server.")
        finally:
            sock.stop()
            self.log("Stopped server.")

    @torch.no_grad()
    def process_model(self, old_forward, args, kwargs,
                        sock: BiDirectionalSocket,
                        init_forward_count=0, role="client", preSPSO-GA="client", log=print, model_hash=""):
        if kwargs is None:
            kwargs = {}

        if role == "client":
            parallel_approach = self.parallel_approach
            scheduler = IDP_scheduler(parallel_approach)
            sock.send_message(parallel_approach)
        else:
            parallel_approach = sock.receive_message()
            self.parallel_approach = parallel_approach
            scheduler = IDP_scheduler(parallel_approach)
        if "intraDP" in parallel_approach:
            self.debug = True   # TODO

        @torch.no_grad()
        def _profile_forward(*args, **kwargs):
            profile_result = profile_model_ops(old_forward, args, kwargs, log=self.log)
            orig_ops_num = len(profile_result.profile)
            self.parse_profile(profile_result)
            # profile_pickle(profile_result, log=log)
            profile_result.end_idx = len(profile_result.profile) - 1
            return profile_result, orig_ops_num

        @torch.no_grad()
        def profile_forward(args, kwargs, warmup=2, repeat=3):
            assert warmup > 0 and repeat > 0
            if role == "client":
                sock.send_message([args, kwargs])
                log(preSPSO-GA + "send init input to server")
            else:
                args, kwargs = sock.receive_message()
                log(preSPSO-GA + "recv init input from client")
            origin_input = copy.deepcopy([args, kwargs])
            log(f"Input size {len(pickle.dumps([args, kwargs]))/1024/1024:.4f}MB")
            log(f"Forwarding for {init_forward_count}(+{warmup} warmup and {repeat} repeat) times for initialization.")
            count = 0
            while count != warmup:
                orig_ret = old_forward(*args, **kwargs)
                torch.cuda.synchronize()
                count += 1

            count = 0
            _init_forward_count = init_forward_count + repeat
            stime = time.time()
            while count != _init_forward_count:
                orig_ret = old_forward(*args, **kwargs)
                torch.cuda.synchronize()
                count += 1
            dur = (time.time() - stime)/count   # Average duration for each forward
            log(f"Forward of the original model takes average {dur:.4f}s.")

            profile_result, orig_ops_num = _profile_forward(*origin_input[0], **origin_input[1])
            profile_pickle(profile_result=profile_result, log=self.log)
            profile_result.original_input = origin_input
            hooked_forward = partial(slice_forward, hook_funcs={}, forward_func=old_forward, profile_result=profile_result, mode=self.role, sock=sock)

            log(f"Output size {len(pickle.dumps(orig_ret))/1024/1024:.4f}MB")

            profile_result.profile[0].func_args = [args, kwargs]
            for _ in range(warmup):
                hooked_forward(args, kwargs)
            stime = time.time()
            for _ in range(repeat):
                hooked_forward(args, kwargs)
            _dur = (time.time() - stime) / repeat
            log(f"Hooked forward takes average {_dur:.4f}s.")

            log("Using torch.profiler for op profile")
            profile_result: OffloadProfile = profile_ops_time(hooked_forward, origin_input[0], origin_input[1],
                                    warmup_runs=warmup, measure_runs=repeat,
                                    profile_result=profile_result, log=self.log)

            factor = dur / sum(p.ops_time for p in profile_result.profile.values())
            if role == "client":
                factor += 0.1

            if role == "client" and self.server_ip == "127.0.0.1":    # For debug
                factor = 10.
                profile_result.local_comp_time *= factor
            log(f"Operator records (align ops time with factor {factor:.4f}): ")
            accumulated_time = 0.
            for p in profile_result.profile.values():
                p.ops_time *= factor
                accumulated_time += p.ops_time
                log(f"{p} accu_time {accumulated_time:.4f}s")
            # try to sleep for every 10ms
            log(f"total {len(profile_result.profile)} ops (filtered from {orig_ops_num} ops); time {sum(p.ops_time for p in profile_result.profile.values()):.4f}s (aligned by {factor:.4f}\n")

            if "intraDP" in parallel_approach:
                if role == "client":
                    sock.send_message({"ops": profile_result.copy_for_transmission(),
                                    "constraint_latency": self.constraint_latency,
                                    "expected_exec_time": profile_result.local_comp_time})
                    log("Waiting for graph processing at the server")
                    server_plan, server_expected_exec_time = sock.receive_message()
                    profile_result.client_comp_time = profile_result.local_comp_time
                    profile_result.server_comp_time = server_expected_exec_time
                    scheduler.recv_plan(server_plan)
                    log("Got graph plan from server")
                elif role == "server":
                    ops_dict = sock.receive_message()
                    robot_ops = ops_dict["ops"]
                    scheduler.update_ops(robot_ops, profile_result)
                    constraint_latency = ops_dict["constraint_latency"]
                    client_comp_time = ops_dict["expected_exec_time"]
                    profile_result.client_comp_time = client_comp_time
                    profile_result.server_comp_time = profile_result.local_comp_time
                    store_plan_path = None
                    if "intraDP" in parallel_approach:
                        store_plan_path = f"intraDP_plan_{model_hash}.pkl"
                    elif "select" in parallel_approach:
                        if constraint_latency:
                            store_plan_path = f"select_constraint_latency_plan_{model_hash}_constraint.pkl"
                        else:
                            store_plan_path = f"select_plan_{model_hash}.pkl"
                    if constraint_latency:
                        # SPSO-GA latency requirement to 1Hz
                        scheduler.required_latency = 1.
                        self.log(f"Setting required_latency to {scheduler.required_latency:.4}s.")
                    self.log("Computing plan for client.")
                    if store_plan_path and osp.exists(store_plan_path):
                        with open(store_plan_path, "rb") as f:
                            data = pickle.load(f)
                            scheduler.server_plans = data["server plan"]
                            scheduler.client_plans = data["client plan"]
                    else:
                        scheduler.build_graph()
                    if store_plan_path and not osp.exists(store_plan_path):
                        with open(store_plan_path, "wb") as f:
                            pickle.dump({"server plan": scheduler.server_plans,
                                        "client plan": scheduler.client_plans}, f)

                    sock.send_message([scheduler.client_plans, profile_result.server_comp_time])
                    scheduler.recv_plan(scheduler.server_plans)  # Server does not output
                    self.log(f"Number of local ops {scheduler.info.intraDP_layers.sum()}")
                    self.log(f"Number of global ops {~(scheduler.info.intraDP_layers).sum()}")
                self.log("--------------------------------")
                self.log(f"offload recv plan: ")
                for bw in range(0, scheduler.max_bw):
                    plan = scheduler.graph_plan[bw]
                    offload_at = np.nonzero(plan['offload'])
                    recv_at = np.nonzero(plan['recv'])
                    self.log(f"bw {bw} offload {offload_at} recv {recv_at}")
            else:
                if role == "client":
                    sock.send_message([profile_result.local_comp_time, profile_result.size_to_dumps_time, profile_result.size_to_loads_time])
                    server_exec_time, server_size_to_dumps_time, server_size_to_loads_time = sock.receive_message()
                    client_exec_time = profile_result.local_comp_time
                    client_size_to_dumps_time = profile_result.size_to_dumps_time
                    client_size_to_loads_time = profile_result.size_to_loads_time
                else:
                    client_exec_time, client_size_to_dumps_time, client_size_to_loads_time = sock.receive_message()
                    sock.send_message([profile_result.local_comp_time, profile_result.size_to_dumps_time, profile_result.size_to_loads_time])
                    server_exec_time = profile_result.local_comp_time
                    server_size_to_dumps_time = profile_result.size_to_dumps_time
                    server_size_to_loads_time = profile_result.size_to_loads_time
                profile_result.client_comp_time = client_exec_time
                profile_result.server_comp_time = server_exec_time
                profile_result.client_size_to_dumps_time = client_size_to_dumps_time
                profile_result.client_size_to_loads_time = client_size_to_loads_time
                profile_result.server_size_to_dumps_time = server_size_to_dumps_time
                profile_result.server_size_to_loads_time = server_size_to_loads_time


            self.log(preSPSO-GA + "init forward complete.")
            local_operator_num = 0
            global_operator_num = 0
            for key, profile in profile_result.profile.items():
                if key not in [0, profile_result.end_idx]:
                    if not profile.excluded and len(profile.output_shapes) > 0:
                        if profile.local_dim is not None:
                            local_operator_num += 1
                        else:
                            global_operator_num += 1
            self.log(preSPSO-GA + f" local operator num {local_operator_num}; global operator num {global_operator_num}")


            return profile_result, orig_ret

        self.log(preSPSO-GA + "start profile forward")
        profile_result, ret = profile_forward(args, kwargs, warmup=10, repeat=10)
        self.log(preSPSO-GA + "profile forward complete")

        self.log(f"Server computation time: {profile_result.server_comp_time:.4f}s; client computaiton time: {profile_result.client_comp_time:.4f}s")

        gc.collect()
        torch.cuda.empty_cache()
        hook_funcs_bw = {}
        expected_exec_time = {}
        idx_array = list(profile_result.profile.keys())
        if self.model_name.startswith(("Model", "SCONet")):
            align_cat = True
            if self.model_name.startswith("Model"):
                align_cat_indices = [197, 220, 243]
            else:
                align_cat_indices = [25, 31, 38]
        else:
            align_cat = False
            align_cat_indices = []

        self.log(f"\nParallel approach: {self.parallel_approach}. Adding hooks for each bw...")
        for bw in range(0, scheduler.max_bw):
            plan = scheduler.graph_plan.get(bw, None)
            partial_agr_time = float('inf')
            local = False
            all = False
            partial_agr = False
            all_offload_time = profile_result.server_comp_time + \
                profile_result.client_offload_time(0, bw * 0.9, False) + \
                    profile_result.server_offload_time(idx_array[-1], bw * 0.9, True)

            all_local_time = profile_result.client_comp_time
            if self.model_name.startswith("SCONet"):
                if role == "client":
                    profiles = profile_result.profile
                    partial_agr_time = sum(profiles[i].ops_time for i in range(8)) +\
                        profile_result.client_offload_time(7, bw * 0.9, True) + \
                        sum(profiles[i].ops_time for i in idx_array[8:]) + \
                        profile_result.server_offload_time(idx_array[-1], bw * 0.9, True)
                    sock.send_message(partial_agr_time)
                else:
                    partial_agr_time = sock.receive_message()
            if parallel_approach == "local":
                local = True
            elif parallel_approach == "all":
                all = True
            elif parallel_approach in ["select", "SPSO-GA", "DSCCS"]:
                if parallel_approach == "SPSO-GA": # 1s latency
                    if all_offload_time <= 1:
                        all = True
                    elif self.model_name.startswith("SCONet") and partial_agr_time < 1:
                        partial_agr = True
                    else:
                        local = True
                else:
                    if self.model_name.startswith("SCONet") and partial_agr_time < all_offload_time and partial_agr_time < all_local_time:
                        partial_agr = True
                    elif all_offload_time > all_local_time:
                        local = True
                    else:
                        all = True
            if local:
                offload = np.zeros_like(idx_array, dtype=bool)
                recv = np.zeros_like(idx_array, dtype=bool)
                align_shape = np.zeros_like(idx_array, dtype=bool)
                recv_first = np.ones_like(idx_array, dtype=bool)
                if role == "client":
                    skip = np.zeros_like(idx_array, dtype=bool)
                else:
                    skip = np.ones_like(idx_array, dtype=bool)
                send_slice = {}
                send_keep_slice = {}
                recv_keep_slice = {}
                expected_exec_time[bw] = all_local_time
            elif all:
                offload = np.zeros_like(idx_array, dtype=bool)
                recv = np.zeros_like(idx_array, dtype=bool)
                align_shape = np.zeros_like(idx_array, dtype=bool)
                recv_first = np.ones_like(idx_array, dtype=bool)
                if role == "client":
                    offload[0] = True
                    recv[idx_array[-1]] = True
                    skip = np.ones_like(idx_array, dtype=bool)
                    send_slice = {0: [...]}
                    send_keep_slice = {0: [slice(0)]}
                    recv_keep_slice = {idx_array[-1]: [slice(0)]}
                else:
                    offload[idx_array[-1]] = True
                    recv[0] = True
                    skip = np.zeros_like(idx_array, dtype=bool)
                    skip[0] = True
                    send_slice = {idx_array[-1]: [...]}
                    send_keep_slice = {idx_array[-1]: [slice(0)]}
                    recv_keep_slice = {0: [slice(0)]}
                expected_exec_time[bw] = all_offload_time
            elif partial_agr:
                offload = np.zeros_like(idx_array, dtype=bool)
                recv = np.zeros_like(idx_array, dtype=bool)
                align_shape = np.zeros_like(idx_array, dtype=bool)
                recv_first = np.zeros_like(idx_array, dtype=bool)
                if role == "client":
                    offload[7] = True
                    recv[idx_array[-1]] = True
                    skip[8:] = True
                    send_slice = {7: [...]}
                    send_keep_slice = {7: [slice(0)]}
                    recv_keep_slice = {idx_array[-1]: [slice(0)]}
                else:
                    recv[7] = True
                    offload[idx_array[-1]] = True
                    skip[:8] = True
                    send_slice = {idx_array[-1]: [...]}
                    send_keep_slice = {idx_array[-1]: [slice(0)]}
                    recv_keep_slice = {7: [slice(0)]}
                expected_exec_time[bw] = partial_agr_time
            else:
                offload = plan["offload"]
                recv = plan["recv"]
                skip = plan["skip"]
                align_shape = plan["align_shape"]
                recv_first = plan["recv_first"]
                send_slice = plan["send_slice"]
                send_keep_slice = plan["send_keep_slice"]
                recv_keep_slice = plan["recv_keep_slice"]
                expected_exec_time[bw] = plan["est_time"]
            hook_funcs = hook_model(profile_result, role, offload, recv, skip, align_shape, recv_first,
                                           send_slice, send_keep_slice, recv_keep_slice, align_cat, align_cat_indices)
            hook_funcs_bw[bw] = hook_funcs
            info = ""
            idx = 0
            end_idx = idx_array[-1]
            while idx <= end_idx:
                pre_hook, post_hook, empty = hook_funcs.get(idx, (None, None, False))
                info += f" {pre_hook} " if pre_hook else ""
                info += f" {post_hook} " if post_hook else ""
                if empty:
                    for _idx in range(idx + 1, end_idx + 1):
                        pre_hook, post_hook, empty = hook_funcs.get(_idx, (None, None, False))
                        info += f" {pre_hook} " if pre_hook else ""
                        info += f" {post_hook} " if post_hook else ""
                        if not empty:
                            _idx -= 1
                            break
                    if idx < end_idx:
                        idx = _idx
                else:
                    for _idx in range(idx + 1, end_idx + 1):
                        pre_hook, post_hook, empty = hook_funcs.get(_idx, (None, None, False))
                        info += f" {pre_hook} " if pre_hook else ""
                        info += f" {post_hook} " if post_hook else ""
                        if empty:
                            _idx -= 1
                            break
                    if idx < end_idx:
                        idx = _idx
                idx += 1
            self.log(f"bw {bw}: {info}")

        def client_forward(*args, **kwargs):
            sock.send_message(self.idx)
            bw = sock.receive_message()
            stime = time.time()
            ret, communicated_bytes, inner_forward_dur = slice_forward(args, kwargs, hook_funcs_bw[bw], old_forward, profile_result, role, sock, log=self.log)
            etime = time.time()
            self.log(f"{self.idx}th client forward: bw {bw}, expected_exec_time {expected_exec_time[bw]:.4f}s, duration {etime - stime:.4f}s, inner_forward_dur {inner_forward_dur:.4f}s communicated_bytes {communicated_bytes/1e6:.4f}MB")
            self.idx += 1
            return ret

        def server_forward(bw=None):
            idx = sock.receive_message()
            preSPSO-GA = f"{idx}th server forward: "
            if idx == "" or idx is None:
                sock.send_message(10)
                return False
            if bw is None:
                bw = int(min(np.around(self.bw_monitor(), 0), scheduler.max_bw-1))
            sock.send_message(bw)
            args, kwargs = profile_result.original_input
            stime = time.time()
            ret, communicated_bytes, inner_forward_dur = slice_forward(
                args, kwargs, hook_funcs_bw[bw], old_forward, profile_result, role, sock, log=self.log)
            etime = time.time()
            self.log(f"{idx}th server forward: bw {bw}, expected_exec_time {expected_exec_time[bw]:.4f}s, duration {etime - stime:.4f}s, inner_forward_dur {inner_forward_dur:.4f}s communicated_bytes {communicated_bytes/1e6:.4f}MB")
            return True

        if role == "client":
            sock.send_message(self.debug)
            if self.debug:
                for bw in range(0, scheduler.max_bw):
                    client_forward(*args, **kwargs)
                log(preSPSO-GA + "Debug forward complete")
            return client_forward, ret
        else:
            debug = sock.receive_message()
            if debug:
                for bw in range(0, scheduler.max_bw):
                    server_forward(bw)
                log(preSPSO-GA + "Debug forward complete")
            return server_forward

    def parse_profile(self, profile_result: OffloadProfile, verbose: bool = False, log=print):
        all_profiles = profile_result.profile
        idx_array = list(all_profiles.keys())
        sorted_ids = []
        excluding_indices = []
        for idx in idx_array:
            profile: TorchOPProfile = all_profiles[idx]
            input_ids: list = profile.input_ids
            output_ids: list = profile.output_ids

            # parse input/output relationship by querying id in previous output
            excluded = True
            for i, _id in enumerate(input_ids):
                for _idx in range(0, idx):
                    if _id in all_profiles[_idx].output_ids:
                        if _idx not in excluding_indices:
                            excluded = False  # Input from non-excluded op
                        sorted_ids.append(_id)
                        sorted_ids = list(set(sorted_ids))
                        hit_idx = all_profiles[_idx].output_ids.index(_id)
                        if _idx in profile.input_from:
                            if verbose:
                                log(f"Warning: {idx}op {profile.func_name} ({profile.file_name}:{profile.line_num}) has duplicated input from {_idx}op {all_profiles[_idx].func_name} ({all_profiles[_idx].file_name}:{all_profiles[_idx].line_num})")
                        else:
                            profile.input_from.append(_idx)
                        if idx in all_profiles[_idx].output_to:
                            if verbose:
                                log(f"Warning: {_idx}op {all_profiles[_idx].func_name} ({all_profiles[_idx].file_name}:{all_profiles[_idx].line_num}) has duplicated output to {idx}op {profile.func_name} ({profile.file_name}:{profile.line_num})")
                        else:
                            all_profiles[_idx].output_to.append(idx)
            if len(profile.input_from) == 0 and idx > 0:    # Temporary SPSO-GA for TorchFunctionMode failure in capturing select op
                temp_idx = idx - 1
                while len(all_profiles[temp_idx].output_shapes) == 0:
                    temp_idx -= 1
                profile.input_from.append(temp_idx)
                all_profiles[temp_idx].output_to.append(idx)
            if excluded and idx not in [0, idx_array[-1]] or not profile.output_shapes and profile.func_name not in ["__setitem__"]:
                # if excluded and idx not in [0, idx_array[-1]]:
                #     profile.static_computed = True
                excluding_indices.append(idx)
                profile.excluded = True
                profile.masked = True


        # patch __setitem__: __setitem__ does not have a return value
        # but only modifies input_from[0] inplace;
        # Correct the data dependency here and also __setitem__ should not be offloaded.
        for i, profile in enumerate(profile_result.profile.values()):
            if profile.func_name == "__setitem__":
                inplace_mod_idx = profile.input_from[0]
                for idx in profile_result.profile[inplace_mod_idx].output_to:
                    if idx > i:     # op after this inplace __setitem__ also depends on this op
                        _p = profile_result.profile[idx]
                        _p.input_from.append(i)
                        profile.output_to.append(idx)

        # Check profile valid
        new_all_profiles = profile_result.profile
        for key, profile in new_all_profiles.items():
            assert key == profile.idx
            for idx in profile.output_to:
                assert idx in new_all_profiles
                assert key in new_all_profiles[idx].input_from
            for idx in profile.input_from:
                assert idx in new_all_profiles
                assert key in new_all_profiles[idx].output_to

        # Temp SPSO-GA for branches
        idx_array = list(new_all_profiles.keys())
        for idx in idx_array:
            profile = new_all_profiles[idx]
            valid_output_len = len(profile.output_to)
            if valid_output_len > 1:
                current_end = set(profile.output_to)
                while len(current_end) > 1:
                    current_idx = min(current_end)
                    current_end = list(current_end)
                    del current_end[current_end.index(current_idx)]
                    current_end += new_all_profiles[current_idx].output_to
                    current_end = set(current_end)
                    if current_idx != idx and len(current_end) > 1:
                        new_all_profiles[current_idx].masked = True

        new_all_profiles[idx_array[-1]].masked = False
        profiles = profile_result.profile
        for profile in list(profile_result.profile.values()):
            idx: int = profile.idx
            func: str = profile.func_name
            input_shapes = profile.input_shapes
            args, kwargs = profile.func_args
            if kwargs is None:
                kwargs = {}
            output_shapes = profile.output_shapes
            if profile.input_from:
                parent_profile = profile_result.profile[profile.input_from[0]]
                last_local_dim = parent_profile.local_dim
            else:
                parent_profile = None
                if profile.input_shapes and len(profile.input_shapes[0]) > 2:
                    last_local_dim = len(profile.input_shapes[0]) - 1
                else:
                    last_local_dim = None
            if func in ["_start"]:
                profile.local_dim = len(profile.output_shapes[0]) - 1
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": False, "profile": profile} # Regular offloading
            elif func in ["_end"]:
                profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": True, "profile": profile} # Regular offloading
            elif func in ["__get__", "dim"]:
                profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": False, "profile": profile} # Regular offloading
            elif func in ["__getitem__"]:
                slice_arg: list = args[1]
                assert len(kwargs) == 0
                if last_local_dim is None:
                    barrier = True
                    profile.local_dim = None
                elif isinstance(slice_arg, int): # get item at the first dim
                    if slice_arg == last_local_dim:
                        barrier = True
                        profile.local_dim = None
                    else:
                        barrier = False
                        profile.local_dim = last_local_dim - 1
                elif isinstance(slice_arg, (torch.Tensor)):
                    # TODO: this situation is very complicated to analyse, leave it to future
                    if len(input_shapes[0]) != len(output_shapes[0]):
                        barrier = True
                        profile.local_dim = None
                    else:
                        barrier = False     
                        profile.local_dim = last_local_dim
                elif isinstance(slice_arg, list):
                    slice_arg = list(slice_arg)
                    try:
                        find_ellipsis = slice_arg.index(...)
                    except ValueError:
                        find_ellipsis = None
                    len_diff = len(input_shapes[0])-len(slice_arg)
                    if find_ellipsis is not None:
                        origin_shape_len = len(input_shapes[0])
                        ellipsis_len = origin_shape_len - len(slice_arg)
                        ellipsis_idx = slice_arg.index(...)
                        [slice_arg.insert(ellipsis_idx, None) for _ in range(ellipsis_len)]
                    elif len_diff > 0:
                        slice_arg += len_diff * [None]
                    if isinstance(slice_arg[last_local_dim], int):
                        barrier = True
                        profile.local_dim = None
                    else:
                        barrier = False
                        dim_reduced = 0
                        for i, a in enumerate(slice_arg):
                            if i == last_local_dim:
                                profile.local_dim = last_local_dim - dim_reduced
                                break
                            if isinstance(a, int):
                                dim_reduced += 1
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["__setitem__",]:
                barrier = last_local_dim is None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
                profile.local_dim = parent_profile.local_dim
            elif func in ["select", "masked_select"]:
                barrier = False
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
                profile.local_dim = 0
            elif func.startswith(("cat")):
                barrier = last_local_dim is None
                if "dim" in kwargs:
                    dim = kwargs["dim"]
                elif len(args) > 1:
                    dim = args[1]
                else:
                    dim = 0
                if dim != last_local_dim and len(profile.input_shapes) > len(args[0]):
                    align_shape = True
                else:
                    align_shape = False
                profile.align_shape = align_shape
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile,
                    "apply_rate": True, "align_shape": align_shape} # Regular offloading
                profile.local_dim = last_local_dim
            elif func.startswith((
                "add", "sub", "rsub", "div", "mul",
                "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
                "exp", "pow",
                )) or func in [
                    "le", "lt", "gt", "ge", "eq","nms",
                ]: # Element-wise operations that keep tensor shape
                align_shape = False
                if last_local_dim is not None:
                    for i in profile.input_from:
                        if all_profiles[i].excluded and len(all_profiles[i].output_shapes[0]) > last_local_dim and all_profiles[i].output_shapes[0][last_local_dim] > 1:
                            align_shape = True
                            break
                profile.align_shape = align_shape
                barrier = last_local_dim is None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile, "align_shape": align_shape} # Regular offloading
                profile.local_dim = last_local_dim
            elif func.startswith(("sin", "cos", "tan", "asin", "acos", "atan", "arc",
                "batch_norm", "layer_norm",
                "relu", "rrelu", "gelu", "sigmoid", "sign", "selu", "hardswish",
                "hardsigmoid", "silu", "to", "to_", "float", "int", "double", "long", "abs",
                "sqrt", "rsqrt",)) or func in [
                    "contiguous", "interpolate", "clone", "detach", 
                    "float", "int", "double", "long", "abs", "type"]:
                barrier = last_local_dim is None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
                profile.local_dim = last_local_dim
            elif func in ["view", "reshape"]:
                barrier = False
                if last_local_dim is not None:
                    input_shape = input_shapes[0]
                    output_shape = output_shapes[0]
                    search_size_h = np.prod(input_shape[:last_local_dim])
                    search_size_hw = np.prod(input_shape[:last_local_dim+1])
                    cum_shape = np.cumprod(output_shape)
                    hw_remained = np.any(cum_shape == search_size_hw)
                    if not hw_remained:
                        barrier = True
                    searched = np.nonzero(cum_shape == search_size_hw)[0]
                    if len(searched):
                        if not barrier:
                            profile.local_dim = searched[0]
                        searched_idx = searched[0]
                        if -1 in args[1:] and args[1+searched_idx] != -1:
                            idx = args[1:].index(-1)
                            args[1+idx] = output_shape[idx]
                        args[1+searched_idx] = -1   # Change the shape of local dim to be DSCCSible
                    else:
                        profile.local_dim = None
                        barrier = True
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["flatten", "ravel"]:
                if last_local_dim is not None and last_local_dim == len(input_shapes[0])-1:
                    profile.local_dim = 0
                    barrier = False
                else:
                    profile.local_dim = None
                    barrier = True
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["unflatten"]:
                dim = args[1]
                if dim != last_local_dim:
                    if dim < last_local_dim:
                        profile.local_dim = last_local_dim + len(args[2]) - 1
                    else:
                        profile.local_dim = last_local_dim
                    barrier = False
                else:
                    profile.local_dim = None
                    barrier = True
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["squeeze", "unsqueeze"]:
                barrier = False
                if last_local_dim is not None:
                    input_shape = input_shapes[0]
                    output_shape = output_shapes[0]
                    search_size = np.prod(input_shape[:last_local_dim+1])
                    searched = np.nonzero(np.cumprod(output_shape) == search_size)[0]
                    if len(searched):
                        profile.local_dim = searched[0]
                    else:
                        profile.local_dim = None
                        barrier = True
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["unbind", "chunk", "split"]:
                if "dim" in kwargs:
                    dim = kwargs["dim"]
                elif len(args) > 1:
                    if func == "unbind":
                        dim = args[1]
                    else:
                        dim = args[2]
                else:
                    dim = 0
                if last_local_dim is not None and dim != last_local_dim:
                    barrier = False
                    if dim > last_local_dim or func in ["chunk", "split"]:
                        profile.local_dim = last_local_dim
                    else:
                        profile.local_dim = last_local_dim - 1
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["max", "min", "any", "all", "argmax", "argmin"]:
                if "dim" in kwargs:
                    dim = kwargs["dim"]
                elif len(args) > 1:
                    dim = args[1]
                else:
                    dim = None
                if last_local_dim is not None and dim is not None and dim != last_local_dim:
                    barrier = False
                    if "keepdim" in kwargs and kwargs["keepdim"] or len(args) > 2 and args[2] or dim > last_local_dim:
                        profile.local_dim = last_local_dim
                    else:
                        profile.local_dim = last_local_dim - 1
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func.startswith(("permute")):
                if last_local_dim is not None:
                    barrier = False
                    if len(args) == 2 and isinstance(args[1], list):
                        indices = args[1]
                    else:
                        indices = args[1:]
                    searched = np.nonzero(
                        np.arange(len(input_shapes[0]))[indices] == last_local_dim)[0]
                    if len(searched):
                        profile.local_dim = searched[0]
                    else:
                        profile.local_dim = last_local_dim
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func.startswith((
                "conv_transpose2d"
            )): # Convolution operations
                profile.hook_kwargs = {
                    "idx": idx, "conv": True, "unconv": True,
                    "barrier": False, "profile": profile}
                profile.local_dim = len(input_shapes[0]) - 1
            elif func.startswith((
                "conv", "max_pool", "avg_pool",
            )): # Convolution operations
                profile.hook_kwargs = {
                    "idx": idx, "conv": True,
                    "barrier": False, "profile": profile}
                if func.startswith(("conv")):
                    kernel_size = args[1].shape[-1]
                elif func.startswith(("max_pool", "avg_pool")):
                    if "kernel_size" in kwargs:
                        kernel_size = kwargs["kernel_size"]
                    else:
                        kernel_size = args[1]
                kernel_size = kernel_size[0] if isinstance(kernel_size, Iterable) else kernel_size
                if kernel_size > 1:
                    profile.local_dim = len(input_shapes[0]) - 1
                else:
                    profile.local_dim = last_local_dim
            # elif func.startswith((
            #     "adaptive_avg_pool2d", "adaptive_max_pool2d"
            # )):
            #     profile.local_dim = last_local_dim
            #     profile.hook_kwargs = {
            #         "idx": idx, "conv": False,
            #         "barrier": barrier, "profile": profile}
            elif func.startswith((
                "bmm",
            )): # Convolution operations
                assert len(input_shapes) == 2, str(profile)
                if profile_result.profile[profile.input_from[0]].local_dim != len(input_shapes[0]) - 1 and \
                    profile_result.profile[profile.input_from[1]].local_dim != len(input_shapes[1]) - 2:
                    profile.local_dim = last_local_dim
                    barrier = False
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
            elif func.startswith((
                "softmax",
            )):
                if len(args) > 1:
                    dim = args[1]
                else:
                    dim = kwargs["dim"]
                if dim == -1:
                    dim = len(input_shapes[0]) - 1
                if last_local_dim is not None and dim != last_local_dim:
                    profile.local_dim = last_local_dim
                    barrier = False
                else:
                    profile.local_dim = None
                    barrier = True
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
            elif func.startswith(("linear")):
                # linear only applies to the last dim; if local dim is not the last dim, the locality remains
                if last_local_dim is not None and last_local_dim != len(input_shapes[0]) - 1:
                    barrier = False
                    profile.local_dim = last_local_dim
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
            elif func in ["shape", "dim"]:
                profile.local_dim = None
            else:
                # Operation that does not support offloading.
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": True, "profile": profile}
                profile.local_dim = None   # Operations that destroyed locality

            if func.startswith(("conv", "max_pool", "avg_pool")):
                if func.startswith(("conv")):
                    if "stride" in kwargs:
                        stride = kwargs["stride"]
                    else:
                        stride = args[3]
                    if "padding" in kwargs:
                        padding = kwargs["padding"]
                    else:
                        padding = args[4]
                    kernel_size = args[1].shape[-1]
                elif func.startswith(("max_pool", "avg_pool")):
                    if "stride" in kwargs:
                        stride = kwargs["stride"]
                    else:
                        stride = args[2]
                    if "padding" in kwargs:
                        padding = kwargs["padding"]
                    else:
                        padding = args[3]
                    if "kernel_size" in kwargs:
                        kernel_size = kwargs["kernel_size"]
                    else:
                        kernel_size = args[1]
                else:
                    raise RuntimeError(str(profile))
                padding = padding if isinstance(padding, Iterable) else [padding, padding]
                kernel_size = kernel_size if isinstance(kernel_size, Iterable) else [kernel_size, kernel_size]
                stride = stride if isinstance(stride, Iterable) else [stride, stride]
                profile.kernel_size, profile.stride, profile.padding = kernel_size, stride, padding
            if idx >= 462 and func != "_end" and self.model_name.startswith("Model"): # For kapao
                profile.local_dim = 0   # SPSO-GA for kapao
            if profile.excluded:
                profile.hook_kwargs["barrier"] = False
        # if self.model_name.startswith("Model"): # For kapao
        #     profile_result.profile[357].align_shape = True
        #     profile_result.profile[388].align_shape = True
        #     profile_result.profile[419].align_shape = True
        #     profile_result.profile[450].align_shape = True
