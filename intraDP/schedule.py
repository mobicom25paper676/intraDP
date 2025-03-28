from typing import Dict, List, Iterable
from collections import OrderedDict
import maxflow
import numpy as np
import math
import time
import torch
from scipy import optimize
from tqdm import tqdm
import itertools
import sys
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

from .hook_tensor import OffloadProfile, TorchOPProfile
from .test_asyncio import bw_count_size

def slice_shapes(shapes: List[torch.Size], sender_rate, recver_rate, diff, offset, order=0, dim=-1, constraint=[]):
    # order == 0: sender keep [0:x], send [x:y]
    # order == 1: sender keep [y:], send [x:y]
    sender_rate = np.around(sender_rate, decimals=2)
    recver_rate = np.around(recver_rate, decimals=2)
    diff = np.around(diff, decimals=2)
    assert sender_rate >= diff
    assert sender_rate + recver_rate == 1.
    def wrap_constraint(d):
        if constraint:
            kernel_size, stride, padding = constraint
            d = np.ceil((d - kernel_size + padding * 2)
                         / stride) * stride + kernel_size - padding * 2
            d += (4-d % 4) if d % 4 > 0 else 0  # Optimize cudnn
            assert d % 4 == 0
        return d
    def wrap_dim(_slice):
        if dim == -1 or dim == len(shape) - 1:
            if _slice is ...:
                return (...)
            return (..., _slice)
        elif dim == 0:
            if _slice is ...:
                return (...)
            return (_slice, ...)
        elif dim > 0:
            return tuple([slice(None)]*dim + [_slice] + [slice(None)]*(len(shape)-1 - dim))
        else:
            return tuple([slice(None)]*(len(shape)-1 + dim) + [_slice] + [slice(None)]*(- dim))
    def _slice_shape(shape: torch.Size):
        dim_len = shape[dim]
        init_sender_size = int(np.around(sender_rate * dim_len))
        init_recver_size = int(np.around(recver_rate * dim_len))
        if diff == 1.:
            send_slice = (...)
            send_keep_slice = slice(0)
            recv_keep_slice = slice(0)
        else:
            sender_keep_size = int(wrap_constraint(np.around((sender_rate - diff) * dim_len) + offset))
            next_recver_size = int(wrap_constraint(np.around((recver_rate + diff) * dim_len) + offset))
            recver_keep_size = int(np.around(recver_rate * dim_len))
            send_size = next_recver_size - recver_keep_size
            if order == 0:
                send_slice = slice(init_sender_size - send_size, init_sender_size)
                send_keep_slice = slice(0, sender_keep_size)
                if recver_keep_size == 0:
                    recv_keep_slice = slice(None, 0)
                else:
                    recv_keep_slice = slice(-recver_keep_size, None)
            else:
                if sender_keep_size == 0:
                    send_slice = slice(-init_sender_size, None)
                    send_keep_slice = slice(None, 0)
                else:
                    send_slice = slice(-init_sender_size, -sender_keep_size)
                    send_keep_slice = slice(-sender_keep_size, None)
                recv_keep_slice = slice(0, recver_keep_size)
            if send_size >= dim_len:
                raise ValueError(f"send_size {send_size} >= dim_len {dim_len}")
        return wrap_dim(send_slice), wrap_dim(send_keep_slice), wrap_dim(recv_keep_slice)
    send_slice = []
    send_keep_slice = []
    recv_keep_slice = []
    for shape in shapes:
        ret = _slice_shape(shape)
        send_slice.append(ret[0])
        send_keep_slice.append(ret[1])
        recv_keep_slice.append(ret[2])
    return send_slice, send_keep_slice, recv_keep_slice


import numba
@numba.jit
def inner_comp(transmit_indices, robot_comp_time, server_comp_time, client_send_time, server_send_time, robot_to_server_mask, server_to_robot_mask):
    robot_time = 0.
    robot_send_finish_time = 0.
    server_time = 0.
    server_send_finish_time = 0.
    for idx, _robot_comp_time, _server_comp_time in zip(transmit_indices,
                    np.split(robot_comp_time, transmit_indices + 1),
                    np.split(server_comp_time, transmit_indices + 1)):
        robot_time += _robot_comp_time.sum()
        server_time += _server_comp_time.sum()
        if robot_to_server_mask[idx]:
            if robot_time > robot_send_finish_time: # transmission already finished
                robot_send_finish_time = robot_time + client_send_time[idx]
            else:   # transmission queued
                robot_send_finish_time += client_send_time[idx]
            server_time = max(server_time, robot_send_finish_time)
        if server_to_robot_mask[idx]:
            if server_time > server_send_finish_time:
                server_send_finish_time = server_time + server_send_time[idx]
            else:
                server_send_finish_time += server_send_time[idx]
            robot_time = max(robot_time, server_send_finish_time)
    return robot_time

def inner_comp_no_numba(transmit_indices, robot_comp_time, server_comp_time, client_send_time, server_send_time, robot_to_server_mask, server_to_robot_mask):
    robot_time = 0.
    robot_send_finish_time = 0.
    server_time = 0.
    server_send_finish_time = 0.
    for idx, _robot_comp_time, _server_comp_time in zip(transmit_indices,
                    np.split(robot_comp_time, transmit_indices + 1),
                    np.split(server_comp_time, transmit_indices + 1)):
        robot_time += _robot_comp_time.sum()
        server_time += _server_comp_time.sum()
        if robot_to_server_mask[idx]:
            if robot_time > robot_send_finish_time:
                robot_send_finish_time = robot_time + client_send_time[idx]
            else:
                robot_send_finish_time += client_send_time[idx]
            server_time = max(server_time, robot_send_finish_time)
        if server_to_robot_mask[idx]:
            if server_time > server_send_finish_time:
                server_send_finish_time = server_time + server_send_time[idx]
            else:
                server_send_finish_time += server_send_time[idx]
            robot_time = max(robot_time, server_send_finish_time)
    return robot_time

class ops_info:
    def __init__(self):
        self.ops_num = 0
        self.robot_ops_time = []
        self.server_ops_time = []
        self.input_data = []
        self.transmit_data = []
        self.dependency = []
        self.conv = []
        self.kernel_size = []
        self.stride = []
        self.padding = []
        self.output_shapes = []
        self.dim_size = []
        self.masked = []
        self.excluded = []
        self.intraDP_layers: np.ndarray = []
        self.barrier = []
        self.local_dim = []
        self.align_shape = []
    
    def update_computation_time(self, robot_profile_result: OffloadProfile,
                                 server_profile_result: OffloadProfile):
        for robot_profile, server_profile in zip(robot_profile_result.profile.values(),
                                                 server_profile_result.profile.values()):
            self.robot_ops_time.append(robot_profile.ops_time)
            self.server_ops_time.append(server_profile.ops_time)
            self.input_data.append(robot_profile.input_size)
            self.transmit_data.append(robot_profile.output_size)
            self.dependency.append(
                [robot_profile.idx, robot_profile.input_from, robot_profile.output_to])
            self.masked.append(robot_profile.masked)
            self.excluded.append(robot_profile.excluded)
            self.conv.append(robot_profile.hook_kwargs.get("conv", False))
            self.output_shapes.append(robot_profile.output_shapes)
            self.local_dim.append(robot_profile.local_dim)
            if self.conv[-1]:
                args, kwargs = server_profile.func_args
                if server_profile.func_name.startswith(("conv")):
                    if "stride" in kwargs:
                        stride = kwargs["stride"]
                    else:
                        stride = args[3]
                    if "padding" in kwargs:
                        padding = kwargs["padding"]
                    else:
                        padding = args[4]
                    kernel_size = args[1].shape[-1]
                elif server_profile.func_name.startswith(("max_pool", "avg_pool")):
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
                    raise RuntimeError(str(server_profile) + str(robot_profile))
                if isinstance(kernel_size, Iterable):
                    kernel_size = list(kernel_size)[-1]
                if isinstance(stride, Iterable):
                    stride = list(stride)[-1]
                if isinstance(padding, Iterable):
                    padding = list(padding)[-1]
                if "unconv" not in robot_profile.hook_kwargs:
                    self.kernel_size.append(kernel_size)
                    self.stride.append(stride)
                else:
                    self.kernel_size.append(-kernel_size)
                    self.stride.append(-stride)
                    
                self.padding.append(padding)
                self.dim_size.append(robot_profile.input_shapes[0][-1])
            else:
                self.kernel_size.append(0)
                self.stride.append(0)
                self.padding.append(0)
                self.dim_size.append(1)
            if server_profile.local_dim is not None:
                dim = server_profile.local_dim
                if len(server_profile.output_shapes) > 0:
                    can_mix = False
                    for shape in server_profile.output_shapes:
                        if dim < len(shape):
                            can_mix = max(can_mix, shape[dim] > 2)
                else:
                    can_mix = np.any([self.intraDP_layers[i] for i in server_profile.input_from])
            else:
                can_mix = False
            if can_mix:
                self.intraDP_layers.append(server_profile.local_dim is not None)
            else:
                self.intraDP_layers.append(False)
            # self.intraDP_layers.append(not robot_profile.hook_kwargs["barrier"])
            self.barrier.append(server_profile.local_dim is not None)
            if "align_shape" in server_profile.hook_kwargs and server_profile.hook_kwargs["align_shape"]:
                self.align_shape.append(True)
            else:
                self.align_shape.append(False)
        self.robot_ops_time = np.array(self.robot_ops_time)
        self.server_ops_time = np.array(self.server_ops_time)
        self.transmit_data = np.array(self.transmit_data)
        self.ops_num = len(robot_profile_result.profile)
        self.barrier = np.array(self.barrier)
        self.intraDP_layers = np.array(self.intraDP_layers)
        self.conv = np.array(self.conv)
        self.kernel_size = np.array(self.kernel_size)
        self.dim_size = np.array(self.dim_size)
        self.dim_size[0] = robot_profile_result.profile[0].input_shapes[0][-1]
        self.align_shape = np.array(self.align_shape)
        self.excluded = np.array(self.excluded)
        assert self.ops_num == len(server_profile_result.profile)

class MyBounds(object):
    def __init__(self, xmax=[2.0], xmin=[-2.0]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


class Graph:
    def __init__(self, num_nodes):
        self.graph = maxflow.Graph[float](num_nodes, num_nodes)
        self.nodes = self.graph.add_nodes(num_nodes)

    def add_edge(self, u, v, capacity):
        # print(f"add edge {u} -> {v} with {capacity}")
        self.graph.add_edge(self.nodes[u], self.nodes[v], capacity, 0)

    def min_cut(self, source, sink):
        self.graph.add_tedge(self.nodes[source], float('inf'), 0)
        self.graph.add_tedge(self.nodes[sink], 0, float('inf'))

        max_flow_value = self.graph.maxflow()
        min_cut_plan = self.graph.get_grid_segments(self.nodes)

        # print(f"min_cut_plan {min_cut_plan} with value {max_flow_value}")
        return max_flow_value, min_cut_plan

class IDP_scheduler:
    def __init__(self, parallel_approach = "all", required_latency = -1.) -> None:
        self.parallel_approach = parallel_approach
        self.robot_ops = None
        self.server_ops = None
        self.max_bw = 30
        self.min_bw = 0 # TODO
        self.bw_step = 2
        self.estimated_bw = 50
        self.total_local_time = 0
        self.SPSO-GAed_bw = None
        self.plan_idx = None
        self.plan: Dict[int, int] = OrderedDict()
        self.graph_plan: Dict[int, Dict[str, list]] = OrderedDict()
        self.client_plans: Dict[int, Dict[str, list]] = OrderedDict()
        self.server_plans: Dict[int, Dict[str, list]] = OrderedDict()
        self.required_latency = required_latency
        self.info = ops_info()
        self.SPSO-GA_min_cut = 5 # SPSO-GA for correct result

        self.robot_size_to_loads_time: np.poly1d = None
        self.server_size_to_loads_time: np.poly1d = None
        self.robot_size_to_dumps_time: np.poly1d = None
        self.server_size_to_dumps_time: np.poly1d = None
        self.robot_cut_cost_time = 0.
        self.robot_cat_cost_time = 0.
        self.server_cut_cost_time = 0.
        self.server_cat_cost_time = 0.
        self.profiles: List[TorchOPProfile] = None
        
        # intraDP
        self.random_times = 50
        self.min_threshold = 1e-02
        self.intraDP_result_x = None
        self.intraDP_estimated_time = None
        self.intraDP_estimated_time = float("inf")
        np.random.seed(42)
    
    def estimate_transmission_time(self, transmit_data, bandwidth,place):
        if place == "client":
            return transmit_data/1e6/bandwidth + self.robot_size_to_dumps_time(transmit_data) + self.server_size_to_loads_time(transmit_data) + self.server_cat_cost_time
        elif place == "server":
            return transmit_data/1e6/bandwidth + self.server_size_to_dumps_time(transmit_data) + self.robot_size_to_loads_time(transmit_data) + self.robot_cat_cost_time


    def transfer_offload_to_slice(self, offload_plan,recv_plan):
        recv_first: Dict[int, slice] = OrderedDict()
        send_slice_plan: Dict[int, slice] = OrderedDict()
        send_keep_slice_plan: Dict[int, slice] = OrderedDict()
        recv_keep_slice_plan: Dict[int, slice] = OrderedDict()
        cat_order_plan: Dict[int, slice] = OrderedDict()
        cat_dim_plan: Dict[int, slice] = OrderedDict()
        for i in range(self.info.ops_num):
            offload = offload_plan[i]
            recv = recv_plan[i]
            recv_first[i] = False
            if offload == True:
                send_slice_plan[i] = [...] # 等效于[...]
                send_keep_slice_plan[i] = [slice(0)]
            if recv == True:
                recv_keep_slice_plan[i] = [slice(0)]
                cat_order_plan[i] = 0    
                cat_dim_plan[i] = -1   

        return recv_first, send_slice_plan, send_keep_slice_plan, recv_keep_slice_plan, cat_order_plan, cat_dim_plan

    def transfer_min_cut_to_skip_offload(self, min_cut_plan):
        skip_plan = min_cut_plan[:self.info.ops_num]
        skip_plan[-1] = False
        offload_plan = []
        recv_plan = []
        for i, node in enumerate(self.info.dependency):
            recv = False
            offload = False
            # print(i,self.dependency[i],self.robot_ops[i][1],self.server_ops[i][1],self.robot_ops[i][2]/1024/1024)
            if skip_plan[i]:
                for each in node[2]:
                    if skip_plan[each] == False:
                        recv = True
                if i == self.info.ops_num - 1:
                    recv = True
            else:
                for each in node[2]:
                    if skip_plan[each]:
                        offload = True
            offload_plan.append(offload)
            recv_plan.append(recv)
        return skip_plan, offload_plan, recv_plan

    def generate_skip_offload_plan(self, bandwidth, factor = 1.):
        # mincut schdule
        num_nodes = self.info.ops_num
        for node in self.info.dependency:
            if len(node[2]) > 1:
                num_nodes += 1
        source = num_nodes
        sink = num_nodes + 1
        num_nodes = num_nodes + 2
        graph = Graph(num_nodes)
        tag_idx = self.info.ops_num
        for i, node in enumerate(self.info.dependency):
            if i == len(self.info.dependency)-1:
                graph.add_edge(len(self.info.dependency)-1, sink, float("inf"))
            else:
                graph.add_edge(node[0], sink, self.info.robot_ops_time[i]*factor)
            transmit_data = self.info.transmit_data[i]
            if i == 0:
                graph.add_edge(source, 0, float("inf"))
            else:
                graph.add_edge(source, node[0], self.info.server_ops_time[i])
            if len(node[2]) == 0:
                continue
            elif len(node[2]) == 1:
                if transmit_data <= bw_count_size or self.info.masked[i] or self.info.excluded[i]:
                    graph.add_edge(node[0], node[2][0], np.inf)
                else:
                    graph.add_edge(node[0], node[2][0], self.estimate_transmission_time(transmit_data,bandwidth,"client"))
            else:
                if transmit_data <= bw_count_size or self.info.masked[i] or self.info.excluded[i]:
                    graph.add_edge(node[0], tag_idx, np.inf)
                else:
                    graph.add_edge(node[0], tag_idx, self.estimate_transmission_time(transmit_data,bandwidth,"server"))
                for each in node[2]:
                    graph.add_edge(tag_idx, each, float("inf"))
                tag_idx += 1
        value, min_cut_plan = graph.min_cut(source, sink)
        skip_plan, offload_plan, recv_plan = self.transfer_min_cut_to_skip_offload(min_cut_plan)
        estimated_time = 0
        for i in range(self.info.ops_num):
            if skip_plan[i]:
                estimated_time += self.info.server_ops_time[i]
            else:
                estimated_time += self.info.robot_ops_time[i]
            transmit_data = self.info.transmit_data[i]
            if offload_plan[i]:
                estimated_time += self.estimate_transmission_time(transmit_data,bandwidth,"client")
            if recv_plan[i]:
                estimated_time += self.estimate_transmission_time(transmit_data,bandwidth,"server")

        return estimated_time, np.array(skip_plan), np.array(offload_plan), np.array(recv_plan)
    
    def generate_select_plan(self, bandwidth):
        if self.required_latency > 0:
            bandwidth *= 2
        estimated_time, skip_plan, offload_plan, recv_plan = self.generate_skip_offload_plan(bandwidth)
        # replace with total local computation
        if estimated_time > self.total_local_time:
            skip_plan = self.total_local_skip_plan
            offload_plan = self.total_local_offload_plan
            recv_plan = self.total_local_recv_plan
            estimated_time = self.total_local_time
        # if self.required_latency < 0:
        #     estimated_time, skip_plan, offload_plan, recv_plan = self.generate_skip_offload_plan(bandwidth)
        #     # replace with total local computation
        #     if estimated_time > self.total_local_time * 0.9:
        #         skip_plan = self.total_local_skip_plan
        #         offload_plan = self.total_local_offload_plan
        #         recv_plan = self.total_local_recv_plan
        #         estimated_time = self.total_local_time
        #     #     print("replace with total local plan")
        #     # else:
        #     #     print("partition plan")
        # else:
        #     def search(bandwidth, factor, step=1.):
        #         last_estimated_time = None
        #         last_skip_plan = None
        #         last_offload_plan = None
        #         last_recv_plan = None
        #         while True:
        #             estimated_time, skip_plan, offload_plan, recv_plan =\
        #                     self.generate_skip_offload_plan(bandwidth, factor)
        #             # print(f"factor {factor} estimated_time {estimated_time}")
        #             if estimated_time > self.required_latency:
        #                 if last_skip_plan is None:
        #                     last_estimated_time = estimated_time
        #                     last_skip_plan = skip_plan
        #                     last_offload_plan = offload_plan
        #                     last_recv_plan = recv_plan

        #                 break
        #             factor += step
        #             last_estimated_time = estimated_time
        #             last_skip_plan = skip_plan
        #             last_offload_plan = offload_plan
        #             last_recv_plan = recv_plan

        #             # no plan better than total offload
        #             if all(value == True for value in last_skip_plan[1:-1]):
        #                 break
        #         return last_estimated_time, last_skip_plan, last_offload_plan, last_recv_plan, factor
        #     _, _, _, _, start_factor = search(bandwidth, factor=1., step=5)
        #     _, _, _, _, start_factor = search(bandwidth, factor=start_factor-5, step=1)
        #     _, _, _, _, start_factor = search(bandwidth, factor=start_factor-1, step=0.5)
        #     _, _, _, _, start_factor = search(bandwidth, factor=start_factor-0.5, step=0.05)
        #     estimated_time, last_skip_plan, last_offload_plan, last_recv_plan, _ = search(
        #         bandwidth, factor=start_factor-0.5, step=0.02)
            
        #     # replace with total local computation
        #     if estimated_time > self.total_local_time and False:
        #         skip_plan = self.total_local_skip_plan
        #         offload_plan = self.total_local_offload_plan
        #         recv_plan = self.total_local_recv_plan
        #         estimated_time = self.total_local_time
        #     else:
        #         skip_plan = last_skip_plan
        #         offload_plan = last_offload_plan
        #         recv_plan = last_recv_plan
        #     # print(f"bw: {bandwidth} estimated_time {last_estimated_time}")
        
        skip_plan = np.array(skip_plan)
        offload_plan = np.array(offload_plan)
        recv_plan = np.array(recv_plan)
        recv_first, send_slice_plan, send_keep_slice_plan, recv_keep_slice_plan, cat_order_plan, cat_dim_plan  = self.transfer_offload_to_slice(offload_plan,recv_plan)

        client_plan = {"skip":skip_plan, "offload": offload_plan, "recv_first": recv_first, "send_slice": send_slice_plan, "send_keep_slice": send_keep_slice_plan, "recv": recv_plan, "recv_keep_slice": recv_keep_slice_plan, "cat_order":cat_order_plan, "cat_dim":cat_dim_plan, "est_time":estimated_time}
        
        return client_plan
    
    def transfer_skip_to_greedy(self, skip_plan, bandwidth):
        estimated_time = 0.
        offload_plan = []
        recv_plan = []
        for i, node in enumerate(self.info.dependency):
            recv = False
            offload = False
            if skip_plan[i]:
                for each in node[2]:
                    if skip_plan[each] == False:
                        recv = True
            else:
                for each in node[2]:
                    if skip_plan[each]:
                        offload = True
            offload_plan.append(offload)
            recv_plan.append(recv)
            
            # compute estimates
            if skip_plan[i]:
                estimated_time += self.info.server_ops_time[i]
            else:
                estimated_time += self.info.robot_ops_time[i]
            transmit_data = self.info.transmit_data[i]
            if offload_plan[i]:
                estimated_time += self.estimate_transmission_time(transmit_data,bandwidth,"client")
            if recv_plan[i]:
                estimated_time += self.estimate_transmission_time(transmit_data,bandwidth,"server")

        return estimated_time, offload_plan, recv_plan
    
    def generate_greedy_plan(self, bandwidth):
        last_skip_plan = None
        last_offload_plan = None
        last_recv_plan = None
        skip_plan = [False for _ in range(self.info.ops_num)]
        estimated_time, offload_plan, recv_plan = self.transfer_skip_to_greedy(skip_plan, bandwidth)
        # self.required_latency = 0.5
        assert self.required_latency > 0
        for i in range(self.info.ops_num-2):
            skip_plan[self.info.ops_num-i-2] = True
            estimated_time, offload_plan, recv_plan = self.transfer_skip_to_greedy(skip_plan, bandwidth) 
            if estimated_time > self.required_latency:
                    # replace with total local computation
                if last_skip_plan is None:
                    last_estimated_time = estimated_time
                    last_skip_plan = np.array(skip_plan)
                    last_offload_plan = np.array(offload_plan)
                    last_recv_plan = np.array(recv_plan)

                break
            last_estimated_time = estimated_time
            last_skip_plan = np.array(skip_plan)
            last_offload_plan = np.array(offload_plan)
            last_recv_plan = np.array(recv_plan)
        
        if estimated_time > self.total_local_time:
            skip_plan = self.total_local_skip_plan
            offload_plan = self.total_local_offload_plan
            recv_plan = self.total_local_recv_plan
            estimated_time = self.total_local_time
        else:
            skip_plan = np.array(last_skip_plan)
            offload_plan = np.array(last_offload_plan)
            recv_plan = np.array(last_recv_plan)
            estimated_time = last_estimated_time
        recv_first, send_slice_plan, send_keep_slice_plan, recv_keep_slice_plan, cat_order_plan, cat_dim_plan  = self.transfer_offload_to_slice(offload_plan,recv_plan)

        client_plan = {"skip":skip_plan, "offload": offload_plan, "recv_first": recv_first, "send_slice": send_slice_plan, "send_keep_slice": send_keep_slice_plan, "recv": recv_plan, "recv_keep_slice": recv_keep_slice_plan, "cat_order":cat_order_plan, "cat_dim":cat_dim_plan,"est_time":estimated_time}
        return client_plan

   
    def generate_tp_plan(self,bandwidth):
        # for client
        client_skip_plan = [False for _ in range(self.info.ops_num)]

        client_offload_plan = [False for _ in range(self.info.ops_num)]
        client_recv_first = [True for _ in range(self.info.ops_num)]
        client_send_slice_plan: Dict[int, slice] = OrderedDict()
        client_send_keep_slice_plan: Dict[int, slice] = OrderedDict()

        client_recv_plan = [False for _ in range(self.info.ops_num)]
        client_recv_keep_slice_plan: Dict[int, slice] = OrderedDict()
        client_cat_order_plan: Dict[int, slice] = OrderedDict()
        client_cat_dim_plan: Dict[int, slice] = OrderedDict()

        # for server
        server_skip_plan = [True for _ in range(self.info.ops_num)]

        server_offload_plan = [False for _ in range(self.info.ops_num)]
        server_recv_first = [True for _ in range(self.info.ops_num)]
        server_send_slice_plan: Dict[int, slice] = OrderedDict()
        server_send_keep_slice_plan: Dict[int, slice] = OrderedDict()
        
        server_recv_plan = [False for _ in range(self.info.ops_num)]
        server_recv_keep_slice_plan: Dict[int, slice] = OrderedDict()
        server_cat_order_plan: Dict[int, slice] = OrderedDict()
        server_cat_dim_plan: Dict[int, slice] = OrderedDict()
        
        need_tp_gather = False
        gather_at_robot = True
        tensor_at_robot = True
        for i in range(self.info.ops_num-1):
            if need_tp_gather:
                client_idx = int(self.info.output_shapes[i][0][-1] * robot_rate)
                server_idx = - self.info.output_shapes[i][0][-1] + client_idx
                if gather_at_robot:
                    # gather at robot
                    
                    # for client
                    client_recv_plan[i] = True
                    client_recv_keep_slice_plan.update({i:(..., slice(0, client_idx))})
                    client_cat_order_plan.update({i: 0})
                    client_cat_dim_plan.update({i: -1})

                    # for server
                    server_offload_plan[i] = True
                    server_recv_first[i] = False
                    server_send_slice_plan.update({i:(..., slice(server_idx, None))})
                    server_send_keep_slice_plan.update({i:(..., slice(0))})
                    
                    tensor_at_robot = True
                else:
                    # gather at sever

                    # for client
                    client_offload_plan[i] = True
                    client_recv_first[i] = False
                    client_send_slice_plan.update({i:(..., slice(0, client_idx))})
                    client_send_keep_slice_plan.update({i:(..., slice(0))})
                    
                    # for server
                    server_recv_plan[i] = True
                    server_recv_keep_slice_plan.update({i:(...,slice(server_idx, None))})
                    server_cat_order_plan.update({i:-1})
                    server_cat_dim_plan.update({i: -1})
                    
                    tensor_at_robot = False
                
                need_tp_gather = False

            if self.info.conv[i+1]:
                # enable tp for each convlution layer
                if tensor_at_robot:
                    # last tensor gather at robot
                    transfer_rate = self.info.robot_ops_time[i+1]/(self.info.robot_ops_time[i+1]+self.info.server_ops_time[i+1]+ self.estimate_transmission_time(self.info.transmit_data[i],bandwidth,"client")) # place the transfer rate percent of tensor on server
                    robot_rate = 1- transfer_rate
                    client_idx = min(math.ceil(self.info.output_shapes[i][0][-1] * robot_rate) + max(self.info.kernel_size[i+1],self.SPSO-GA_min_cut),self.info.output_shapes[i][0][-1])
                    server_idx = max(math.floor(self.info.output_shapes[i][0][-1] * robot_rate) - max(self.info.kernel_size[i+1],self.SPSO-GA_min_cut),0)
                     
                    # for client
                    client_offload_plan[i] = True
                    client_recv_first[i] = True
                    client_send_slice_plan.update({i:(..., slice(server_idx,None))})
                    client_send_keep_slice_plan.update({i:(..., slice(0,client_idx))})

                    # for server
                    server_recv_plan[i] = True
                    server_recv_keep_slice_plan.update({i:(..., slice(0))})
                    server_cat_order_plan.update({i: -1})
                    server_cat_dim_plan.update({i: -1})

                    if (transfer_rate > 0.5 or self.info.masked[i] or i < self.info.ops_num-1 and self.info.masked[i+1]) and not i == self.info.ops_num-2:  # Temp SPSO-GA for branches
                        # gather at server
                        gather_at_robot = False
                    else:
                        # gather at robot
                        gather_at_robot = True 
                else:
                    # last tensor gather at server
                    transfer_rate = self.info.server_ops_time[i+1]/(self.info.robot_ops_time[i+1]+self.info.server_ops_time[i+1]+ self.estimate_transmission_time(self.info.transmit_data[i],bandwidth,"server")) # place the transfer rate percent of tensor on robot
                    robot_rate = transfer_rate
                    client_idx = min(math.ceil(self.info.output_shapes[i][0][-1] * robot_rate) + max(self.info.kernel_size[i+1],self.SPSO-GA_min_cut),self.info.output_shapes[i][0][-1])
                    server_idx = max(math.floor(self.info.output_shapes[i][0][-1] * robot_rate) - max(self.info.kernel_size[i+1],self.SPSO-GA_min_cut),0)
                    
                    # for client
                    client_recv_plan[i] = True
                    client_recv_keep_slice_plan.update({i:(..., slice(0))})
                    client_cat_order_plan.update({i: 0})
                    client_cat_dim_plan.update({i: -1})

                    # for server
                    server_offload_plan[i] = True
                    server_recv_first[i] = True
                    server_send_slice_plan.update({i:(..., slice(0, client_idx))})
                    server_send_keep_slice_plan.update({i:(..., slice(server_idx,None))})

                    if (transfer_rate > 0.5 and not (self.info.masked[i] or i < self.info.ops_num-1 and self.info.masked[i+1])) or i >= self.info.ops_num-2:
                        # gather at robot
                        gather_at_robot = True
                    else:
                        # gather at server
                        gather_at_robot = False 
                
                need_tp_gather = True
                server_skip_plan[i+1] = False
                
            else:
                if tensor_at_robot == False:
                    # tensor at server, and no tp
                    client_skip_plan[i+1] = True
                    server_skip_plan[i+1] = False
                else:
                    client_skip_plan[i+1] = False
                    server_skip_plan[i+1] = True

        
        if tensor_at_robot == False:
            # return at final op to robot

            # for client
            client_skip_plan[-1] = True
            client_recv_plan[-1] = True
            client_recv_keep_slice_plan.update({i:(slice(0))})
            client_cat_order_plan.update({i: 0})
            client_cat_dim_plan.update({i: -1})


            # for server
            server_skip_plan[-1] = False
            server_offload_plan[-1] = True
            server_recv_first[-1] = False
        last_idx = self.info.ops_num - 1
        server_send_slice_plan[last_idx] = (...)
        server_send_keep_slice_plan[last_idx] = slice(0)

        client_plan = {"skip":client_skip_plan, "offload": client_offload_plan, "recv_first": client_recv_first, "send_slice": client_send_slice_plan, "send_keep_slice": client_send_keep_slice_plan, "recv": client_recv_plan, "recv_keep_slice": client_recv_keep_slice_plan, "cat_order":client_cat_order_plan, "cat_dim":client_cat_dim_plan, "est_time": 0.}
        server_skip_plan[0] = True
        server_plan = {"skip":server_skip_plan, "offload": server_offload_plan, "recv_first": server_recv_first, "send_slice": server_send_slice_plan, "send_keep_slice": server_send_keep_slice_plan, "recv": server_recv_plan, "recv_keep_slice": server_recv_keep_slice_plan, "cat_order":server_cat_order_plan, "cat_dim":server_cat_dim_plan, "est_time": 0.}
        
        return client_plan, server_plan

    def transfer_client_plan_to_server_plan(self, client_plan):
        server_skip_plan = ~client_plan["skip"]
        server_offload_plan = client_plan["recv"]
        server_recv_plan = client_plan["offload"]
        recv_first, send_slice_plan, send_keep_slice_plan, recv_keep_slice_plan, cat_order_plan, cat_dim_plan  = self.transfer_offload_to_slice(server_offload_plan,server_recv_plan)

        server_plan = {"skip":server_skip_plan, "offload": server_offload_plan, "recv_first": recv_first, "send_slice": send_slice_plan, "send_keep_slice": send_keep_slice_plan, "recv": server_recv_plan, "recv_keep_slice": recv_keep_slice_plan, "cat_order":cat_order_plan, "cat_dim":cat_dim_plan,"est_time":client_plan["est_time"]}

        return server_plan
    
    def allocate_x_for_ops(self):
        self.x_for_ops = [None for _ in range(self.info.ops_num)]
        idx = 0
        last_transmit_data = [0. for _ in range(self.info.ops_num)]
        for i in range(self.info.ops_num):
            if i == 0 or self.x_for_ops[i] is not None:
                # op 0 is "_start", robot rate is 1.
                # self.x_for_ops[0] = None, skip
                continue
           
            if len(self.info.dependency[i][1]) == 1:
                # only one parent
                parent = self.info.dependency[i][1][0]
                if len(self.info.dependency[parent][2]) == 1:
                    #only one child
                    if self.info.intraDP_layers[parent] == self.info.intraDP_layers[i]:
                        if self.info.transmit_data[i] < last_transmit_data[parent]:
                            self.x_for_ops[i] = idx
                            idx += 1
                            last_transmit_data[i] = self.info.transmit_data[i]
                        else:
                            # impossible to tranbsmit
                            self.x_for_ops[i] = self.x_for_ops[parent]
                            last_transmit_data[i] = last_transmit_data[parent]
                    else:
                        self.x_for_ops[i] = idx
                        idx += 1
                        last_transmit_data[i] = self.info.transmit_data[i]
                else:
                    # multi child
                    total_transmit_data = 0.
                    for each in self.info.dependency[parent][2]:
                        self.x_for_ops[each] = idx
                        total_transmit_data += self.info.transmit_data[each]
                    
                    for each in self.info.dependency[parent][2]:
                        last_transmit_data[each] = total_transmit_data
                    idx += 1
            else:
                # multi parent
                self.x_for_ops[i] = idx
                idx += 1
                last_transmit_data = self.info.transmit_data[i]

        self.x_num = idx
        # print(f"intraDP layers {self.info.intraDP_layers}")
        # print(f"x num {self.x_num}")
        # print(f"x for ops {self.x_for_ops}")


    def get_actual_robot_rate(self,x,idx):
        if len(self.info.dependency[idx][1]) == 1:
            # only one parent
            parent = self.info.dependency[idx][1][0]
            if len(self.info.dependency[parent][2]) == 1:
                #only one child
                not_barrier = self.info.intraDP_layers[idx]
            else:
                # multi child, 只要有一个child强同步，全部child强同步
                not_barrier = True 
                for each in self.info.dependency[parent][2]:
                    if self.info.intraDP_layers[each] is False:
                        not_barrier = False
                        break
        else:
            # multi parent
            not_barrier = self.info.intraDP_layers[idx]
        
        if not_barrier:
            if x < self.min_threshold:
                return 0.
            elif x > 1. - self.min_threshold:
                return 1.
            else:
                return x
        else:
            return float(round(x))
    
    def get_offset_for_conv_send(self,idx):
        offset = 0
        x_idx = self.x_for_ops[idx]
        layers = [i for i,x in enumerate(self.x_for_ops) if x == x_idx ]
        for each in layers:
            if self.info.conv[each]:
                offset += int(self.info.kernel_size[each]/2)
        return offset

    
    def transfer_robot_rate_to_intraDP_plan(self,bandwidth):
        # for client
        client_skip_plan = [False for _ in range(self.info.ops_num)]

        client_offload_plan = [False for _ in range(self.info.ops_num)]
        client_recv_first = [True for _ in range(self.info.ops_num)]
        client_send_slice_plan: Dict[int, slice] = OrderedDict()
        client_send_keep_slice_plan: Dict[int, slice] = OrderedDict()

        client_recv_plan = [False for _ in range(self.info.ops_num)]
        client_recv_keep_slice_plan: Dict[int, slice] = OrderedDict()
        client_cat_order_plan: Dict[int, slice] = OrderedDict()
        client_cat_dim_plan: Dict[int, slice] = OrderedDict()

        # for server
        server_skip_plan = [False for _ in range(self.info.ops_num)]

        server_offload_plan = [False for _ in range(self.info.ops_num)]
        server_recv_first = [True for _ in range(self.info.ops_num)]
        server_send_slice_plan: Dict[int, slice] = OrderedDict()
        server_send_keep_slice_plan: Dict[int, slice] = OrderedDict()
        
        server_recv_plan = [False for _ in range(self.info.ops_num)]
        server_recv_keep_slice_plan: Dict[int, slice] = OrderedDict()
        server_cat_order_plan: Dict[int, slice] = OrderedDict()
        server_cat_dim_plan: Dict[int, slice] = OrderedDict()

        client_time = 0.
        server_time = 0.
        robot_rate = []
        for i in range(self.info.ops_num):
            if i == 0:
                current_robot_rate = 1.
            else:
                x_idx = self.x_for_ops[i]
                current_robot_rate = self.get_actual_robot_rate(self.intraDP_result_x[x_idx],i)
            current_idx = int(self.info.output_shapes[i][0][-1] * current_robot_rate)
            
            if i == self.info.ops_num -1:
                next_robot_rate = 1.
            else:
                x_idx = self.x_for_ops[self.info.dependency[i][2][0]]
                next_robot_rate = self.get_actual_robot_rate(self.intraDP_result_x[x_idx],self.info.dependency[i][2][0])
            next_idx = int(self.info.output_shapes[i][0][-1] * next_robot_rate)
            
            client_time += current_robot_rate*self.info.robot_ops_time[i]
            server_time += (1.-current_robot_rate)*self.info.server_ops_time[i]
            robot_rate.append(current_robot_rate)

            if next_idx > current_idx:
                # transmit to client
                transfer_rate = (next_idx - current_idx)/self.info.output_shapes[i][0][-1]
                transmit_time = self.estimate_transmission_time(self.info.transmit_data[i]*transfer_rate, bandwidth, "server")
                server_time += self.server_cut_cost_time
                client_time = max(client_time, server_time + transmit_time)
                
                # for client
                client_recv_plan[i] = True
                client_recv_keep_slice_plan.update({i:(..., slice(0,current_idx))})
                client_cat_order_plan.update({i: 0})
                client_cat_dim_plan.update({i: -1})
                
                # for server
                server_offload_plan[i] = True
                server_recv_first[i] = True
                if i == self.info.ops_num - 1:
                        send_idx = next_idx
                else:
                    offset = self.get_offset_for_conv_send(i)
                    send_idx = min(self.info.output_shapes[i][0][-1],next_idx + offset)

                server_send_slice_plan.update({i:(..., slice(current_idx,send_idx))})
                server_send_keep_slice_plan.update({i:(..., slice(next_idx,None))})
                

            if next_idx < current_idx:
                # transmit to server
                transfer_rate = (current_idx - next_idx)/self.info.output_shapes[i][0][-1]
                transmit_time = self.estimate_transmission_time(self.info.transmit_data[i]*transfer_rate, bandwidth, "client")
                server_time = max(server_time, client_time + transmit_time)
                client_time += self.robot_cut_cost_time
                
                # for client
                client_offload_plan[i] = True
                client_recv_first[i] = True
                if i == self.info.ops_num - 1:
                        send_idx = next_idx
                else:
                    offset = self.get_offset_for_conv_send(i)
                    send_idx = max(0,next_idx - offset)
                client_send_slice_plan.update({i:(..., slice(send_idx,current_idx))})
                client_send_keep_slice_plan.update({i:(..., slice(0,next_idx))})
                
                
                # for server
                server_recv_plan[i] = True
                server_recv_keep_slice_plan.update({i:(..., slice(current_idx,None))})
                server_cat_order_plan.update({i: -1})
                server_cat_dim_plan.update({i: -1})

            if current_robot_rate == 0.:
                client_skip_plan[i] = True
            elif current_robot_rate == 1.:
                server_skip_plan[i] = True
                         
        estimated_time = client_time

        client_plan = {"skip":client_skip_plan, "offload": client_offload_plan, "recv_first": client_recv_first, "send_slice": client_send_slice_plan, "send_keep_slice": client_send_keep_slice_plan, "recv": client_recv_plan, "recv_keep_slice": client_recv_keep_slice_plan, "cat_order":client_cat_order_plan, "cat_dim":client_cat_dim_plan, "est_time":estimated_time}

        server_plan = {"skip":server_skip_plan, "offload": server_offload_plan, "recv_first": server_recv_first, "send_slice": server_send_slice_plan, "send_keep_slice": server_send_keep_slice_plan, "recv": server_recv_plan, "recv_keep_slice": server_recv_keep_slice_plan, "cat_order":server_cat_order_plan, "cat_dim":server_cat_dim_plan, "est_time":estimated_time}


        print(f"bandwidth {bandwidth}")
        print(f"robot rate {robot_rate}")
        print(f"estimated time {estimated_time}")

        # traditional method
        transfer_time = self.estimate_transmission_time(self.info.transmit_data[0],bandwidth,"client") + self.estimate_transmission_time(self.info.transmit_data[-1],bandwidth,"server")
        all_offload_time = self.total_offload_time + transfer_time
        print(f"all offload {all_offload_time} {estimated_time/all_offload_time}")
        print(f"all local {self.total_local_time} {estimated_time/self.total_local_time}")
        print("#############################")
        
        _, unique_index, inverse = np.unique(x_idx, return_index=True, return_inverse=True)
        return client_plan, server_plan

    def generate_intraDP_plan_2(self, bandwidth, partial_steps=3, linspace_num=11, bandwidth_degrade=0.9):
        # mincut schdule
        x_idx = []
        for idx, input_from, _ in self.info.dependency:
            min_peer = idx
            for _idx in input_from:
                for __idx in self.info.dependency[_idx][2]:
                    if __idx < len(x_idx):
                        min_peer = min(min_peer, x_idx[__idx])
            x_idx.append(min_peer)
        _, unique_idx, inverse = np.unique(x_idx, return_index=True, return_inverse=True)
        ops_num = self.info.ops_num
        op_output_to_idx = np.zeros(ops_num, dtype=np.int32)
        for i, profile in enumerate(self.profiles):
            for j in profile.output_to:
                if not self.info.excluded[j]:
                    op_output_to_idx[i] = j
                    break
            else:
                op_output_to_idx[i] = i
        op_output_to_idx[-1] = -1

        all_data = self.info.transmit_data
        conv_num = self.info.conv.sum()
        row_nums = []
        for s in self.info.output_shapes:
            if len(s) > 0:
                row_nums.append(s[0][-1])
            else:
                row_nums.append(0)
        row_nums = np.array(row_nums)
        worst_transmit_size = self.info.input_data[0] + self.info.transmit_data[-1]
        op_indices = np.arange(ops_num)

        barriers = np.nonzero(~self.info.barrier & ~self.info.excluded)[0]
        def constraints(x: np.ndarray):
            # Constraint x to 0.0, 0.2..., 1.0 for non-barrier layers (already done by grid search)
            # Constraint barrier layers to 0.0, 1.0
            # Constraint barrier layers to 0.0, 1.0
            x = np.array(x)
            x[[0, -1]] = 1.
            x[barriers] = np.where(x[barriers] > 0.5, 1., 0.)
            # Constraint x that takes the same input to be equal (already done by fill_x_between_ops)
            return x

        def obj_func(x, no_numba=False):
            x = np.round(np.array(x), decimals=2)
            next_x = x[op_output_to_idx]
            transmit_indices = np.nonzero(x!=next_x)[0]
            robot_comp_x = np.max([x, np.where(x > 0, min_effective_x, 0)], axis=0)
            server_comp_x = np.max([1-x, np.where(x < 1, min_effective_x, 0)], axis=0)
            robot_comp_time = robot_comp_x * self.info.robot_ops_time
            server_comp_time = server_comp_x * self.info.server_ops_time

            if len(transmit_indices) > 0:
                robot_to_server_mask = x > next_x
                server_to_robot_mask = x < next_x
                # partial_idx = np.nonzero((x > 0) & (x < 1))
                # conv_rate = sum(conv_offset[partial_idx])
                server_comp_time = (1.-x) * self.info.server_ops_time
                transmit_rate = np.abs(x - next_x)
                # Compute offset
                partial_transmit_indices = np.nonzero(((x==0)|(x==1)) & (
                    x != next_x) & (next_x > 0) & (next_x < 1))[0]
                # whole_indices = np.nonzero(((x==0) & (next_x == 0))|((x==1) & (next_x == 1)))[0]
                # diff = whole_indices[None] - partial_transmit_indices[..., None]
                # _whole_indices = np.argmax(diff > 0, axis=-1)
                # # dist = whole_indices[_whole_indices] - partial_transmit_indices
                # indices = np.tile(op_indices, (len(partial_transmit_indices), 1))
                # covered_mask = (indices >= partial_transmit_indices[..., None]) & (
                #     indices <= whole_indices[_whole_indices][..., None])
                # covered_convs_num = np.sum(self.info.conv)
                # extra_row_num = dist / ops_num * conv_num # number of extra rows
                # extra_rate = covered_convs_num / 2 / row_nums[partial_transmit_indices]
                # transmit_rate[partial_transmit_indices] += extra_rate # emulate offset
                if np.any(transmit_rate[partial_transmit_indices] >= 1):
                    return np.inf   # offset causes the total transmitted
                                    # data to exceed whole data size
                                    # skip this partial transmission plan
                transmit_rate[partial_transmit_indices] = np.clip(
                    transmit_rate[partial_transmit_indices]*1.05 + 0.05, 0., 1.) # emulate offset

                partial_mask = (x>0) & (x < 1)
                # robot_comp_time[partial_mask] = np.clip(x[partial_mask], 0., 1.) * self.info.robot_ops_time[partial_mask]
                # server_comp_time[partial_mask] = np.clip((1-x[partial_mask]), 0., 1.) * self.info.server_ops_time[partial_mask]
                # robot_comp_time[partial_mask] *= 1.2
                # server_comp_time[partial_mask] *= 1.2
                transmit_data = all_data * transmit_rate
                if sum(transmit_data) >= 0.9 * worst_transmit_size:
                    return np.inf
                client_to_server_data = np.where(robot_to_server_mask, transmit_data, 0)
                server_to_client_data = np.where(server_to_robot_mask, transmit_data, 0)
                server_dumps_time = np.where(
                    server_to_robot_mask, self.server_size_to_dumps_time(server_to_client_data), 0.)
                server_loads_time = np.where(
                    robot_to_server_mask, self.server_size_to_loads_time(client_to_server_data), 0.)
                client_dumps_time = np.where(
                    robot_to_server_mask, self.robot_size_to_dumps_time(client_to_server_data), 0.)
                client_loads_time = np.where(
                    server_to_robot_mask, self.robot_size_to_loads_time(server_to_client_data), 0.)

                client_send_time = client_to_server_data / (bandwidth * bandwidth_degrade) / 1e6
                server_send_time = server_to_client_data / (bandwidth * bandwidth_degrade) / 1e6

                robot_comp_time += client_dumps_time + server_loads_time
                server_comp_time += server_dumps_time + client_loads_time

                if not no_numba:
                    robot_time = inner_comp(
                        transmit_indices, robot_comp_time, server_comp_time,
                        client_send_time, server_send_time,
                        robot_to_server_mask, server_to_robot_mask)
                else:
                    robot_time = inner_comp_no_numba(
                        transmit_indices, robot_comp_time, server_comp_time,
                        client_send_time, server_send_time,
                        robot_to_server_mask, server_to_robot_mask)
                robot_time += robot_comp_time[transmit_indices[-1]+1:].sum()
            else:
                robot_time = robot_comp_time.sum()
            return robot_time

        # baseline_time, baseline_x = self.generate_skip_offload_plan(bandwidth)[:2]
        baseline_x = np.zeros(ops_num)
        baseline_x[0] = 1.
        baseline_x[-1] = 1.
        baseline_time = (self.info.transmit_data[0] + self.info.transmit_data[-1]) / 1e6 / bandwidth + self.info.server_ops_time.sum() + self.info.robot_ops_time.sum() / 10. # Empirical 10% time for all offload
        baseline_x_local = np.ones(ops_num)
        baseline_time_local = self.info.robot_ops_time.sum()
        if bandwidth >= 1:
            num_nodes = self.info.ops_num
            for node in self.info.dependency:
                if len(node[2]) > 1:
                    num_nodes += 1
            source = num_nodes
            sink = num_nodes + 1
            num_nodes = num_nodes + 2
            all_ops = np.arange(self.info.ops_num)
            def search_for_min_cut(exclude_nodes, bandwidth):
                graph = Graph(num_nodes)
                tag_idx = self.info.ops_num
                for i, node in enumerate(self.info.dependency):
                    if i == len(self.info.dependency)-1:
                        graph.add_edge(len(self.info.dependency)-1, sink, float("inf"))
                    else:
                        graph.add_edge(node[0], sink, self.info.robot_ops_time[i])
                    transmit_data = self.info.transmit_data[i]
                    if i == 0:
                        graph.add_edge(source, 0, np.inf)
                    else:
                        graph.add_edge(source, node[0], self.info.server_ops_time[i])

                    if len(node[2]) == 0:
                        continue
                    elif len(node[2]) == 1:
                        if transmit_data <= bw_count_size or i in exclude_nodes or self.info.excluded[i] or self.info.masked[i]:
                            graph.add_edge(node[0], node[2][0], np.inf)
                        else:
                            graph.add_edge(node[0], node[2][0], self.estimate_transmission_time(transmit_data,bandwidth,"client"))
                    else:
                        if transmit_data <= bw_count_size or i in exclude_nodes or self.info.excluded[i] or self.info.masked[i]:
                            graph.add_edge(node[0], tag_idx, np.inf)
                        else:
                            graph.add_edge(node[0], tag_idx, self.estimate_transmission_time(transmit_data,bandwidth,"server"))
                        for each in node[2]:
                            graph.add_edge(tag_idx, each, float("inf"))
                        tag_idx += 1
                value, min_cut_bool = graph.min_cut(source, sink)
                min_cut_bool = min_cut_bool[:self.info.ops_num]
                first_segment = all_ops[~min_cut_bool]
                second_segment = all_ops[min_cut_bool]
                # nx.draw(G, pos=pos)
                # plt.show()
                min_cut_start = []
                min_cut_end = []
                for op in first_segment:
                    op = int(op)
                    isin = np.isin(np.array(self.info.dependency[op][2]), second_segment)
                    if np.any(isin):
                        min_cut_start.append(op)
                        min_cut_end.append(self.info.dependency[op][2][0])
                return min_cut_start, min_cut_end
            min_cut_starts = []
            min_cut_ends = []
            # barrier_idx = np.nonzero(self.info.barrier)[0]
            while len(min_cut_ends) < partial_steps:
                min_cut_start, min_cut_end = search_for_min_cut(min_cut_starts, bandwidth=10)
                min_cut_starts += min_cut_start
                min_cut_ends += min_cut_end
                min_cut_ends = np.array(min_cut_ends)
                min_cut_ends = min_cut_ends[~np.isin(min_cut_ends, barriers)].tolist()
            # baseline_time = obj_func(constraints(baseline_x.astype(float)))
            min_cut_ends += [1] # Should always consider offload at the start
            _min_cut_ends = []
            input_dim_len = self.profiles[0].input_shapes[0][-1]
            input_size = self.profiles[0].input_size
            min_effective_x = np.zeros(ops_num)
            for idx, profile in enumerate(self.profiles):
                local_dim = profile.local_dim
                if local_dim is not None:
                    _input_size = profile.input_size
                    min_effective_x[idx] = max(1 - _input_size / input_size - 0.2, 0)
                else:
                    min_effective_x[idx] = 1.
            for idx in min_cut_ends:
                input_idx = self.profiles[idx].input_from[0]
                if self.profiles[input_idx].local_dim is None or self.profiles[input_idx].input_shapes[0][self.profiles[input_idx].local_dim] < input_dim_len / 4:
                    continue
                elif self.profiles[input_idx].input_size > input_size * 1.5 and idx < 10 and idx > 0:
                    continue
                else:
                    _min_cut_ends.append(idx)
            min_cut_ends = _min_cut_ends
            varied_ops = np.unique(min_cut_ends)
            

            min_cut_possibles = [[0., 0.5, 0.6, 0.7, 1.0] for _ in varied_ops]
            # min_cut_possibles = [np.linspace(0, 1, linspace_num) for _ in varied_ops]
            min_cut_possibles = np.array(np.meshgrid(*min_cut_possibles)).T.reshape(-1, len(varied_ops))

            def fill_x_between_ops(x, ops_idx):
                for i, _ in enumerate(x):
                    if i not in ops_idx:
                        peers = []
                        [peers.extend(self.info.dependency[input_idx][2]) for input_idx in self.info.dependency[i][1]]
                        if not any(peer_idx in ops_idx for peer_idx in peers):
                            for input_idx in self.info.dependency[i][1]:
                                x[i] = min(x[i], x[input_idx])
                                for peer_idx in self.info.dependency[input_idx][2]:
                                    x[i] = min(x[i], x[peer_idx])
                        else:
                            for peer_idx in peers:
                                if peer_idx in ops_idx:
                                    x[i] = min(x[i], x[peer_idx])
                return x
            assigned_ops = varied_ops.tolist()
            optimal_val = np.inf
            optimal_x = np.ones(ops_num)
            
            with tqdm(total=len(min_cut_possibles)) as pbar:
                for min_cut_possible in min_cut_possibles:
                    x = np.ones(ops_num)
                    x[varied_ops] = min_cut_possible
                    x = fill_x_between_ops(x, assigned_ops)
                    x = constraints(x)
                    val = obj_func(x) + baseline_time_local / 10. # Empirical 10% extra time for offloading
                    if val < optimal_val:
                        optimal_val = val
                        optimal_x = x
                        pbar.set_description(f"bandwidth {int(bandwidth)}MB {optimal_val:.4f}/{baseline_time:.4f}")
                    pbar.update(1)
            x = optimal_x
            estimated_time = obj_func(constraints(x), no_numba=True)
            if estimated_time > baseline_time:
                x = baseline_x
                estimated_time = baseline_time
            if estimated_time > baseline_time_local:
                x = baseline_x_local
                estimated_time = baseline_time_local
        else:
            x = np.ones(ops_num)
            estimated_time = sum(self.info.robot_ops_time)

        while True:
            try:
                client_skip_plan = np.isclose(x, 0.)
                client_offload_plan = x > x[op_output_to_idx]
                client_recv_plan = x < x[op_output_to_idx]
                server_skip_plan = np.isclose(x, 1.)
                server_offload_plan = x < x[op_output_to_idx]
                server_recv_plan = x > x[op_output_to_idx]

                client_recv_first = [True for _ in range(self.info.ops_num)]
                client_send_slice_plan: Dict[int, slice] = OrderedDict()
                client_send_keep_slice_plan: Dict[int, slice] = OrderedDict()
                client_recv_keep_slice_plan: Dict[int, slice] = OrderedDict()
                client_cat_order_plan: Dict[int, slice] = OrderedDict()
                client_cat_dim_plan: Dict[int, slice] = OrderedDict()

                server_recv_first = [True for _ in range(self.info.ops_num)]
                server_send_slice_plan: Dict[int, slice] = OrderedDict()
                server_send_keep_slice_plan: Dict[int, slice] = OrderedDict()
                server_recv_keep_slice_plan: Dict[int, slice] = OrderedDict()
                server_cat_order_plan: Dict[int, slice] = OrderedDict()
                server_cat_dim_plan: Dict[int, slice] = OrderedDict()
                transmit_indices = np.nonzero(x != x[op_output_to_idx])[0]
                offsets = {}

                for idx in transmit_indices:
                    next_idx = op_output_to_idx[idx]
                    diff = np.abs(x[idx] - x[next_idx])
                    # idx to last idx to determine offset
                    descendant_idx = next_idx
                    dim = self.profiles[idx].local_dim
                    if dim is None:
                        dim = -1
                    offset = 0
                    size_shrink = 0
                    first_constraint_info = []
                    while descendant_idx > 0:
                        if self.info.conv[descendant_idx]:
                            offset += max(int(self.info.kernel_size[descendant_idx]/2 - 1) + self.info.stride[descendant_idx] - 1 - self.info.padding[descendant_idx], 0)
                            if not first_constraint_info:
                                first_constraint_info = [self.info.kernel_size[descendant_idx],
                                                        self.info.stride[descendant_idx],
                                                        self.info.padding[descendant_idx]]
                        if descendant_idx in transmit_indices:
                            # Should keep robot rate valid until next further_communication
                            break
                        descendant_idx = op_output_to_idx[descendant_idx]
                    if x[descendant_idx] in [0., 1.]: # all offload 
                        offset = 0
                    if offset % 2 > 0:
                        offset += 1
                    if x[idx] > x[next_idx]:   # robot to server
                        (client_send_slice_plan[idx],
                        client_send_keep_slice_plan[idx],
                        server_recv_keep_slice_plan[idx]) = slice_shapes(
                            self.info.output_shapes[idx], x[idx], 1-x[idx], diff, offset, order=0, dim=dim,
                            constraint=first_constraint_info)
                    else:   # server to robot
                        (server_send_slice_plan[idx],
                        server_send_keep_slice_plan[idx],
                        client_recv_keep_slice_plan[idx]) = slice_shapes(
                            self.info.output_shapes[idx], 1-x[idx], x[idx], diff, offset, order=1, dim=dim,
                            constraint=first_constraint_info)
                    offsets[idx] = offset

                    server_cat_order_plan[idx] = 1
                    client_cat_order_plan[idx] = 0
                    server_cat_dim_plan[idx] = dim
                    client_cat_dim_plan[idx] = dim
                align_shape = self.info.align_shape & (x > 0) & (x < 1)
                break
            except ValueError:
                if baseline_time_local > baseline_time:
                    x = baseline_x
                    estimated_time = baseline_time
                else:
                    x = baseline_x_local
                    estimated_time = baseline_time_local
                continue
        next_x = x[op_output_to_idx]
        transmit_indices = np.nonzero(x != next_x)[0]
        info = f"bandwidth {bandwidth}MB, estimated time {estimated_time:.4f}s, baseline time {baseline_time:.4f}s scheduled: "
        for idx in transmit_indices:
            info += f"{idx}:{self.profiles[idx].func_name} {x[idx]:.4f} -> {next_x[idx]:.4f} offset {offsets[idx]}; "
        print(info)
        
        transmission_time = all_data / bandwidth / 1024 / 1024

        client_skip_plan[0] = True
        client_skip_plan[-1] = True
        server_skip_plan[0] = True
        server_skip_plan[-1] = True

        client_plan = {"skip":client_skip_plan, "offload": client_offload_plan, "recv_first": client_recv_first, "send_slice": client_send_slice_plan, "send_keep_slice": client_send_keep_slice_plan, "recv": client_recv_plan, "recv_keep_slice": client_recv_keep_slice_plan, "cat_order":client_cat_order_plan, "cat_dim":client_cat_dim_plan, "est_time": estimated_time, "align_shape": align_shape * 1, "x": x, "comp_time": self.info.robot_ops_time, "transmission_time": transmission_time}

        server_plan = {"skip":server_skip_plan, "offload": server_offload_plan, "recv_first": server_recv_first, "send_slice": server_send_slice_plan, "send_keep_slice": server_send_keep_slice_plan, "recv": server_recv_plan, "recv_keep_slice": server_recv_keep_slice_plan, "cat_order":server_cat_order_plan, "cat_dim":server_cat_dim_plan, "est_time": estimated_time, "align_shape": align_shape * 2, "x": x, "comp_time": self.info.server_ops_time, "transmission_time": transmission_time}

        return client_plan, server_plan

    def generate_intraDP_plan_3(self, bandwidth, max_possible_poses_number = 3):
        # use min-cut to find possible poses, no diff to intraDP
        num_nodes = self.info.ops_num
        for node in self.info.dependency:
            if len(node[2]) > 1:
                num_nodes += 1
        source = num_nodes
        sink = num_nodes + 1
        num_nodes = num_nodes + 2
        all_ops = np.arange(self.info.ops_num)
        def search_for_min_cut(exclude_nodes, bandwidth):
            graph = Graph(num_nodes)
            tag_idx = self.info.ops_num
            for i, node in enumerate(self.info.dependency):
                if i == len(self.info.dependency)-1:
                    graph.add_edge(len(self.info.dependency)-1, sink, float("inf"))
                else:
                    graph.add_edge(node[0], sink, self.info.robot_ops_time[i])
                transmit_data = self.info.transmit_data[i]
                if i == 0:
                    graph.add_edge(source, 0, np.inf)
                else:
                    graph.add_edge(source, node[0], self.info.server_ops_time[i])

                if len(node[2]) == 0:
                    continue
                elif len(node[2]) == 1:
                    if transmit_data <= bw_count_size or i in exclude_nodes:
                        graph.add_edge(node[0], node[2][0], np.inf)
                    else:
                        graph.add_edge(node[0], node[2][0], self.estimate_transmission_time(transmit_data,bandwidth,"client"))
                else:
                    if transmit_data <= bw_count_size or i in exclude_nodes:
                        graph.add_edge(node[0], tag_idx, np.inf)
                    else:
                        graph.add_edge(node[0], tag_idx, self.estimate_transmission_time(transmit_data,bandwidth,"server"))
                    for each in node[2]:
                        graph.add_edge(tag_idx, each, float("inf"))
                    tag_idx += 1
            value, min_cut_bool = graph.min_cut(source, sink)
            min_cut_bool = min_cut_bool[:self.info.ops_num]
            first_segment = all_ops[~min_cut_bool]
            second_segment = all_ops[min_cut_bool]
            # nx.draw(G, pos=pos)
            # plt.show()
            min_cut_start = []
            min_cut_end = []
            for op in first_segment:
                op = int(op)
                isin = np.isin(np.array(self.info.dependency[op][2]), second_segment)
                if np.any(isin):
                    min_cut_start.append(op)
                    min_cut_end.append(self.info.dependency[op][2][0])
            return min_cut_start, min_cut_end
        min_cut_starts = []
        min_cut_ends = []
        barrier_idx = np.nonzero(self.info.barrier)[0]
        while len(min_cut_ends) < max_possible_poses_number:
            min_cut_start, min_cut_end = search_for_min_cut(min_cut_starts, bandwidth=10)
            min_cut_starts += min_cut_start
            min_cut_ends += min_cut_end
            min_cut_ends = np.array(min_cut_ends)
            min_cut_ends = min_cut_ends[~np.isin(min_cut_ends, barrier_idx)].tolist()
        min_cut_ends += [1] # Should always consider offload at the start
        varied_ops = np.setdiff1d(np.unique(min_cut_ends+barrier_idx.tolist()), [0,self.info.ops_num-1])
        # print(f"varied_ops {varied_ops}")
        
        # dfs to find best schdule plan        
        def generated_operation_num(idx, input_num):
            kernel_size = np.array(self.info.kernel_size[idx])
            padding = np.array(self.info.padding[idx])
            stride = np.array(self.info.stride[idx])

            return np.where(self.info.conv[idx], (input_num - kernel_size + 1 + 2 * padding) // stride, input_num)

        def required_operation_num(idx, output_num):
            kernel_size = np.array(self.info.kernel_size[idx])
            padding = np.array(self.info.padding[idx])
            stride = np.array(self.info.stride[idx])

            return np.where((self.info.conv[idx] & (output_num > 0)), 
                            (output_num * stride + kernel_size - 1 - 2 * padding),
                            output_num)

        def get_operation_distribution(params_x, dependency, num_nodes, max_operations, varied_ops):

            robot_operation_num = np.array(max_operations, dtype=np.int32)
            server_operation_num = np.zeros(num_nodes, dtype=np.int32)

            for i, node in enumerate(dependency):
                if i in varied_ops:
                    index = np.where(varied_ops == i)[0]
                    robot_operation_num[i] = params_x[index[0]]

                    for parent in node[1]:
                        server_operation_num[parent] = max(server_operation_num[parent], 
                                                        max_operations[parent] - robot_operation_num[parent])
                elif i == 0 or i == num_nodes - 1:
                    continue
                else:
                    parent_nodes = node[1]
                    input_num = np.min(robot_operation_num[parent_nodes]) if parent_nodes else 0
                    robot_operation_num[i] = generated_operation_num(i, input_num)

                server_operation_num[i] = max_operations[i] - robot_operation_num[i]

            for i in reversed(range(num_nodes)):
                node = self.info.dependency[i]
                for parent in node[1]:  
                    server_operation_num[parent] = max(server_operation_num[parent], 
                                                    required_operation_num(i, server_operation_num[i]))
            server_operation_num[0] = 0

            # for i in range(num_nodes):
            #     print(i,robot_operation_num[i], server_operation_num[i], max_operations[i])
            #     if robot_operation_num[i] + server_operation_num[i] > max_operations[i]:
            #         print(f"re-caculation!!!!")

            return robot_operation_num, server_operation_num


        def generate_send_slice(client_operation_num, server_operation_num):
            num_nodes = self.info.ops_num

            client_send_slice = np.zeros(num_nodes, dtype=int)
            server_send_slice = np.zeros(num_nodes, dtype=int)

            client_operation_num = np.array(client_operation_num, dtype=int)
            server_operation_num = np.array(server_operation_num, dtype=int)
            
            varied_ops_np = np.array(varied_ops.tolist() + [num_nodes - 1], dtype=int)

            for i in varied_ops_np:
                node = self.info.dependency[i]
                client_required_input = required_operation_num(i, client_operation_num[i])
                server_required_input = required_operation_num(i, server_operation_num[i])

                for parent in node[1]:
                    if self.info.barrier[parent]:
                        client_parent_output_slice = max_slices[parent] * client_operation_num[parent]
                        server_parent_output_slice = max_slices[parent] * server_operation_num[parent]
                    else:
                        client_parent_output_slice = client_operation_num[parent]
                        server_parent_output_slice = server_operation_num[parent]

                    if self.info.barrier[i]:
                        client_required_slice = max_slices[parent] * client_required_input
                        server_required_slice = max_slices[parent] * server_required_input
                    else:
                        client_required_slice = client_required_input
                        server_required_slice = server_required_input

                    # print(f"{i} {client_parent_output_slice} {client_required_slice} {server_parent_output_slice} {server_required_slice}")

                    client_deficit = client_required_slice - client_parent_output_slice
                    server_deficit = server_required_slice - server_parent_output_slice

                    if client_deficit > 0:
                        server_send_slice[parent] = max(client_deficit, server_send_slice[parent])
                    
                    if server_deficit > 0:
                        client_send_slice[parent] = max(server_deficit, client_send_slice[parent])
            return client_send_slice, server_send_slice

        def find_minimum(ranges, min_value = float('inf'),best_params = None, num_workers=None, task_batchsize=5000, step_start=4, step_length=1.2):
            """
            🚀 Parallel search for the minimum value of an objective function with real-time progress updates.
            Allows early termination if no better values are found.
            
            :param ranges: Search space ranges for each variable, e.g., [1000, 500, 2000]
            :param num_workers: Number of parallel workers (default: auto-detected)
            :param log_interval: How frequently to update progress in the console
            :return: (minimum_value, best_parameters)
            """
            def adjust_range(r, step_start, step_length):
                """调整搜索范围，确保两端点(0 和 r-1)存在, 并根据长度按对数关系调整步长。"""
                if r <= step_start:
                    return list(range(r))
                
                step = int(np.log(r) / np.log(step_length))  # 计算 logn(r)，得到步长
                adjusted_range = list(range(0, r, step))  # 生成按步长的序列
                
                if adjusted_range[-1] != r - 1:  # 确保最后一个元素是 r-1
                    adjusted_range.append(r - 1)
                
                return adjusted_range

            def objective_function(params_x):
                """Computes the objective function."""
                robot_operation_num, server_operation_num = get_operation_distribution(params_x, self.info.dependency, self.info.ops_num, max_operations, varied_ops)
                client_send_slice, server_send_slice = generate_send_slice(robot_operation_num, server_operation_num)

                # print(f"max_operations {max_operations}")
                # print(f"client_operation_num {robot_operation_num}")
                # print(f"server_operation_num {server_operation_num}")
                # print(f"client_send_slice {client_send_slice}")
                # print(f"server_send_slice {server_send_slice}")

                n = len(self.info.dependency)

                client_time = np.zeros(n, dtype=float)
                server_time = np.zeros(n, dtype=float)

                inverse_max_slice_lengths = 1/ max_slices

                client_send_delta = np.array(client_send_slice)
                server_send_delta = np.array(server_send_slice)

                transmit_data_client = client_send_delta * np.array(self.info.transmit_data) * inverse_max_slice_lengths
                transmit_data_server = server_send_delta * np.array(self.info.transmit_data) * inverse_max_slice_lengths
                
                all_client_comp_time = robot_operation_num / max_operations * self.info.robot_ops_time
                all_server_comp_time = server_operation_num / max_operations * self.info.server_ops_time
                
                client_send_indices = np.where(client_send_delta > 0)[0]
                client_transmit_time = self.estimate_transmission_time(transmit_data_client, bandwidth, "client")
                server_send_indices = np.where(server_send_delta > 0)[0]
                server_transmit_time = self.estimate_transmission_time(transmit_data_server, bandwidth, "server")

                for i, node in enumerate(self.info.dependency):
                    if client_time[i] > min_value or server_time[i] > min_value:
                        return None  

                    common_factor = inverse_max_slice_lengths[i]
                    client_comp_time = all_client_comp_time[i]
                    server_comp_time = all_server_comp_time[i]

                    for parent in node[1]:
                        client_time[i] = max(client_time[i], client_time[parent] + client_comp_time)
                        server_time[i] = max(server_time[i], server_time[parent] + server_comp_time)

                        if client_transmit_time[parent] > 0:
                            server_time[i] = max(server_time[i], client_time[parent] + client_comp_time + client_transmit_time[parent])

                        if server_transmit_time[parent] > 0:
                            client_time[i] = max(client_time[i], server_time[parent] + server_comp_time + server_transmit_time[parent])

                    # print(f"{i} client_time {client_time[i]} server_time {server_time[i]}")
                
                return client_time[-1], params_x

            # Initialize global best value for early stopping
            if num_workers is None:
                num_workers = max(1, multiprocessing.cpu_count() - 1)  # 自动检测 CPU 核数
            adjusted_ranges = [adjust_range(r,step_start, step_length) for r in ranges]
            iterator = iter(product(*adjusted_ranges)) 
            total_combinations = np.prod([len(r) for r in adjusted_ranges])
            # print(f"bandwidth {bandwidth} search space: {adjusted_ranges} min value {min_value}")
            with tqdm(total=total_combinations, desc="Processing") as pbar:
                for params in iterator:
                    result = objective_function(params)
                    if result is not None:
                        value, current_params = result
                        if value < min_value:
                            min_value = value
                            best_params = current_params
                            pbar.set_description(f"Best: {min_value:.6f}")
                    pbar.update(1)
            # with ProcessPoolExecutor(max_workers=num_workers) as executor, tqdm(total=total_combinations, desc="Processing") as pbar:
            #     while True:
            #         batch_params = []

            #         # 收集 batch_size 个任务
            #         for _ in range(task_batchsize):
            #             try:
            #                 batch_params.append(next(iterator))
            #             except StopIteration:
            #                 break  # 迭代器耗尽，终止本次 batch 数据填充

            #         if not batch_params:  # 如果 batch_params 为空，表示迭代器完全耗尽，结束循环
            #             break 
                    
            #         futures = {executor.submit(objective_function, params): params for params in batch_params}

            #         for future in as_completed(futures):
            #             result = future.result()
            #             if result is not None:
            #                 value, current_params = result
            #                 if value < min_value:
            #                     min_value = value
            #                     best_params = current_params
            #                     pbar.set_description(f"Best: {min_value:.6f}")

            #             pbar.update(1)

            return min_value, best_params
        
        num_nodes = self.info.ops_num
        max_operations = []
        max_slices = []
        for idx in range(num_nodes):
            dim = self.profiles[idx].local_dim
            if (dim is None or self.info.barrier[idx]) and len(self.info.output_shapes[idx]) > 0:
                max_operations.append(1) # 0, 1
                max_slices.append(self.info.output_shapes[idx][0][-1])
            elif self.profiles[idx].excluded or not self.info.output_shapes[idx]:
                max_operations.append(64)
                max_slices.append(64)
            else:
                max_operations.append(self.info.output_shapes[idx][0][dim]) # 0, ..., dim_len
                max_slices.append(self.info.output_shapes[idx][0][dim])
        max_operations = np.array(max_operations, dtype=np.int32)
        max_slices = np.array(max_slices, dtype=np.int32)


        # params_x = []
        # for idx in varied_ops:
        #     dim = self.profiles[idx].local_dim
        #     if dim is None or self.info.barrier[idx]:
        #         params_x.append(0) # 0, 1
        #     else:
        #         params_x.append(int(self.info.output_shapes[idx][0][dim]/2)) # 0, ..., dim_len
        # params_x = tuple(params_x)
        # client_operation_num, server_operation_num = get_operation_distribution(params_x)
      
        min_estimated_time = sum(self.info.robot_ops_time)
        if len(varied_ops) > 0 and bandwidth > 0.5:
            search_ranges = []
            for op in varied_ops:
                dim = self.profiles[op].local_dim
                if dim is None or self.info.barrier[op]:
                    search_ranges.append(2)
                else:
                    search_ranges.append(self.info.output_shapes[op][0][dim]+1)
            min_estimated_time, best_x_varied_op = find_minimum(search_ranges,min_value = min_estimated_time,best_params=max_operations)
            client_operation_num, server_operation_num = get_operation_distribution(best_x_varied_op, self.info.dependency, self.info.ops_num, max_operations)
        else:
            # lcoal compute by default
            client_operation_num = np.array(max_operations, dtype=np.int32)
            server_operation_num = np.zeros(num_nodes, dtype=np.int32)
            
            
        #from x, y to client plan, server plan
        def generate_send_keep_recv_plan(client_operation_num, server_operation_num,client_send_slice, server_send_slice):    
            client_send = OrderedDict()
            client_keep = OrderedDict()
            server_recv = OrderedDict()
            client_keys = []

            server_send = OrderedDict()
            server_keep = OrderedDict()
            client_recv = OrderedDict()
            server_keys = []

            for i, node in enumerate(self.info.dependency):
                client_output_slice = max_slices[i] * client_operation_num[i] if self.info.barrier[i] else client_operation_num[i]
                server_output_slice = max_slices[i] * server_operation_num[i] if self.info.barrier[i] else server_operation_num[i]
                if client_send_slice[i] > 0:
                    client_keep_slice = np.array([
                        max_slices[i] * required_operation_num(child, client_operation_num[child]) if self.info.barrier[child]
                        else required_operation_num(child, client_operation_num[child])
                        for child in node[2]
                    ], dtype=int)

                    client_keep[i] = (0, np.max(client_keep_slice))
                    client_send[i] = (client_output_slice - client_send_slice[i], client_output_slice)
                    server_recv[i] = (-server_output_slice, 0)
                    client_keys.append(i)

                if server_send_slice[i] > 0:
                    server_keep_slice = np.array([
                        max_slices[i] * required_operation_num(child, server_operation_num[child]) if self.info.barrier[child]
                        else required_operation_num(child, server_operation_num[child])
                        for child in node[2]
                    ], dtype=int)         

                    server_keep[i] = (-np.max(server_keep_slice), 0)
                    server_send[i] = (-server_output_slice,-server_output_slice + server_send_slice[i])                    
                    client_recv[i] = (0, client_output_slice)
                    server_keys.append(i)   

            return client_send, client_keep, server_recv, client_keys, server_send, server_keep, client_recv, server_keys
        

        def get_slice_shapes(idx, shard):
            def wrap_constraint(d):
                if d < 0:
                    negative = True
                    d = -d
                else:
                    negative = False
                if constraint:
                    kernel_size, stride, padding = constraint
                    print(d,kernel_size, stride, padding)
                    d = np.ceil((d - kernel_size + padding * 2)
                                / stride) * stride + kernel_size - padding * 2
                    print(d)
                    d += (4-d % 4) if d % 4 > 0 else 0  # Optimize cudnn
                    print(d)
                    assert d % 4 == 0
                    d = min(max(d, 0),shape[dim])
                if negative:
                    return -int(d)
                else:
                    return int(d)
            def wrap_dim(_slice):
                if dim == -1:
                    return (..., _slice)
                elif dim == 0:
                    return (_slice, ...)
                elif dim > 0:
                    return tuple([slice(None)]*dim + [_slice] + [slice(None)]*(len(shape)-1 - dim))
                else:
                    return tuple([slice(None)]*(len(shape)-1 + dim) + [_slice] + [slice(None)]*(- dim))
            def _slice_shape(shape: torch.Size):
                start_pose = wrap_constraint(shard[0]/shapes[0][dim]*shape[dim])
                end_pose = wrap_constraint(shard[1]/shapes[0][dim]*shape[dim])
                if start_pose == 0:
                    if end_pose == 0:
                        return wrap_dim(slice(0))
                    else:
                        return wrap_dim(slice(None,end_pose))
                else:
                    if end_pose == 0:
                        return wrap_dim(slice(start_pose,None))
                    else:
                        return wrap_dim(slice(start_pose,end_pose))
            
            shapes = self.info.output_shapes[idx]
            if self.info.conv[idx]:
                constraint = [self.info.kernel_size[idx], self.info.stride[idx], self.info.padding[idx]]
            else:
                constraint = []
            
            dim = self.profiles[idx].local_dim
            if dim is None or self.info.barrier[idx]:
                dim = -1
            
            slice_shapes = []
            for shape in shapes:
                ret = _slice_shape(shape)
                slice_shapes.append(ret)
            return slice_shapes


        client_send_slice, server_send_slice = generate_send_slice(client_operation_num, server_operation_num)
        client_send, client_keep, server_recv, client_keys, server_send, server_keep, client_recv, server_keys = generate_send_keep_recv_plan(client_operation_num, server_operation_num,client_send_slice, server_send_slice)
        
        client_skip_plan = np.zeros(num_nodes, dtype=bool)
        client_offload_plan = np.zeros(num_nodes, dtype=bool)
        client_recv_plan = np.zeros(num_nodes, dtype=bool)
        server_skip_plan = np.zeros(num_nodes, dtype=bool)
        server_offload_plan = np.zeros(num_nodes, dtype=bool)
        server_recv_plan = np.zeros(num_nodes, dtype=bool)

        client_recv_first = np.ones(num_nodes, dtype=bool)
        server_recv_first = np.ones(num_nodes, dtype=bool)

        client_send_slice_plan: Dict[int, slice] = OrderedDict()
        client_send_keep_slice_plan: Dict[int, slice] = OrderedDict()
        client_recv_keep_slice_plan: Dict[int, slice] = OrderedDict()
        server_send_slice_plan: Dict[int, slice] = OrderedDict()
        server_send_keep_slice_plan: Dict[int, slice] = OrderedDict()
        server_recv_keep_slice_plan: Dict[int, slice] = OrderedDict()


        client_cat_order_plan: Dict[int, slice] = OrderedDict()
        client_cat_dim_plan: Dict[int, slice] = OrderedDict()
        server_cat_order_plan: Dict[int, slice] = OrderedDict()
        server_cat_dim_plan: Dict[int, slice] = OrderedDict()

        client_keys_set = set(client_keys)  
        server_keys_set = set(server_keys)

        for idx in range(num_nodes):
            dim = self.profiles[idx].local_dim
            if dim is None or self.info.barrier[idx]:
                dim = -1
            
            client_skip_plan[idx] = (client_operation_num[idx] == 0)
            server_skip_plan[idx] = (server_operation_num[idx] == 0)

            if idx in client_keys_set or idx in server_keys_set:
                if idx in client_keys_set:
                    client_offload_plan[idx] = True
                    server_recv_plan[idx] = True
                    client_send_slice_plan[idx] = get_slice_shapes(idx, client_send[idx])
                    client_send_keep_slice_plan[idx] = get_slice_shapes(idx, client_keep[idx])
                    server_recv_keep_slice_plan[idx] = get_slice_shapes(idx, server_recv[idx])
                if idx in server_keys_set:
                    server_offload_plan[idx] = True
                    client_recv_plan[idx] = True
                    server_send_slice_plan[idx] = get_slice_shapes(idx, server_send[idx])
                    server_send_keep_slice_plan[idx] = get_slice_shapes(idx, server_keep[idx])
                    client_recv_keep_slice_plan[idx] = get_slice_shapes(idx, client_recv[idx])
                
                server_cat_order_plan[idx] = 1
                client_cat_order_plan[idx] = 0
                server_cat_dim_plan[idx] = dim
                client_cat_dim_plan[idx] = dim

        x = client_operation_num / max_operations
        align_shape = self.info.align_shape & (x > 0) & (x < 1)
        transmission_time = self.info.transmit_data / bandwidth / 1024 / 1024

        client_plan = {"skip":client_skip_plan, "offload": client_offload_plan, "recv_first": client_recv_first, "send_slice": client_send_slice_plan, "send_keep_slice": client_send_keep_slice_plan, "recv": client_recv_plan, "recv_keep_slice": client_recv_keep_slice_plan, "cat_order":client_cat_order_plan, "cat_dim":client_cat_dim_plan, "est_time": min_estimated_time, "align_shape": align_shape * 1, "x": x, "comp_time": self.info.robot_ops_time, "transmission_time": transmission_time}

        server_plan = {"skip":server_skip_plan, "offload": server_offload_plan, "recv_first": server_recv_first, "send_slice": server_send_slice_plan, "send_keep_slice": server_send_keep_slice_plan, "recv": server_recv_plan, "recv_keep_slice": server_recv_keep_slice_plan, "cat_order":server_cat_order_plan, "cat_dim":server_cat_dim_plan, "est_time": min_estimated_time, "align_shape": align_shape * 2, "x": x, "comp_time": self.info.server_ops_time, "transmission_time": transmission_time}

        # print(f"max_operations {max_operations}")
        # print(f"client_operation_num {client_operation_num}")
        # print(f"server_operation_num {server_operation_num}")          
        # print(f"client_send_slice {client_send_slice}")
        # print(f"server_send_slice {server_send_slice}")
        # print(f"***********************")
        # print(f"client_send_keys {client_keys}")
        # print(f"client_send {client_send}")
        # print(f"client_send_slice_plan {client_send_slice_plan}")
        # print(f"client_keep {client_keep}")
        # print(f"client_send_keep_slice_plan {client_send_keep_slice_plan}")
        # print(f"server_recv_keep_slice_plan {server_recv_keep_slice_plan}")
        # print(f"server_recv {server_recv}")
        # print(f"***********************")
        # print(f"server_send_keys {server_keys}")
        # print(f"server_send {server_send}")
        # print(f"server_send_slice_plan {server_send_slice_plan}")
        # print(f"server_keep {server_keep}")
        # print(f"server_send_keep_slice_plan {server_send_keep_slice_plan}")
        # print(f"client_recv {client_recv}")
        # print(f"client_recv_keep_slice_plan {client_recv_keep_slice_plan}")
        
        return client_plan, server_plan

    def generate_intraDP_plan(self,bandwidth):
        if self.x_for_ops is None:
            self.allocate_x_for_ops()

        def objective_function(x, bandwidth):
            client_time = 0.
            server_time = 0.
            for i in range(self.info.ops_num):
                if i == 0:
                    current_robot_rate = 1.
                else:
                    x_idx = self.x_for_ops[i]
                    current_robot_rate = self.get_actual_robot_rate(x[x_idx],i)
                
                if i == self.info.ops_num -1:
                    next_robot_rate = 1.
                else:
                    x_idx = self.x_for_ops[self.info.dependency[i][2][0]]
                    next_robot_rate = self.get_actual_robot_rate(x[x_idx],self.info.dependency[i][2][0])
                
                client_time += current_robot_rate*self.info.robot_ops_time[i]
                server_time += (1-current_robot_rate)*self.info.server_ops_time[i]

                if next_robot_rate > current_robot_rate:
                    # transmit to client
                    transfer_rate = next_robot_rate - current_robot_rate
                    transmit_time = self.estimate_transmission_time(self.info.transmit_data[i]*transfer_rate, bandwidth, "server")
                    server_time += self.server_cut_cost_time
                    client_time = max(client_time, server_time + transmit_time)

                if next_robot_rate < current_robot_rate:
                    # transmit to server
                    transfer_rate = current_robot_rate - next_robot_rate
                    transmit_time = self.estimate_transmission_time(self.info.transmit_data[i]*transfer_rate, bandwidth, "client")
                    server_time = max(server_time, client_time + transmit_time)
                    client_time += self.robot_cut_cost_time

            return client_time
        
        # local solver
        def custom_constraint1(x,idx):
            return x[idx]
        
        def custom_constraint2(x,idx):
            return 1. - x[idx]

        constraints = []
        for i in range(self.x_num):
            # x >= 0.
            constraints.append({'type':'ineq','fun': custom_constraint1,'args':(i,)})
            # x <= 1.
            constraints.append({'type':'ineq','fun': custom_constraint2,'args':(i,)})
        
        for i in range(self.random_times):
            x0 = np.random.uniform(0., 1., size=self.x_num)
            result = optimize.minimize(objective_function,x0,constraints = constraints, args=(bandwidth,))
            if self.intraDP_estimated_time > result.fun:
                self.intraDP_estimated_time = result.fun
                self.intraDP_result_x = result.x

        # global solver differential_evolution
        ranges = [(0., 1.) for _ in range(self.x_num)]
        result = optimize.differential_evolution(objective_function, ranges, args=(bandwidth,))
        if self.intraDP_estimated_time > result.fun:
            self.intraDP_estimated_time = result.fun
            self.intraDP_result_x = result.x
        
        return self.transfer_robot_rate_to_intraDP_plan(bandwidth)   

    def generate_plan(self, bandwidth):
        if self.parallel_approach == "select" or self.parallel_approach in ["SPSO-GA", "DSCCS"]:
            client_plan = self.generate_select_plan(bandwidth) 
            server_plan = self.transfer_client_plan_to_server_plan(client_plan)
            client_plan["x"] = np.array(~client_plan["skip"], dtype=float)
            client_plan["comp_time"] = self.info.robot_ops_time
            client_plan["transmission_time"] = self.info.transmit_data / bandwidth / 1024 / 1024
            server_plan["x"] = np.array(~client_plan["skip"], dtype=float)
            server_plan["comp_time"] = self.info.server_ops_time
            server_plan["transmission_time"] = self.info.transmit_data / bandwidth / 1024 / 1024
        elif self.parallel_approach == "greedy":
            client_plan = self.generate_greedy_plan(bandwidth) 
            server_plan = self.transfer_client_plan_to_server_plan(client_plan)
        elif self.parallel_approach == "tp":
            client_plan, server_plan = self.generate_tp_plan(bandwidth)
        elif self.parallel_approach == "intraDP1":
            client_plan, server_plan = self.generate_intraDP_plan(bandwidth)
        elif self.parallel_approach == "intraDP":
            client_plan, server_plan = self.generate_intraDP_plan_2(bandwidth)
        elif self.parallel_approach == "intraDP3":
            client_plan, server_plan = self.generate_intraDP_plan_3(bandwidth)
        for plan in [client_plan, server_plan]:
            if "align_shape" not in plan:
                plan["align_shape"] = np.zeros_like(plan["skip"])
        return client_plan, server_plan

    def build_graph(self):
        self.total_local_skip_plan = np.array([False for _ in range(self.info.ops_num)])
        self.total_local_offload_plan = np.array([False for _ in range(self.info.ops_num)])
        self.total_local_recv_plan = np.array([False for _ in range(self.info.ops_num)])

        self.total_offload_skip_plan = np.array([True for _ in range(self.info.ops_num)])
        self.total_offload_skip_plan[0] = False
        self.total_offload_skip_plan[-1] = False
        self.total_offload_offload_plan = np.array([False for _ in range(self.info.ops_num)])
        self.total_offload_offload_plan[0] = True
        self.total_offload_recv_plan = np.array([False for _ in range(self.info.ops_num)])
        self.total_offload_recv_plan[-2] = True # recv before '_end'

        self.x_for_ops = None

        for bandwidth in range(self.min_bw, self.max_bw + 1):
            if bandwidth == 0:
                bandwidth = 1e-3
            try:
                client_plan, server_plan = self.generate_plan(bandwidth)
            except Exception as e:
                print(str(e))
                raise e
            bw = int(bandwidth)
            self.client_plans[bw] = client_plan
            self.server_plans[bw] = server_plan

    def recv_plan(self, plans):
        self.graph_plan = plans

    def total_computation_time(self,layer,start,end):
        return layer[start:end, 1].sum()

    def update_ops(self, robot_ops: OffloadProfile, server_ops: OffloadProfile):
        self.total_local_time = sum([p.ops_time for p in robot_ops.profile.values()])
        self.robot_size_to_dumps_time, self.robot_size_to_loads_time = \
            robot_ops.size_to_dumps_time, robot_ops.size_to_loads_time
        self.server_size_to_dumps_time, self.server_size_to_loads_time = \
            server_ops.size_to_dumps_time, server_ops.size_to_loads_time
        self.info.update_computation_time(robot_ops, server_ops)
        self.profiles = list(server_ops.profile.values())
