from collections import OrderedDict
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode,_get_current_dispatch_mode
from typing import Dict, List, Union, Callable, Tuple, Any
import torch
import numpy as np
import warnings
import copy
import inspect
from ._utils import iterate_tensor, iterate_tensor_with_reference, args_kwargs_to_args
warnings.filterwarnings("ignore")


class InputSlot:
    """Handler of func args / intermediates
    """
    def __init__(self, idx: int, container: Union[List, Dict], index: Union[int, str]) -> None:
        self.idx = idx  # idx that this input slot belongs to
        self.container = container  # container to access the tensor
        self.index = index          # index to access the tensor from container

    def empty(self):
        self.container[self.index] = None

    def fill(self, val: torch.Tensor):
        self.container[self.index] = val

    @property
    def tensor(self)->torch.Tensor:
        return self.container[self.index]

class TorchOPProfile:
    def __init__(self, *, idx: int, func_name: str, file_name: str, line_num: int,
                 func, keep: bool, func_args: List[torch.Tensor], store_ret: List[torch.Tensor],
                 num_output: Union[int, None],
                 input_shapes: list, input_size: int,
                 output_shapes: list, output_size: int,
                 input_ids: list, output_ids: list, input_slots: List[InputSlot],
                 hook_kwargs: dict={}) -> None:
        self.idx = idx
        self.func = func
        self.keep = keep
        self.file_name = file_name
        self.line_num = line_num

        # function args and kwargs at exec (intermediates already emptied)
        self.align_shape = False
        self.func_args = func_args
        self.store_ret = store_ret
        self.func_name = func_name
        self.num_output = num_output
        self.input_shapes = input_shapes # Bytes
        self.input_size = input_size
        self.input_ids = input_ids
        self.output_shapes = output_shapes # Bytes
        self.output_size = output_size
        self.output_ids = output_ids
        self.input_slots = input_slots
        self.local_dim = None
        self.hook_kwargs = hook_kwargs
        self.ops_time = 0.
        from .profile_ops import OpsStamp
        self.ops_stamp: OpsStamp = None
        self.input_from = []
        self.output_to = []
        self.input_idx = []
        self.output_idx_slots: Dict[int, List[InputSlot]] = OrderedDict()
        self.caching = False
        self.masked = False
        self.excluded = False
        self.static_computed = False
    def copy_for_transmission(self):
        """Clear the unpicklable objects: func_args, func and input slots"""
        ret = TorchOPProfile(idx=self.idx, func_name=self.func_name, func=None, file_name=self.file_name, line_num=self.line_num, store_ret=None, keep=self.keep,
                            func_args=[], num_output=self.num_output, input_shapes=self.input_shapes,
                            input_size=self.input_size, output_shapes=self.output_shapes,
                            output_size=self.output_size, input_ids=self.input_ids,
                            output_ids=self.output_ids, input_slots=None, hook_kwargs={})
        ret.input_from = self.input_from
        ret.ops_time = self.ops_time
        ret.ops_stamp = self.ops_stamp
        ret.input_from = self.input_from
        ret.output_to = self.output_to
        ret.masked = self.masked
        ret.excluded = self.excluded
        ret.hook_kwargs = {}
        ret.hook_kwargs.update(self.hook_kwargs if self.hook_kwargs is not None else {})
        if "profile" in ret.hook_kwargs:
            del ret.hook_kwargs["profile"]
        return ret

    def __repr__(self) -> str:
        return f"{self.idx} {self.func_name}: input_from: {self.input_from}, output_to: {self.output_to}, output_shapes: {self.output_shapes}, local dim: {self.local_dim}, masked: {self.masked}, align_shape: {self.align_shape}, {self.file_name}:{self.line_num} excluded: {self.excluded} ops_time: {self.ops_time*1000:.2f}ms; "


class OffloadProfile:
    def __init__(self) -> None:
        """Changed from dataclass to regular class since pickle seems unable to handle dataclass"""
        self.idx: int = 0
        self.end_idx = 0
        self.local_comp_time = 0.
        self.profile: Dict[int, TorchOPProfile] = OrderedDict()
        self.ignore_ops = []
        self.original_input = None
        self.ret_store = None
        self.ret_slots: List[InputSlot] = []
        self.size_to_loads_time: np.poly1d = np.poly1d([0,0,0])
        self.size_to_dumps_time: np.poly1d = np.poly1d([0,0,0])
        self.exec_plan: Dict[int, List[Tuple[TorchOPProfile, int, bool]]] = OrderedDict()

        self.client_comp_time = 0.
        self.server_comp_time = 0.
        self.client_size_to_loads_time: np.poly1d = np.poly1d([0,0,0])
        self.client_size_to_dumps_time: np.poly1d = np.poly1d([0,0,0])
        self.server_size_to_loads_time: np.poly1d = np.poly1d([0,0,0])
        self.server_size_to_dumps_time: np.poly1d = np.poly1d([0,0,0])

    def client_offload_time(self, op_idx: int, bw: float, offload_output: bool) -> float:
        # bw is in MB/s
        profile = self.profile[op_idx]
        if offload_output:
            size = profile.output_size / 1e6
        else:
            size = profile.input_size / 1e6
        return size / (bw + 1e-6) + self.client_size_to_dumps_time(size) + self.server_size_to_loads_time(size)

    def server_offload_time(self, op_idx: int, bw: float, offload_output: bool) -> float:
        # bw is in MB/s
        profile = self.profile[op_idx]
        if offload_output:
            size = profile.output_size / 1e6
        else:
            size = profile.input_size / 1e6
        return size / (bw + 1e-6) + self.client_size_to_loads_time(size) + self.server_size_to_dumps_time(size)

    def __getitem__(self, index: Union[int, str]):
        return getattr(self, index)

    def __setitem__(self, index: Union[int, str], val):
        setattr(self, index, val)


    def copy_for_transmission(self):
        new_profile = OffloadProfile()
        new_profile.idx = self.idx
        new_profile.end_idx = self.end_idx
        new_profile.profile = OrderedDict()
        for idx, p in self.profile.items():
            new_profile.profile[idx] = p.copy_for_transmission()
        new_profile.size_to_loads_time = self.size_to_loads_time
        new_profile.size_to_dumps_time = self.size_to_dumps_time
        new_profile.local_comp_time = self.local_comp_time
        return new_profile


def get_ops(profile_result: OffloadProfile):
    ops = []
    all_profiles = profile_result.profile
    idx_array = list(all_profiles.keys())
    for idx in idx_array:
        profile: TorchOPProfile = all_profiles[idx]
        ops.append([profile.idx, profile.ops_time, profile.output_size])
    return np.array(ops)

def get_dependency(profile_result: OffloadProfile):
    ops = []
    all_profiles = profile_result.profile
    idx_array = list(all_profiles.keys())
    for idx in idx_array:
        profile: TorchOPProfile = all_profiles[idx]
        ops.append([profile.idx, profile.input_from, profile.output_to])
    return ops

def return_as_is(*args, **kwargs):
    return args, kwargs

keep_funcs = {"__setitem__", "_start", "_end"}
skip_funcs = {"__get__", "dim", "size"}
def profile_model_ops(forward_func: Callable=None,
                      args: List[torch.Tensor]=None,
                      kwargs: Dict[str, Any]=None, log=print) -> OffloadProfile:
    profile_result = OffloadProfile()
    class ProfileInfoMode(TorchFunctionMode):
        def __init__(self):
            super().__init__()
    
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            ret = func(*args, **kwargs)
            stacks = inspect.stack()
            for stack in stacks[1:]:
                if "overrides.py" in stack.filename or "torch" in stack.filename:
                    continue
                else:
                    break
            file_name, line_num = stack.filename, stack.lineno
            ret = self.add_profile(func.__name__, file_name, line_num, args, kwargs, ret)
            return ret

        @classmethod
        def add_profile(cls, func_name: str, file_name: str, line_num: int,
                        args: List[torch.Tensor], kwargs: Dict[str, Any],
                        ret: torch.Tensor):
            idx = profile_result.idx
            profile_result.idx += 1
            outputs = []
            input_shapes = []
            input_size = [0]
            input_ids = []
            output_shapes = []
            output_size = [0]
            output_ids = []
            input_slots = []
            def profile_input(arg: torch.Tensor):
                input_shapes.append(arg.shape)
                input_size[0] += arg.nelement() * arg.element_size()
                _id = id(arg)
                input_ids.append(_id)
                return arg
            store_func_args = iterate_tensor(
                [args, kwargs], profile_input)

            def profile_output(_ret: torch.Tensor):
                output_shapes.append(_ret.shape)
                output_size[0] += _ret.nelement()*_ret.element_size()
                outputs.append(True)
                _id = id(_ret)
                output_ids.append(_id)
                return _ret
            ret = iterate_tensor(ret, profile_output)
            if len(output_ids) == 0:
                assert func_name in keep_funcs or func_name in skip_funcs, \
                    f"op {idx} {func_name} has no output and not handled by us."
            num_output = len(outputs) if len(outputs) != 1 else None
            if len(output_ids) == 0:
                output_size = input_size
            profile_result.profile[idx] = TorchOPProfile(
                idx=idx,
                func_name=func_name,
                file_name=file_name,
                line_num=line_num,
                func=None,
                keep=func_name in keep_funcs,
                func_args=store_func_args,
                store_ret=ret,
                num_output=num_output,
                input_shapes=input_shapes,
                input_size=input_size[0],
                output_shapes=output_shapes,
                output_size=output_size[0],
                input_ids=input_ids,
                output_ids=output_ids,
                input_slots=input_slots
            )
            return ret
    ProfileInfoMode.add_profile("_start", "", 0, args, kwargs, (args, kwargs))
    with ProfileInfoMode():
        ret = forward_func(*args, **kwargs)
    ProfileInfoMode.add_profile("_end", "ours", 0, ret, {}, ret)
    return profile_result

def profile_tensor_factory(profile_result: OffloadProfile, hook_level: List, debug=False):
    class ProfileTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            idx = profile_result.idx
            if kwargs is None:
                kwargs = {}
            if func is not None:
                ret = super().__torch_function__(func, types, args, kwargs)
                func_name = func.__name__
            else:
                if idx == 0:
                    func_name = "_start"
                    ret = [args, kwargs]
                else:
                    func_name = "_end"
                    assert len(kwargs) == 0
                    ret = args[0]
                func = return_as_is
            # Avoid recursive call or profiling non-computation functions.
            if hook_level[0]:
                return ret
            hook_level[0] += 1

            outputs = []
            input_shapes = []
            input_size = [0]
            input_ids = []
            output_shapes = []
            output_size = [0]
            output_ids = []
            input_slots = []
            def profile_input(arg: ProfileTensor,
                            container: Union[List, Dict], index: Union[int, str]):
                input_shapes.append(arg.shape)
                input_size[0] += arg.nelement() * arg.element_size()
                _id = id(arg)
                input_ids.append(_id)

                # Handler to fill and modify intermediates.
                assert container is not None
                input_slots.append(InputSlot(idx, container, index))
                return arg.as_subclass(torch.Tensor)[:0]
            store_func_args = iterate_tensor_with_reference(
                [args, kwargs], profile_input, ProfileTensor, None, None)

            def profile_output(_ret: ProfileTensor):
                output_shapes.append(_ret.shape)
                output_size[0] += _ret.nelement()*_ret.element_size()
                outputs.append(True)
                _id = id(_ret)
                output_ids.append(_id)
                return _ret
            stacks = inspect.stack()
            for stack in stacks[1:]:
                if "overrides.py" in stack.filename or "torch" in stack.filename:
                    continue
                else:
                    break
            file_name, line_num = stack.filename, stack.lineno
            ret = iterate_tensor(ret, profile_output, ProfileTensor)
            store_ret = iterate_tensor(ret, lambda x: copy.deepcopy(x.as_subclass(torch.Tensor)), ProfileTensor)
            if func_name == "_end":
                # A place to hold output of _end op.
                def profile_forward_ret(_ret: ProfileTensor, container, index):
                    profile_result.ret_slots.append(InputSlot(idx, container, index))
                    return _ret.as_subclass(torch.Tensor)
                profile_result.ret_store = iterate_tensor_with_reference(
                    ret, profile_forward_ret, ProfileTensor, profile_result, "ret_store")
            if len(output_ids) == 0:
                assert func_name in keep_funcs or func_name in skip_funcs, \
                    f"op {idx} {func_name} has no output and not handled by us."
            num_output = len(outputs) if len(outputs) != 1 else None

            profile_result.profile[idx] = TorchOPProfile(
                idx=idx,
                func_name=func_name,
                file_name=file_name,
                line_num=line_num,
                func=func,
                keep=func_name in keep_funcs,
                func_args=store_func_args,
                store_ret=store_ret,
                num_output=num_output,
                input_shapes=input_shapes,
                input_size=input_size[0],
                output_shapes=output_shapes,
                output_size=output_size[0],
                input_ids=input_ids,
                output_ids=output_ids,
                input_slots=input_slots
            )
            if False:    # For debug
                profile_result.profile[idx].store_ret = iterate_tensor(
                    ret, lambda x: copy.deepcopy(x.as_subclass(torch.Tensor)), ProfileTensor)
            profile_result.idx += 1
            hook_level[0] -= 1
            return ret
    return ProfileTensor

def offload_tensor_factory(hook_func, profile_result: OffloadProfile, hook_level: List,
                           send_queue=None, recv_queue=None):
    profile = profile_result.profile
    class OffloadTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            ret = super().__torch_function__(func, types, args, kwargs)
            if hook_level[0]:  # Avoid recursive call
                return ret
            hook_level[0] += 1

            idx = profile_result.idx
            offload_kwargs = profile[idx].hook_kwargs
            assert func.__name__ == profile[idx].func_name, f"{idx} {func.__name__} != {profile[idx].func_name}"  # TODO remove at deployment
            ret = hook_func(ret, send_queue=send_queue,
                            recv_queue=recv_queue, **offload_kwargs)
            profile_result.idx += 1
            hook_level[0] -= 1
            return ret
    return OffloadTensor

