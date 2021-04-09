import pickle
from functools import wraps, partial, update_wrapper
from collections import Mapping, defaultdict

import torch


class Namespace(Mapping):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __iter__(self):
        return iter(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def __len__(self):
        return len(self.__dict__)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__getstate__(), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp = pickle.load(f)
        self.__setstate__(tmp)


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def trace_func(func):
    @wraps(func)
    def _inner(*args, **kwargs):
        print(f"entering {func.__name__} (args={args}, kwargs={kwargs})")
        ret = func(*args, **kwargs)
        print(f"exiting {func.__name__} (ret={ret})")
        return ret
    return _inner

def trace_method(func):
    @wraps(func)
    def _inner(self, *args, **kwargs):
        print(f"entering {self.__class__.__name__}.{func.__name__} (args={args}, kwargs={kwargs})")
        ret = func(self, *args, **kwargs)
        print(f"exiting {self.__class__.__name__}.{func.__name__} (ret={ret})")
        return ret
    return _inner

def get_device(cuda=True):
    return torch.device('cuda:0' if cuda and torch.cuda.is_available() else 'cpu')

def format_param(name, value, size):
    if isinstance(value, (list, tuple)):
        if len(value) != size:
            raise ValueError(f"expected {size} values for {name}, got {len(value)}")
        return value
    else:
        return [value] * size

def format_opt_param(optimizer, name, value):
    if isinstance(value, (list, tuple)):
        if len(value) != len(optimizer.param_groups):
            raise ValueError(f"expected {len(optimizer.param_groups)} values for {name}, got {len(value)}")
        return value
    else:
        return [value] * len(optimizer.param_groups)

def set_opt_param(optimizer, name, value):
    value = format_opt_param(optimizer, name, value)
    for i, pg in enumerate(optimizer.param_groups):
        if name in pg:
            pg[name] = value[i]

def cat_list_tensor_list(list_tensor_list):
    # concatenate list of tensor lists into a list of tensors
    list_tensor = [torch.cat(l) for l in zip(*list_tensor_list)]
    return list_tensor

def cat_list_tensor_dict(list_tensor_dict):
    # concatenate list of tensor dicts into a dict of tensors
    dict_tensor_list = defaultdict(list)
    for tensor_dict in list_tensor_dict:
        for k, v in tensor_dict.items():
            dict_tensor_list[k].append(v)
    dict_tensor = {}
    for k, v in dict_tensor_list.items():
        dict_tensor[k] = torch.cat(v)
    return dict_tensor
