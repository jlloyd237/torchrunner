import pdb
from functools import wraps
from collections import OrderedDict
from warnings import warn

from torchrunner.utils import Namespace


class Callback:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.context = None

    def __getattr__(self, name):
        if self.context is None or name.startswith(('before_', 'after_', 'cancel_')):
            raise AttributeError
        return getattr(self.context, name)

    def __setattr__(self, name, value):
        if name not in ('name', 'context') and hasattr(self.context, name):
            warn((f"You are setting an attribute ({name}) that also exists in the context. "
                  f"Please note that you're not setting it in the context but in the callback. "
                  f"Use `self.context.{name}` if you would like to change it in the context."))
        super().__setattr__(name, value)

    def register(self, context):
        self.context = context


class CallbackClient:
    def __init__(self, callbacks=None):
        self.init_callbacks()
        if callbacks is not None:
            self.add_callbacks(callbacks)

    def init_callbacks(self):
        self.callbacks = OrderedDict()

    def add_callbacks(self, callbacks):
        for cb in callbacks:
            cb.register(self)
            self.callbacks[cb.name] = cb

    def remove_callbacks(self, callbacks):
        for cb in callbacks:
            if isinstance(cb, Callback):
                del self.callbacks[cb.name]
            else:
                del self.callbacks[cb]

    def exec_callbacks(self, func_name, func_ns):
        for cb in self.callbacks.values():
            cb_func = getattr(cb, func_name, None)
            if cb_func is not None:
                cb_func(func_ns)


class CancelOpException(Exception):
    def __init__(self, op_name):
        self.op_name = op_name


def callback(func):
    @wraps(func)
    def _inner(self, **kwargs):
        ns = Namespace(**kwargs)
        try:
            self.exec_callbacks('before_' + func.__name__, ns)
            ns.ret = func(self, **ns)
            self.exec_callbacks('after_' + func.__name__, ns)
            return ns.ret
        except CancelOpException as e:
            if e.op_name != func.__name__:
                raise e
            self.exec_callbacks('cancel_' + func.__name__, ns)
    return _inner


class Debugger(Callback):
    def __init__(self, func_name, debug_func=None):
        super().__init__()
        self.func_name = func_name
        self.debug_func = debug_func

    def __getattr__(self, name):
        if name == self.func_name:
            if self.debug_func is not None:
                self.debug_func(self.context)
            else:
                pdb.set_trace()


class Tracer(Callback):
    def __init__(self, trace_epochs=1, trace_batches=1):
        super().__init__()
        self.trace_epochs = trace_epochs
        self.trace_batches = trace_batches
        self.processing_batch = False

    def __getattr__(self, name):
        if self.processing_batch and self.batch_idx > self.trace_batches:
            return
        print(name)

    def before_train_all_epochs(self, ns):
        ns.n_epochs = self.trace_epochs

    def before_train_all_batches(self, ns):
        self.batch_idx = 0
        self.__getattr__('before_train_all_batches')

    def before_train_batch(self, ns):
        self.processing_batch = True
        self.batch_idx += 1
        self.__getattr__('before_train_batch')

    def after_train_batch(self, ns):
        self.__getattr__('after_train_batch')
        self.processing_batch = False

    def before_eval_all_batches(self, ns):
        self.batch_idx = 0
        self.__getattr__('before_eval_all_batches')

    def before_eval_batch(self, ns):
        self.processing_batch = True
        self.batch_idx += 1
        self.__getattr__('before_eval_batch')

    def after_eval_batch(self, ns):
        self.__getattr__('after_eval_batch')
        self.processing_batch = False
