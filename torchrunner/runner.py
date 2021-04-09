from warnings import warn

import torch

from torchrunner.callback import CallbackClient, callback
from torchrunner.utils import set_opt_param, cat_list_tensor_list, cat_list_tensor_dict


class Runner(CallbackClient):
    def __init__(self, model, train_dl=None, val_dl=None, test_dl=None, loss_fn=None,
                 metric_fns=None, optimizer=None, lr=3e-3, wd=None, device='cpu', callbacks=None):
        super().__init__(callbacks)
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns or []
        self.optimizer = optimizer
        self.lr = lr
        self.wd = wd
        self.device = torch.device(device)

    @callback
    def configure_optimizer(self, reset_opt):
        if reset_opt:
            self.optimizer = self.optimizer.__class__(self.optimizer.param_groups)
        else:
            set_opt_param(self.optimizer, 'lr', self.lr)
            if self.wd is not None \
                    and self.optimizer.param_groups[0].get('weight_decay', None) is not None:
                set_opt_param(self.optimizer, 'weight_decay', self.wd)

    @callback
    def forward_pass(self, inp):
        return self.model(inp)

    @callback
    def calc_loss(self, out, targ):
        return self.loss_fn(out, targ)

    @callback
    def backward_pass(self, loss):
        loss.backward()

    @callback
    def update_params(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    @callback
    def train_batch(self, inp, targ):
        out = self.forward_pass(inp=inp)
        loss = self.calc_loss(out=out, targ=targ)
        self.backward_pass(loss=loss)
        self.update_params()
        loss = loss.detach().cpu()
        metrics = [metric_fn(out, targ).detach().cpu() for metric_fn in self.metric_fns]
        return loss, metrics

    @callback
    def train_all_batches(self, dl):
        self.model.train()
        acc_loss = torch.tensor(0.)
        acc_metrics = None
        for inp, targ in dl:
            inp, targ = inp.to(self.device), targ.to(self.device)
            loss, metrics = self.train_batch(inp=inp, targ=targ)
            acc_loss += loss * inp.shape[0]
            if acc_metrics is None:
                acc_metrics = [torch.zeros_like(metric) for metric in metrics]
            for i in range(len(metrics)):
                acc_metrics[i] += metrics[i] * inp.shape[0]
        acc_loss /= len(dl.dataset)
        acc_metrics = [acc_metric / len(dl.dataset) for acc_metric in acc_metrics]
        return acc_loss, acc_metrics

    @callback
    def train_epoch(self, epoch_idx):
        train_loss, train_metrics = self.train_all_batches(dl=self.train_dl)
        if self.val_dl:
            val_loss, val_metrics = self.eval_all_batches(dl=self.val_dl)
            return train_loss, train_metrics, val_loss, val_metrics
        return train_loss, train_metrics

    @callback
    def train_all_epochs(self, n_epochs):
        if self.val_dl:
            for epoch_idx in range(n_epochs):
                train_loss, train_metrics, val_loss, val_metrics = self.train_epoch(epoch_idx=epoch_idx)
            return train_loss, train_metrics, val_loss, val_metrics
        for epoch_idx in range(n_epochs):
            train_loss, train_metrics = self.train_epoch(epoch_idx=epoch_idx)
        return train_loss, train_metrics

    @callback
    def train_init(self, reset_opt):
        self.model.to(self.device)  # move model to GPU before configuring optimizer
        self.configure_optimizer(reset_opt=reset_opt)

    @callback
    def eval_batch(self, inp, targ):
        out = self.forward_pass(inp=inp)
        loss = self.calc_loss(out=out, targ=targ)
        loss = loss.detach().cpu()
        metrics = [metric_fn(out, targ).detach().cpu() for metric_fn in self.metric_fns]
        return loss, metrics

    @callback
    def eval_all_batches(self, dl):
        self.model.eval()
        acc_loss = torch.tensor(0.)
        acc_metrics = None
        with torch.no_grad():
            for inp, targ in dl:
                inp, targ = inp.to(self.device), targ.to(self.device)
                loss, metrics = self.eval_batch(inp=inp, targ=targ)
                acc_loss += loss * inp.shape[0]
                if acc_metrics is None:
                    acc_metrics = [torch.zeros_like(metric) for metric in metrics]
                for i in range(len(metrics)):
                    acc_metrics[i] += metrics[i] * inp.shape[0]
        acc_loss /= len(dl.dataset)
        acc_metrics = [acc_metric / len(dl.dataset) for acc_metric in acc_metrics]
        return acc_loss, acc_metrics

    @callback
    def eval_init(self):
        self.model.to(self.device)

    @callback
    def pred_batch(self, inp):
        return self.forward_pass(inp=inp)

    @callback
    def pred_all_batches(self, dl):
        pred_outs, pred_targs = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in dl:
                inp = batch[0].to(self.device)
                out = self.pred_batch(inp=inp)
                if isinstance(out, (tuple, list)):
                    out = [item.cpu() for item in out]
                elif isinstance(out, dict):
                    out = {k: v.cpu() for k, v in out.items()}
                else:
                    out = out.cpu()
                pred_outs.append(out)
                if len(batch) > 1:
                    targ = batch[1]
                    pred_targs.append(targ)
        if isinstance(pred_outs[0], (tuple, list)):
            pred_outs = cat_list_tensor_list(pred_outs)
        elif isinstance(pred_outs[0], dict):
            pred_outs = cat_list_tensor_dict(pred_outs)
        else:
            pred_outs = torch.cat(pred_outs)
        if len(pred_targs) > 0:
            pred_targs = torch.cat(pred_targs)
            return pred_outs, pred_targs
        return pred_outs

    @callback
    def pred_init(self):
        self.model.to(self.device)

    def train(self, n_epochs, lr=None, wd=None, train_dl=None, val_dl=None,
              loss_fn=None, metric_fns=None, optimizer=None, device=None, reset_opt=False):
        self.lr = lr or self.lr
        self.wd = wd or self.wd
        self.train_dl = train_dl or self.train_dl
        self.val_dl = val_dl or self.val_dl
        self.loss_fn = loss_fn or self.loss_fn
        self.metric_fns = metric_fns or self.metric_fns
        self.optimizer = optimizer or self.optimizer
        self.device = torch.device(device) if device else self.device

        assert n_epochs > 0
        assert self.train_dl is not None
        assert self.loss_fn is not None
        assert self.optimizer is not None

        self.train_init(reset_opt=reset_opt)
        return self.train_all_epochs(n_epochs=n_epochs)

    def evaluate(self, test_dl=None, loss_fn=None, metric_fns=None, device=None):
        self.test_dl = test_dl or self.test_dl
        self.loss_fn = loss_fn or self.loss_fn
        self.metric_fns = metric_fns or self.metric_fns
        self.device = torch.device(device) if device else self.device

        assert self.test_dl is not None
        assert self.loss_fn is not None

        self.eval_init()
        return self.eval_all_batches(dl=self.test_dl)

    def predict(self, test_dl=None, device=None):
        self.test_dl = test_dl or self.test_dl
        self.device = torch.device(device) if device else self.device

        assert self.test_dl is not None

        self.pred_init()
        return self.pred_all_batches(dl=self.test_dl)

    def predict_online(self, inp, device=None):
        self.device = torch.device(device) if device else self.device

        self.pred_init()

        self.model.eval()
        with torch.no_grad():
            inp = inp.to(self.device)
            out = self.pred_batch(inp=inp)
            if isinstance(out, (tuple, list)):
                out = [item.cpu() for item in out]
            elif isinstance(out, dict):
                out = {k: v.cpu() for k, v in out.items()}
            else:
                out = out.cpu()
        return out

    def save(self, file, with_opt=True):
        if self.optimizer is None:
            with_opt = False
        state = self.model.state_dict()
        if with_opt:
            state = {'model': state, 'optimizer': self.optimizer.state_dict()}
        torch.save(state, file)

    def load(self, file, with_opt=True, strict=True):
        if self.optimizer is None:
            with_opt = False
        state = torch.load(file)
        has_opt = (set(state) == {'model', 'optimizer'})
        model_state = state['model'] if has_opt else state
        self.model.load_state_dict(model_state, strict=strict)
        if has_opt and with_opt:
            try:
                self.optimizer.load_state_dict(state['optimizer'])
            except Exception:
                if with_opt:
                    warn("Could not load optimizer state.")
        elif with_opt:
            warn("Saved file doesn't contain optimizer state.")
