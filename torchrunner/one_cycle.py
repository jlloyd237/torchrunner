from collections import defaultdict

from torchrunner.callback import Callback
from torchrunner.scheduler import combine_schedules, cos_schedule


class OneCycleScheduler(Callback):
    def __init__(self, div=25., div_final=1e5, pct_start=0.25, mom_rng=(0.85, 0.95)):
        super().__init__()
        self.div = div
        self.div_final = div_final
        self.pct_start = pct_start
        self.mom_rng = mom_rng

    def before_train_all_epochs(self, ns):
        # Get current learning rates from optimizer
        lr_max = [pg['lr'] for pg in self.optimizer.param_groups]

        # Different learning rates for each parameter group
        self.lr_sched = [combine_schedules([self.pct_start, 1 - self.pct_start],
                         [cos_schedule(lr_max_i/self.div, lr_max_i),
                          cos_schedule(lr_max_i, lr_max_i/self.div_final)]) for lr_max_i in lr_max]

        # Single momentum range applies to all parameter groups
        self.mom_sched = [combine_schedules([self.pct_start, 1 - self.pct_start],
                          [cos_schedule(self.mom_rng[1], self.mom_rng[0]),
                           cos_schedule(self.mom_rng[0], self.mom_rng[1])]) for _ in range(len(lr_max))]

        self.n_steps = ns.n_epochs * len(self.train_dl.dataset) // self.train_dl.batch_size
        self.step_idx = 0
        self.history = defaultdict(list)

    def before_train_batch(self, ns):
        for pg_idx, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self.lr_sched[pg_idx](self.step_idx / self.n_steps)
            if pg.get('betas', None) is not None:
                pg['betas'] = self.mom_sched[pg_idx](self.step_idx / self.n_steps), *pg['betas'][1:]
            elif pg.get('momentum', None) is not None:
                pg['momentum'] = self.mom_sched[pg_idx](self.step_idx / self.n_steps)
        self.history['lr'].append([pg['lr'] for pg in self.optimizer.param_groups])
        self.history['betas'].append([pg['betas'] for pg in self.optimizer.param_groups \
                                      if pg.get('betas', None) is not None])
        self.history['momentum'].append([pg['momentum'] for pg in self.optimizer.param_groups \
                                         if pg.get('momentum', None) is not None])
        self.step_idx += 1
