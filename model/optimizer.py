import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
        )
        self.init_lr = train_config["optimizer"]["init_lr"]
        self._init_learning_rate()

    def step_and_update_lr(self):
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _init_learning_rate(self):
        lr = self.init_lr
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
