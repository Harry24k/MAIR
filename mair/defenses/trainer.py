import os
import logging
from collections import OrderedDict
import numpy as np

import torch
from torch.optim import *
from torch.optim.lr_scheduler import *

from .rm import RecordManager
from .. import RobModel
from ..optim import *


r"""
Base class for all trainers.

"""


class Trainer:
    def __init__(self, rmodel, device=None):
        assert isinstance(rmodel, RobModel)
        self.rmodel = rmodel
        if device is None:
            self.device = next(rmodel.parameters()).device
        else:
            device = device

        # Init train info and record manager
        self.accumulated_epoch = 0
        self.accumulated_iter = 0
        self.curr_epoch = 0
        self.curr_iter = 0
        self.rm = RecordManager()

        # Init dicts for record and save for each iteration
        self.dict_record = OrderedDict()
        self.dict_save = OrderedDict()

        # Init setups
        self.optimizer = None
        self.scheduler = None
        self.scheduler_type = None
        self.minimizer = None
        self.clip_grad_norm = None

    def setup(
        self,
        optimizer,
        scheduler=None,
        scheduler_type=None,
        n_epochs=None,
        n_iters=None,
        minimizer=None,
        clip_grad_norm=None,
    ):
        params = self.rmodel.parameters()
        self.optimizer = self.generate_optimizer(optimizer, params)
        self.scheduler, self.scheduler_type = self.generate_schdeuler(
            scheduler, self.optimizer, scheduler_type, n_epochs, n_iters
        )
        self.minimizer = self.generate_minimizer(minimizer, self.rmodel, self.optimizer)
        self.clip_grad_norm = clip_grad_norm

    def fit(
        self,
        train_loader,
        n_epochs,
        n_iters=None,
        record_type="Epoch",
        save_path=None,
        save_type="Epoch",
        save_best=None,
        save_overwrite=False,
        refit=False,
    ):

        # Check Save and Record Values
        self._check_valid_options(save_type)
        self._check_valid_options(record_type)
        self._check_valid_save_path(save_path, save_type, save_overwrite)

        if refit:
            if self.rm.count == 0:
                raise ValueError("Please call load_dict for refitting.")
            # Update record and init
            self.rm.update(
                record_type=record_type, save_path=save_path, best_option=save_best
            )
            record_type = self.rm.record_type

            self.accumulated_epoch += -1
            start_epoch = self.curr_epoch - 1
            start_iter = self.curr_iter
        else:
            # Start record and save init
            self.rm.initialize(
                record_type=record_type, save_path=save_path, best_option=save_best
            )
            if (save_path is not None) and (self.accumulated_iter == 0):
                self.save_dict(save_path, is_init=True)

            start_epoch = 0
            start_iter = 0

        # Print train information
        self.rm.print(record_type, "[%s]" % self.__class__.__name__)
        self.rm.print(record_type, "Training Information.")
        self.rm.print(record_type, "-Epochs: %s" % n_epochs)
        self.rm.print(record_type, "-Optimizer: %s" % self.optimizer)
        self.rm.print(record_type, "-Scheduler: %s" % self.scheduler)
        self.rm.print(record_type, "-Minmizer: %s" % self.minimizer)
        self.rm.print(record_type, "-Save Path: %s" % save_path)
        self.rm.print(record_type, "-Save Type: %s" % str(save_type))
        self.rm.print(record_type, "-Record Type: %s" % str(record_type))
        self.rm.print(record_type, "-Device: %s" % self.device)

        # Start training
        for epoch in range(start_epoch, n_epochs):
            # Update current epoch and n_iters
            self.curr_epoch = epoch + 1
            self.accumulated_epoch += 1
            if n_iters is None:
                n_iters = len(train_loader)

            for i, train_data in enumerate(train_loader):
                # Update current iteration
                self.curr_iter = i + 1
                if self.curr_iter <= start_iter:  # For refit
                    continue
                self.accumulated_iter += 1
                is_last_batch = self.curr_iter == n_iters

                # Init records and dicts
                self._init_dicts_record_save()
                self.add_record_item("Epoch", self.accumulated_epoch)
                self.add_record_item("Iter", self.curr_iter)

                # Set train mode
                self.rmodel.train()

                # Update weight
                self.rm.progress_start()
                self._update_weight(train_data)
                self.rm.progress_end()

                # Eval mode
                if self._check_run_condition(record_type, is_last_batch):
                    if type(self).record_during_eval != self.record_during_eval:
                        self.rmodel.eval()
                        self.record_during_eval()
                    self.add_record_item("lr", self.optimizer.param_groups[0]["lr"])
                    self.rm.add(self.dict_record)
                    
                    # If record added, save dicts.
                    if self.rm.check_best(self.dict_record):
                        self.save_dict(save_path, save_type, is_best=True)

                # Save dicts
                if save_path is not None:
                    is_save_condition = self._check_run_condition(
                        save_type, is_last_batch
                    )
                    # Save if condition is satisfied or best or end of the epoch.
                    if is_save_condition or is_last_batch:
                        self.save_dict(save_path, save_type, is_best=False)

                # Update scheduler
                if self._check_run_condition(self.scheduler_type, is_last_batch):
                    self.scheduler.step()

                # Check number of iterations
                if is_last_batch:
                    break

            # Check nan values in records and terminate if it is true.
            # isnan = False
            # for key in self.rm.records.keys():
            #     if np.isnan(list(self.rm.records[key].history.values())[-1]):
            #         isnan = True
            #         break
            # if isnan:
            #     break

            start_iter = 0  # For refit

        if (save_path is not None) and (record_type is not None):
            self.rm.generate_summary()

    def calculate_cost(self, *inputs):
        raise NotImplementedError

    def record_during_eval(self, *args):
        raise NotImplementedError

    def _init_dicts_record_save(self):
        del self.dict_record, self.dict_save
        self.dict_record = OrderedDict()
        self.dict_save = OrderedDict()

    def add_record_item(self, key, val):
        if key in self.dict_record.keys():
            n = 1
            while key + "_" + str(n) in self.dict_record.keys():
                n = n + 1
            self.dict_record[key + "_" + str(n)] = val
        else:
            self.dict_record[key] = val

    def add_save_item(self, key, val):
        if key in self.dict_save.keys():
            n = 1
            while key + "_" + str(n) in self.dict_save.keys():
                n = n + 1
            self.dict_save[key + "_" + str(n)] = val
        else:
            self.dict_save[key] = val

    @staticmethod
    def _check_run_condition(run_type, is_last_batch):
        if run_type == "Epoch" and is_last_batch:
            return True
        elif run_type == "Iter":
            return True
        return False

    @staticmethod
    def generate_optimizer(optimizer, params):
        if isinstance(optimizer, str):
            optimizer = eval(
                optimizer.split("(")[0] + "(params," + optimizer.split("(")[1]
            )
        return optimizer

    @staticmethod
    def generate_schdeuler(
        scheduler, optimizer, scheduler_type=None, n_epochs=None, n_iters=None
    ):
        if isinstance(scheduler, str):
            # Step(milestones=[2, 4], gamma=0.1)
            if scheduler.startswith("Step("):
                scheduler = eval("MultiStepLR(optimizer, " + scheduler.split("(")[1])
                scheduler_type = "Epoch"
            # Cyclic(base_lr=0, max_lr=0.3)
            elif scheduler.startswith("Cyclic("):
                lr_steps = n_epochs * n_iters
                scheduler = eval(
                    "CyclicLR(optimizer, "
                    + scheduler.split("(")[1].split(")")[0]
                    + ", step_size_up=lr_steps/2, step_size_down=lr_steps/2)"
                )
                scheduler_type = "Iter"
            # Cosine
            elif "Cosine" == scheduler:
                scheduler = CosineAnnealingLR(optimizer, n_epochs, eta_min=0)
                scheduler_type = "Epoch"
            else:
                scheduler = eval(
                    scheduler.split("(")[0] + "(optimizer, " + scheduler.split("(")[1]
                )

        if (scheduler is not None) and (scheduler_type is None):
            raise ValueError(
                "The type of scheduler must be specified as 'Epoch' or 'Iter'."
            )

        return scheduler, scheduler_type

    @staticmethod
    def generate_minimizer(minimizer, model, optimizer):
        if isinstance(minimizer, str):
            minimizer = eval(
                minimizer.split("(")[0] + "(model, optimizer," + minimizer.split("(")[1]
            )
        return minimizer

    def _update_weight(self, *inputs):
        if self.minimizer is not None:
            self.minimizer.step(lambda x: self.calculate_cost(x), *inputs)
        else:
            cost = self.calculate_cost(*inputs)
            self.optimizer.zero_grad()
            cost.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.rmodel.parameters(), self.clip_grad_norm
                )
            self.optimizer.step()

    @staticmethod
    def _check_valid_options(key):
        if key in ["Epoch", "Iter", None]:
            pass
        else:
            raise ValueError(key, " is not valid. [Hint:'Epoch', 'Iter', None]")

    # Check and Create Path
    @staticmethod
    def _check_and_create_path(path, overwrite=False, file=False):
        if os.path.exists(path):
            if overwrite:
                logging.warning("Save file(s) will be overwritten:" + path)
            else:
                raise ValueError("[%s] is already exists." % (path))
        else:
            if not file:
                os.makedirs(path)

    def _check_valid_save_path(self, save_path, save_type, save_overwrite):
        if save_path is not None:
            if save_path[-1] == "/":
                save_path = save_path[:-1]
            # Save Initial Model
            self._check_and_create_path(save_path, overwrite=save_overwrite)
            self._check_and_create_path(
                save_path + "/init.pth", overwrite=save_overwrite, file=True
            )
            self._check_and_create_path(
                save_path + "/last.pth", overwrite=save_overwrite, file=True
            )
            self._check_and_create_path(
                save_path + "/best.pth", overwrite=save_overwrite, file=True
            )
            self._check_and_create_path(
                save_path + "/log.txt", overwrite=save_overwrite, file=True
            )

            if save_type in ["Epoch", "Iter"]:
                self._check_and_create_path(
                    save_path + "/epoch_iter/", overwrite=save_overwrite
                )
                self.save_dict(save_path, 0)
        else:
            if save_type is not None:
                raise ValueError("save_path should be given for save_type != None.")

    def save_dict(self, save_path, save_type=None, is_init=False, is_best=False):
        self.add_save_item("accumulated_epoch", self.accumulated_epoch)
        self.add_save_item("accumulated_iter", self.accumulated_iter)
        self.add_save_item("curr_epoch", self.curr_epoch)
        self.add_save_item("curr_iter", self.curr_iter)

        self.add_save_item("rmodel", self.rmodel.state_dict())
        if self.optimizer is not None:
            self.add_save_item("optimizer", self.optimizer.state_dict())
        if self.scheduler is not None:
            self.add_save_item("scheduler", self.scheduler.state_dict())
            self.add_save_item("scheduler_type", self.scheduler_type)
        if self.minimizer is not None:
            self.add_save_item("minimizer", self.minimizer.state_dict())
        if self.rm is not None:
            self.add_save_item("recordmanager", self.rm)
        if self.dict_record is not None:
            self.add_save_item("record_info", self.dict_record)

        # Save last and best
        torch.save(self.dict_save, save_path + "/last.pth")
        if is_best:
            torch.save(self.dict_save, save_path + "/best.pth")

        # Save init and epoch or iter
        if is_init:
            torch.save(self.dict_save, save_path + "/init.pth")

        if save_type == "Epoch":
            save_name = "/epoch_iter/%s_%s.pth" % (
                str(self.accumulated_epoch).zfill(5),
                str(0).zfill(5),
            )
            torch.save(self.dict_save, save_path + save_name)
        elif save_type == "Iter":
            save_name = "/epoch_iter/%s_%s.pth" % (
                str(self.accumulated_epoch).zfill(5),
                str(self.curr_iter).zfill(5),
            )
            torch.save(self.dict_save, save_path + save_name)

    def load_dict(self, save_path):
        save_dict = torch.load(save_path)
        default_keys = [
            "accumulated_epoch",
            "accumulated_iter",
            "curr_epoch",
            "curr_iter",
        ]

        for key in default_keys:
            if key not in save_dict.keys():
                raise ValueError("Not supported type of dictionary.")

        for key in [
            "accumulated_epoch",
            "accumulated_iter",
            "curr_epoch",
            "curr_iter",
            "scheduler_type",
        ]:
            setattr(self, key, save_dict[key])
            print("%s is successfully loaded!" % key)

        for key in ["record_info"]:
            self.dict_record = save_dict[key]
            print("%s is successfully loaded!" % key)

        for key in ["recordmanager"]:
            self.rm = save_dict[key]
            print("%s is successfully loaded!" % key)

        for key in ["rmodel", "optimizer", "scheduler", "minimizer"]:
            value = getattr(self, key)
            if value is not None:
                value.load_state_dict(save_dict[key])
                print("%s is successfully loaded!" % key)
