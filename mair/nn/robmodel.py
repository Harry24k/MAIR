from collections import OrderedDict

import torch
import torch.nn as nn

from ..attacks import FGSM, PGD, PGDL2, GN, AutoAttack, MultiAttack

from ..utils import get_accuracy
from .modules.normalize import Normalize


class RobModel(nn.Module):
    r"""
    Wrapper class for PyTorch models.
    """

    def __init__(
        self,
        model,
        n_classes,
        normalization_used={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        device=None,
    ):
        super().__init__()

        # Set device
        if device is None:
            device = next(model.parameters()).device

        # Set n_class
        assert isinstance(n_classes, int)
        self.register_buffer("n_classes", torch.tensor(n_classes))
        self.register_buffer("mean", torch.tensor(normalization_used["mean"]))
        self.register_buffer("std", torch.tensor(normalization_used["std"]))

        # Set model structure
        self.model = model.to(device)
        self.to(device)

    def add_normalize(self):
        if getattr(self, "mean") is not None and getattr(self, "std") is not None:
            pass
        else:
            raise ValueError("There is no normalization to be added.")

        self.model = nn.Sequential(
            Normalize(self.mean.cpu().numpy(), self.std.cpu().numpy()), self.model
        )

        delattr(self, "mean")
        delattr(self, "std")

    def forward(self, inputs, *args, **kargs):
        out = self.model(inputs, *args, **kargs)
        return out

    # Load from state dict
    def load_dict(self, save_path):
        state_dict = torch.load(save_path, map_location="cpu")
        self.load_state_dict_auto(state_dict["rmodel"])
        print("Model loaded.")

        if "record_info" in state_dict.keys():
            print("Record Info:")
            print(state_dict["record_info"])

    # DataParallel considered version of load_state_dict.
    def load_state_dict_auto(self, state_dict):
        state_dict = self._convert_dict_auto(state_dict)
        self.load_state_dict(state_dict)

    # Automatically changes pararell mode and non-parallel mode.
    def _convert_dict_auto(self, state_dict):
        keys = state_dict.keys()

        save_parallel = any(key.startswith("model.module.") for key in keys)
        curr_parallel = any(
            key.startswith("model.module.") for key in self.state_dict().keys()
        )
        if save_parallel and not curr_parallel:
            new_state_dict = {
                k.replace("model.module.", "model."): v for k, v in state_dict.items()
            }
            return new_state_dict
        elif curr_parallel and not save_parallel:
            new_state_dict = {
                k.replace("model.", "model.module."): v for k, v in state_dict.items()
            }
            return new_state_dict
        else:
            return state_dict

    def save_dict(self, save_path):
        save_dict = OrderedDict()
        save_dict["rmodel"] = self.state_dict()
        torch.save(save_dict, save_path)

    def set_parallel(self):
        self.model = torch.nn.DataParallel(self.model)
        return self

    def named_parameters_with_module(self):
        module_by_name = {}
        for name, module in self.named_modules():
            module_by_name[name] = module

        for name, param in self.named_parameters():
            if "." in name:
                module_name = name.rsplit(".", maxsplit=1)[0]
                yield name, param, module_by_name[module_name]
            else:
                yield name, param, None

    #################################################
    ############# Evaluate Robustness ###############
    #################################################
    @torch.no_grad()
    def eval_accuracy(self, data_loader):
        return get_accuracy(self, data_loader)

    def eval_rob_accuracy(self, data_loader, atk, **kargs):
        return atk.save(data_loader, return_verbose=True, **kargs)[0]

    def eval_rob_accuracy_gn(self, data_loader, std, **kargs):
        atk = GN(self, std=std)
        return self.eval_rob_accuracy(data_loader, atk, **kargs)

    def eval_rob_accuracy_fgsm(self, data_loader, eps, **kargs):
        atk = FGSM(self, eps=eps)
        return self.eval_rob_accuracy(data_loader, atk, **kargs)

    def eval_rob_accuracy_pgd(
        self,
        data_loader,
        eps,
        alpha,
        steps,
        random_start=True,
        restart_num=1,
        norm="Linf",
        **kargs
    ):
        if norm == "Linf":
            atk = PGD(
                self, eps=eps, alpha=alpha, steps=steps, random_start=random_start
            )
        elif norm == "L2":
            atk = PGDL2(
                self, eps=eps, alpha=alpha, steps=steps, random_start=random_start
            )
        else:
            raise ValueError("Invalid norm.")

        if restart_num > 1:
            atk = MultiAttack([atk] * restart_num)
        return self.eval_rob_accuracy(data_loader, atk, **kargs)

    def eval_rob_accuracy_autoattack(
        self, data_loader, eps, version="standard", norm="Linf", **kargs
    ):
        atk = AutoAttack(
            self, norm=norm, eps=eps, version=version, n_classes=self.n_classes
        )
        return self.eval_rob_accuracy(data_loader, atk, **kargs)

    ##############################################################
    ############# Evaluate Generalization Measures ###############
    ##############################################################
