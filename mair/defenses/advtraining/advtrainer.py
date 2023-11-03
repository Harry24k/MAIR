from ..trainer import Trainer
from ...utils import get_subloader
from collections import OrderedDict

r"""
Base class for adversarial trainers.

Functions:
    self.record_rob : function for recording standard accuracy and robust accuracy against FGSM, PGD, and GN.

"""


class AdvTrainer(Trainer):
    def __init__(self, rmodel, device=None):
        super().__init__(rmodel, device)
        self.rob_dict = OrderedDict()

    def record_rob(
        self,
        train_loader,
        val_loader,
        eps=None,
        alpha=None,
        steps=None,
        std=None,
        n_train_limit=None,
        n_val_limit=None,
    ):
        if (alpha is None) and (steps is not None):
            raise ValueError("Both alpha and steps should be given for PGD.")
        elif (alpha is not None) and (steps is None):
            raise ValueError("Both alpha and steps should be given for PGD.")

        self.rob_dict["train_loader"] = get_subloader(train_loader, n_train_limit)
        self.rob_dict["val_loader"] = get_subloader(val_loader, n_val_limit)
        self.rob_dict["loaders"] = {
            "(Tr)": self.rob_dict["train_loader"],
            "(Val)": self.rob_dict["val_loader"],
        }
        self.rob_dict["eps"] = eps
        self.rob_dict["alpha"] = alpha
        self.rob_dict["steps"] = steps
        self.rob_dict["std"] = std

    def record_during_eval(self):
        for flag, loader in self.rob_dict["loaders"].items():
            self.dict_record["Clean" + flag] = self.rmodel.eval_accuracy(loader)

            eps = self.rob_dict.get("eps")
            if eps is not None:
                self.dict_record["FGSM" + flag] = self.rmodel.eval_rob_accuracy_fgsm(
                    loader, eps=eps, verbose=False
                )
                steps = self.rob_dict.get("steps")
                alpha = self.rob_dict.get("alpha")
                if steps is not None:
                    self.dict_record["PGD" + flag] = self.rmodel.eval_rob_accuracy_pgd(
                        loader, eps=eps, alpha=alpha, steps=steps, verbose=False
                    )

            std = self.rob_dict.get("std")
            if std is not None:
                self.dict_record["GN" + flag] = self.rmodel.eval_rob_accuracy_gn(
                    loader, std=std, verbose=False
                )
