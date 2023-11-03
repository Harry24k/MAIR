import torch
import torch.nn as nn

from .advtrainer import AdvTrainer


class Standard(AdvTrainer):
    r"""
    Attributes:
        self.rmodel : rmodel.
        self.device : device where rmodel is.
        self.optimizer : optimizer.
        self.scheduler : scheduler (Automatically updated).
        self.curr_epoch : current epoch starts from 1 (Automatically updated).
        self.curr_iter : current iters starts from 1 (Automatically updated).

    Arguments:
        rmodel (nn.Module): rmodel to train.
    """

    def __init__(self, rmodel):
        super().__init__(rmodel)

    def calculate_cost(self, train_data, reduction="mean"):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.rmodel(images)

        cost = nn.CrossEntropyLoss(reduction="none")(logits, labels)
        self.add_record_item("CALoss", cost.mean().item())

        return cost.mean() if reduction == "mean" else cost
