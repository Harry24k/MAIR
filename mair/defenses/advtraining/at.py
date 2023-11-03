import torch
import torch.nn as nn

from ...attacks import PGD

from .advtrainer import AdvTrainer


class AT(AdvTrainer):
    r"""
    Adversarial training in 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Attributes:
        self.rmodel : rmodel.
        self.device : device where rmodel is.
        self.optimizer : optimizer.
        self.scheduler : scheduler (Automatically updated).
        self.curr_epoch : current epoch starts from 1 (Automatically updated).
        self.curr_iter : current iters starts from 1 (Automatically updated).

    Arguments:
        rmodel (nn.Module): rmodel to train.
        eps (float): strength of the attack or maximum perturbation.
        alpha (float): step size.
        steps (int): number of steps.
        random_start (bool): using random initialization of delta.
    """

    def __init__(self, rmodel, eps, alpha, steps, random_start=True):
        super().__init__(rmodel)
        self.atk = PGD(rmodel, eps, alpha, steps, random_start)

    def calculate_cost(self, train_data, reduction="mean"):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)

        adv_images = self.atk(images, labels)
        logits_adv = self.rmodel(adv_images)

        cost = nn.CrossEntropyLoss(reduction="none")(logits_adv, labels)
        self.add_record_item("CALoss", cost.mean().item())

        return cost.mean() if reduction == "mean" else cost
