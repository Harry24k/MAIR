import torch
import torch.nn as nn
import torch.nn.functional as F

from ...attacks import TPGD

from .advtrainer import AdvTrainer


class TRADES(AdvTrainer):
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
        beta (float): trade-off regularization parameter.
    """

    def __init__(self, rmodel, eps, alpha, steps, beta):
        super().__init__(rmodel)
        self.atk = TPGD(rmodel, eps, alpha, steps)
        self.beta = beta

    def calculate_cost(self, train_data, reduction="mean"):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits_clean = self.rmodel(images)
        loss_ce = nn.CrossEntropyLoss(reduction=reduction)(logits_clean, labels)

        adv_images = self.atk(images)
        logits_adv = self.rmodel(adv_images)
        probs_clean = F.softmax(logits_clean, dim=1)
        log_probs_adv = F.log_softmax(logits_adv, dim=1)
        loss_kl = nn.KLDivLoss(reduction="none")(log_probs_adv, probs_clean).sum(dim=1)

        cost = loss_ce + self.beta * loss_kl

        self.add_record_item("Loss", cost.mean().item())
        self.add_record_item("CELoss", loss_ce.mean().item())
        self.add_record_item("KLLoss", loss_kl.mean().item())

        return cost.mean() if reduction == "mean" else cost
