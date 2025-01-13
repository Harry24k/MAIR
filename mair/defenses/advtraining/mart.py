import torch
import torch.nn as nn
import torch.nn.functional as F

from ...attacks import PGD

from .advtrainer import AdvTrainer


class MART(AdvTrainer):
    r"""
    MART in 'Improving Adversarial Robustness Requires Revisiting Misclassified Examples'
    [https://openreview.net/forum?id=rklOg6EFwS]
    [https://github.com/YisenWang/MART]

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
        random_start (bool): using random initialization of delta.
    """

    def __init__(self, rmodel, eps, alpha, steps, beta, random_start=True):
        super().__init__(rmodel)
        self.atk = PGD(rmodel, eps, alpha, steps, random_start)
        self.beta = beta

    def calculate_cost(self, train_data, reduction="mean"):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)

        adv_images = self.atk(images, labels)

        logits_clean = self.rmodel(images)
        logits_adv = self.rmodel(adv_images)

        probs_adv = F.softmax(logits_adv, dim=1)

        # Caculate BCELoss
        tmp1 = torch.argsort(probs_adv, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
        loss_bce_adv = F.cross_entropy(
            logits_adv, labels, reduction="none"
        ) + F.nll_loss(torch.log(1.0001 - probs_adv + 1e-12), new_y, reduction="none")

        # Caculate KLLoss
        probs_clean = torch.clamp(F.softmax(logits_clean, dim=1), min=1e-12)
        log_prob_adv = torch.log(probs_adv + 1e-12)
        loss_kl = torch.sum(
            nn.KLDivLoss(reduction="none")(log_prob_adv, probs_clean), dim=1
        )
        true_probs = torch.gather(
            probs_clean, 1, (labels.unsqueeze(1)).long()
        ).squeeze()
        loss_weighted_kl = loss_kl * (1.0000001 - true_probs)

        cost = loss_bce_adv + self.beta * loss_weighted_kl

        self.add_record_item("Loss", cost.mean().item())
        self.add_record_item("BALoss", loss_bce_adv.mean().item())
        self.add_record_item("WKLoss", loss_weighted_kl.mean().item())

        return cost.mean() if reduction == "mean" else cost
