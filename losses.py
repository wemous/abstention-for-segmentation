import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

nn.CrossEntropyLoss()


class CELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(preds, targets)


class GCELoss(nn.Module):
    """Generalized Cross Entropy Loss"""

    def __init__(self, q: float = 0.01, **kwargs) -> None:
        super().__init__()
        self.q = q

    def forward(self, preds: Tensor, targets: Tensor):
        preds = F.softmax(preds, dim=1)
        targets = F.one_hot(targets, preds.shape[1]).float().movedim(-1, 1)
        loss = (1 - (preds * targets).sum(dim=1) ** self.q) / self.q
        return loss.mean()


class SCELoss(nn.Module):
    """Symmetric Cross Entropy Loss"""

    def __init__(self, noise_rate: float, alpha=None, A=-4, **kwargs):
        super().__init__()
        self.alpha = alpha if alpha else 1 - noise_rate
        self.A = A

    def forward(self, preds: Tensor, targets: Tensor):
        # CCE
        ce = F.cross_entropy(preds, targets)

        # RCE
        preds = F.softmax(preds, dim=1)
        targets = F.one_hot(targets, preds.shape[1]).float().movedim(-1, 1)
        targets = torch.clamp(targets, min=torch.e**self.A, max=1.0)
        rce = -torch.sum(preds * torch.log(targets), dim=1).mean()

        loss = self.alpha * ce + rce
        return loss


class DACLoss(nn.Module):
    """Deep Abstaining Classifier Loss"""

    def __init__(
        self,
        max_epochs: int,
        warmup_epochs: int = 10,
        alpha_final=1.0,
        mu=0.05,
        rho=64,
        **kwargs,
    ):
        super().__init__()

        # fixed values
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.alpha_final = alpha_final
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-7

        # values that will be updated
        self.alpha = None
        self.alpha_step = None
        self.alpha_update_epoch = 0
        self.alpha_thershold_smoothed = None

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        training: bool = False,
        epoch: int = 0,
    ):
        ce_loss = F.cross_entropy(preds[:, :-1, :, :], targets)

        if not training:
            return ce_loss
        else:
            preds = F.log_softmax(preds, dim=1).exp()
            abstention_rate = (preds.argmax(dim=1) == (preds.shape[1] - 1)).float().mean()
            abstention = preds[:, -1, :, :].mean().clamp_max(1 - self.epsilon)
            regularization = 0

            if epoch < self.warmup_epochs:
                alpha_threshold = ((1 - abstention) * ce_loss).item()
                # initialize the smoothed moving average of alpha threshold
                if not self.alpha_thershold_smoothed:
                    self.alpha_thershold_smoothed = alpha_threshold
                # update the smoothed moving average of alpha threshold
                else:
                    self.alpha_thershold_smoothed = (
                        1 - self.mu
                    ) * self.alpha_thershold_smoothed + self.mu * alpha_threshold
                loss = ce_loss
            else:
                # initialize alpha once warmup is finished
                if not self.alpha:
                    self.alpha = self.alpha_thershold_smoothed / self.rho
                    self.alpha_step = (self.alpha_final - self.alpha) / (
                        self.max_epochs - self.warmup_epochs
                    )
                    self.alpha_update_epoch = epoch
                else:
                    if epoch > self.alpha_update_epoch:
                        self.alpha += self.alpha_step
                        self.alpha_update_epoch = epoch
                regularization = -self.alpha * torch.log(1 - abstention)
                loss = (1 - abstention) * ce_loss + regularization
            output = {
                "loss": loss,
                "CE loss": ce_loss,
                "Regularization": regularization,
                "Abstention": abstention,
                "Abstention Rate": abstention_rate,
            }
            return output


class IDACLoss(nn.Module):
    """Informed Deep Abstaining Classifier Loss"""

    def __init__(
        self,
        noise_rate: float,
        warmup_epochs: int = 10,
        alpha=1.0,
        **kwargs,
    ):
        super().__init__()
        self.noise_rate = noise_rate
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        training: bool = False,
        epoch: int = 0,
    ):
        ce_loss = F.cross_entropy(preds[:, :-1, :, :], targets)

        if not training:
            return ce_loss
        else:
            preds = F.log_softmax(preds, dim=1).exp()
            abstention_rate = (preds.argmax(dim=1) == (preds.shape[1] - 1)).float().mean()
            abstention = preds[:, -1, :, :].mean()
            regularization = 0

            if epoch < self.warmup_epochs:
                loss = ce_loss
            else:
                regularization = self.alpha * (self.noise_rate - abstention) ** 2
                loss = (1 - abstention) * ce_loss + regularization
            output = {
                "loss": loss,
                "CE loss": ce_loss,
                "Regularization": regularization,
                "Abstention": abstention,
                "Abstention Rate": abstention_rate,
            }
            return output


class DiceLoss(nn.Module):
    def __init__(self, reduction="mean", **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        abstention=0,
        gamma=1.0,
    ) -> Tensor:
        dims = (-1, -2)
        preds = preds.log_softmax(dim=1).exp()
        targets = F.one_hot(targets, preds.shape[1]).movedim(-1, 1)
        instersection = (preds * targets).sum(dims)
        sum_preds = preds.sum(dims)
        sum_targets = targets.sum(dims)
        scores = 2 * instersection / (sum_preds + sum_targets)
        scores = (scores / (1 - abstention * gamma)).clamp_max(1)
        loss = 1 - scores
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            loss = loss
        return loss


class ADSLoss(nn.Module):
    """Abstaining Dice Segmenter Loss"""

    def __init__(
        self,
        max_epochs: int,
        warmup_epochs: int = 20,
        alpha_final=1.0,
        gamma=1.0,
        mu=0.05,
        rho=64,
        **kwargs,
    ):
        super().__init__()

        # fixed values
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.alpha_final = alpha_final
        self.gamma = gamma
        self.mu = mu
        self.rho = rho
        self.dice = DiceLoss(reduction="none")

        # values that will be updated
        self.alpha = None
        self.alpha_step = None
        self.alpha_update_epoch = 0
        self.alpha_thershold_smoothed = None

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        abstention=None,
        epoch: int = 0,
    ):
        dice_loss = self.dice(preds, targets).mean()

        if abstention is None:
            return dice_loss
        else:
            abstention = F.logsigmoid(abstention).exp()
            abstention = abstention.clamp_max(1 - 1e-7)
            regularization = 0

            if epoch < self.warmup_epochs:
                alpha_threshold = ((1 - abstention.mean()) * dice_loss).item()
                if not self.alpha_thershold_smoothed:
                    self.alpha_thershold_smoothed = alpha_threshold
                else:
                    self.alpha_thershold_smoothed = (
                        1 - self.mu
                    ) * self.alpha_thershold_smoothed + self.mu * alpha_threshold
                loss = dice_loss
            else:
                if not self.alpha:
                    self.alpha = self.alpha_thershold_smoothed / self.rho
                    self.alpha_step = (self.alpha_final - self.alpha) / (
                        self.max_epochs - self.warmup_epochs
                    )
                    self.alpha_update_epoch = epoch
                else:
                    if epoch > self.alpha_update_epoch:
                        self.alpha += self.alpha_step
                        self.alpha_update_epoch = epoch
                dice_loss = self.dice(preds, targets, abstention, self.gamma)
                regularization = -self.alpha * torch.log(1 - abstention)
                loss = (1 - abstention) * dice_loss + regularization
                regularization = regularization.mean()
            output = {
                "loss": loss.mean(),
                "Dice loss": dice_loss.mean(),
                "Regularization": regularization,
                "Abstention": abstention.mean(),
            }
            return output
