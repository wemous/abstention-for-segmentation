import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.ce(preds, targets)


class GCELoss(nn.Module):
    """Apply Generalized Cross Entropy Loss"""

    def __init__(self, q: float = 0.01, **kwargs) -> None:
        super().__init__()
        self.q = q

    def forward(self, preds: Tensor, targets: Tensor):
        preds = F.softmax(preds, dim=1)
        targets = F.one_hot(targets, preds.shape[1]).float().movedim(-1, 1)
        loss = (1 - (preds * targets).sum(dim=1) ** self.q) / self.q
        return loss.mean()


class SCELoss(nn.Module):
    """Apply Symmetric Cross Entropy Loss"""

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
    """Apply Deep Abstaining Classifier Loss"""

    def __init__(
        self,
        max_epochs: int,
        warmup_rate: float = 0.2,
        alpha_final=2.5,
        alpha_init_factor=64,
        mu=0.05,
        **kwargs,
    ):
        super().__init__()

        # fixed values
        self.max_epochs = max_epochs
        self.warmup_epochs = round(max_epochs * warmup_rate)
        self.alpha_final = alpha_final
        self.alpha_init_factor = alpha_init_factor
        self.mu = mu
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
            abstention_rate = (preds.argmax(dim=1) == preds.shape[1] - 1).float().mean()
            abstention = torch.exp(F.log_softmax(preds, dim=1))[:, -1, :, :].mean()
            abstention = torch.min(
                abstention,
                torch.tensor(
                    1 - self.epsilon,
                    device=abstention.device,
                ),
            )
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
                    self.alpha = self.alpha_thershold_smoothed / self.alpha_init_factor  # type: ignore
                    self.alpha_step = (self.alpha_final - self.alpha) / (
                        self.max_epochs - self.warmup_epochs
                    )
                    self.alpha_update_epoch = epoch
                else:
                    if epoch > self.alpha_update_epoch:
                        self.alpha += self.alpha_step  # type: ignore
                        self.alpha_update_epoch = epoch
                ce_loss = (1 - abstention) * ce_loss
                regularization = -self.alpha * torch.log(1 - abstention)
                loss = ce_loss + regularization
                # loss = (1 - abstention) * (
                #     ce_loss + torch.log(1 - abstention)
                # ) - self.alpha * torch.log(1 - abstention)
            output = {
                "loss": loss,
                "CE loss": ce_loss,
                "Regularization": regularization,
                "Abstention": abstention,
                "Abstention Rate": abstention_rate,
            }
            return output


class IDACLoss(nn.Module):
    """Apply Informed Deep Abstaining Classifier Loss"""

    def __init__(
        self,
        noise_rate: float,
        max_epochs: int,
        warmup_rate: float = 0.2,
        alpha=None,
        **kwargs,
    ):
        super().__init__()
        self.noise_rate = noise_rate
        self.warmup_epochs = round(max_epochs * warmup_rate)
        self.alpha = alpha if alpha else max(1, noise_rate * 20)

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
            preds = torch.exp(F.log_softmax(preds, dim=1))
            abstention_rate = (preds.argmax(dim=1) == preds.shape[1] - 1).float().mean()
            abstention = preds[:, -1, :, :].mean()
            regularization = 0

            if epoch < self.warmup_epochs:
                loss = ce_loss
            else:
                ce_loss = (1 - abstention) * ce_loss
                regularization = self.alpha * (self.noise_rate - abstention) ** 2
                loss = ce_loss + regularization
            output = {
                "loss": loss,
                "CE loss": ce_loss,
                "Regularization": regularization,
                "Abstention": abstention,
                "Abstention Rate": abstention_rate,
            }
            return output
