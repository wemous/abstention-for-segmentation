import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.focal_loss import sigmoid_focal_loss


class CELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.ce(preds, targets)


class DACLoss(nn.Module):
    """Apply Deep Abstaining Classifier Loss"""

    def __init__(
        self,
        max_epochs: int,
        warmup_rate: float = 0.2,
        alpha_final=2.0,
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
        self.delta_alpha = None
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
            abstention_rate = (
                (preds.argmax(dim=1) == preds.shape[1] - 1).float().mean().item()
            )
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
                    self.delta_alpha = (self.alpha_final - self.alpha) / (
                        self.max_epochs - self.warmup_epochs
                    )
                    self.alpha_update_epoch = epoch
                else:
                    if epoch > self.alpha_update_epoch:
                        self.alpha += self.delta_alpha  # type: ignore
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
        self.alpha = alpha if alpha else max(1, noise_rate * 50)

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


class SCELoss(nn.Module):
    """Apply Symmetric Cross Entropy Loss"""

    def __init__(self, noise_rate: float, alpha=None, A=-4):
        super().__init__()
        self.alpha = alpha if alpha else 1 - noise_rate
        self.A = A
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred: Tensor, targets: Tensor):
        # CCE
        ce = self.cross_entropy(pred, targets)

        # RCE
        pred = F.softmax(pred, dim=1)
        targets = F.one_hot(targets, pred.shape[1]).float().movedim(-1, 1)
        targets = torch.clamp(targets, min=torch.e**self.A, max=1.0)
        rce = -torch.sum(pred * torch.log(targets), dim=1).mean()

        loss = self.alpha * ce + rce
        return loss


class ASCELoss(nn.Module):
    """Apply Abstaining Symmetric Cross Entropy Loss"""

    def __init__(
        self,
        noise_rate: float,
        max_epochs: int,
        warmup_epochs=None,
        warmup_rate=None,
        A=-2,
        alpha=None,
        beta=None,
        **kwargs,
    ):
        super().__init__()
        if warmup_epochs is not None:
            assert warmup_epochs < max_epochs
            self.warmup_epochs = warmup_epochs
        elif warmup_rate is not None:
            self.warmup_epochs = round(max_epochs * warmup_rate)
        else:
            warmup_rate = 0.2
            self.warmup_epochs = round(max_epochs * warmup_rate)

        self.noise_rate = noise_rate
        self.A = A
        self.alpha = alpha if alpha else 1 - noise_rate
        self.beta = beta if beta else max(1, noise_rate * 50)

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        training: bool = False,
        epoch: int = 0,
    ):
        ce_loss = F.cross_entropy(preds[:, :-1, :, :], targets)
        preds = torch.exp(F.log_softmax(preds, dim=1))
        targets = F.one_hot(targets, preds.shape[1] - 1).float().movedim(-1, 1)

        if not training:
            return self.alpha * ce_loss
        else:
            abstention_rate = (preds.argmax(dim=1) == preds.shape[1] - 1).float().mean()
            abstention = preds[:, -1, :, :].mean()
            regularization = 0
            rce = 0

            if epoch < self.warmup_epochs:
                loss = self.alpha * ce_loss

            else:
                targets = torch.clamp(targets, min=torch.e**self.A, max=1.0)
                rce = -torch.sum(preds[:, :-1, :, :] * torch.log(targets), dim=1).mean()
                ce_loss = (1 - abstention) * self.alpha * ce_loss
                rce = (1 - abstention) * rce
                regularization = self.beta * (self.noise_rate - abstention) ** 2
                loss = ce_loss + rce + regularization
            output = {
                "loss": loss,
                "CE loss": ce_loss,
                "RCE loss": rce,
                "Regularization": regularization,
                "Abstention": abstention,
                "Abstention Rate": abstention_rate,
            }
            return output


# --------------------------- irrelevant ------------------------------

# class FocalLoss(nn.Module):
#     """Apply Focal Loss"""

#     def __init__(
#         self,
#         gamma: float = 2,
#         alpha: float = 0.25,
#         num_classes: int = 151,
#         **kwargs,
#     ):
#         super().__init__()
#         assert gamma >= 0 and alpha >= 0 and alpha <= 1
#         self.gamma = gamma
#         self.alpha = alpha
#         self.num_classes = num_classes

#     def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
#         if targets.ndim < 4:
#             targets = F.one_hot(targets.long(), self.num_classes).movedim(-1, 1).float()
#         return sigmoid_focal_loss(
#             inputs=preds,
#             targets=targets,
#             alpha=self.alpha,
#             gamma=self.gamma,
#             reduction="mean",
#         )


# class SFLoss(nn.Module):
#     """Apply Symmetric Focal Loss"""

#     def __init__(
#         self,
#         alpha: float = 0.1,
#         beta: float = 1,
#         A: float = -6,
#         gamma: float = 2,
#         focal_alpha: float = 0.25,
#         num_classes: int = 151,
#         **kwargs,
#     ):
#         super().__init__()
#         assert (
#             alpha > 0
#             and beta >= 0
#             and A < 0
#             and gamma >= 0
#             and focal_alpha >= 0
#             and focal_alpha <= 1
#         )
#         self.alpha = alpha
#         self.beta = beta
#         self.A = A
#         self.gamma = gamma
#         self.focal_alpha = focal_alpha
#         self.num_classes = num_classes
#         self.focal = FocalLoss(gamma=gamma, alpha=focal_alpha, num_classes=num_classes)

#     def forward(self, preds: Tensor, targets: Tensor):
#         # focal
#         focal = self.focal(preds, targets)

#         # reverse focal
#         if targets.ndim < 4:
#             targets = F.one_hot(targets.long(), self.num_classes).movedim(-1, 1).float()
#         targets = (targets - 0.5) * 2
#         preds = F.sigmoid(preds)
#         reverse_focal = self.focal(-targets * self.A, preds)

#         loss = self.alpha * focal + self.beta * reverse_focal
#         return loss
