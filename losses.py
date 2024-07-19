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
        warmup_epochs: int = 0,
        alpha_final=2.0,
        alpha_init_factor=64,
        mu=0.05,
        **kwargs,
    ):
        super().__init__()

        # fixed values
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs if warmup_epochs > 0 else max_epochs // 10
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
        ce_loss = F.cross_entropy(preds[:, :-1, :, :], targets, reduction="none").mean(
            dim=(-2, -1)
        )

        if not training:
            return ce_loss.mean()
        else:
            abstention_rate = (
                (preds.argmax(dim=1) == preds.shape[1] - 1).float().mean().item()
            )
            abstention = torch.exp(F.log_softmax(preds, dim=1))[:, -1, :, :].mean(
                dim=(-2, -1)
            )
            abstention = torch.min(
                abstention,
                torch.tensor(
                    [1 - self.epsilon],
                    device=abstention.device,
                ),
            )

            if epoch < self.warmup_epochs:
                alpha_threshold = ((1 - abstention) * ce_loss).mean().item()
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
                loss = (1 - abstention) * ce_loss - self.alpha * torch.log(1 - abstention)
                # loss = (1 - abstention) * (
                #     ce_loss + torch.log(1 - abstention)
                # ) - self.alpha * torch.log(1 - abstention)
            return loss.mean(), abstention_rate


class IDACLoss(nn.Module):
    """Apply Informed Deep Abstaining Classifier Loss"""

    def __init__(
        self,
        noise_rate: float = 0,
        warmup_epochs: int = 10,
        alpha: float = 10,
        **kwargs,
    ):
        super().__init__()
        self.noise_rate = noise_rate
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.epsilon = 1e-7

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        training: bool = False,
        epoch: int = 0,
    ):
        ce_loss = F.cross_entropy(preds[:, :-1, :, :], targets, reduction="none").mean(
            dim=(-2, -1)
        )

        if not training:
            return ce_loss.mean()
        else:
            abstention_rate = (
                (preds.argmax(dim=1) == preds.shape[1] - 1).float().mean().item()
            )
            abstention = torch.exp(F.log_softmax(preds, dim=1))[:, -1, :, :].mean(
                dim=(-2, -1)
            )
            abstention = torch.min(
                abstention,
                torch.tensor(
                    1 - self.epsilon,
                    device=abstention.device,
                ),
            )

            if epoch < self.warmup_epochs:
                loss = ce_loss
            else:
                loss = (1 - abstention) * ce_loss + self.alpha * (
                    self.noise_rate - abstention.mean()
                ) ** 2
            return loss.mean(), abstention_rate


class SCELoss(nn.Module):
    """Apply Symmetric Cross Entropy Loss"""

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1,
        A: float = -6,
        num_classes: int = 151,
        **kwargs,
    ):
        super().__init__()
        assert alpha > 0 and beta >= 0 and A < 0
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, targets: Tensor):
        # CCE
        ce = self.cross_entropy(preds, targets)

        # RCE
        pred = F.softmax(preds, dim=1)
        if targets.ndim < 4:
            targets = F.one_hot(targets.long(), self.num_classes).movedim(-1, 1)
        log_targets = torch.clamp(torch.log(targets), min=self.A)
        rce = (-1 * torch.sum(pred * log_targets, dim=1)).mean()

        loss = self.alpha * ce + self.beta * rce
        return loss


class FocalLoss(nn.Module):
    """Apply Focal Loss"""

    def __init__(
        self,
        gamma: float = 2,
        alpha: float = 0.25,
        num_classes: int = 151,
        **kwargs,
    ):
        super().__init__()
        assert gamma >= 0 and alpha >= 0 and alpha <= 1
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        if targets.ndim < 4:
            targets = F.one_hot(targets.long(), self.num_classes).movedim(-1, 1).float()
        return sigmoid_focal_loss(
            inputs=preds,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="mean",
        )


class SFLoss(nn.Module):
    """Apply Symmetric Focal Loss"""

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1,
        A: float = -6,
        gamma: float = 2,
        focal_alpha: float = 0.25,
        num_classes: int = 151,
        **kwargs,
    ):
        super().__init__()
        assert (
            alpha > 0
            and beta >= 0
            and A < 0
            and gamma >= 0
            and focal_alpha >= 0
            and focal_alpha <= 1
        )
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.gamma = gamma
        self.focal_alpha = focal_alpha
        self.num_classes = num_classes
        self.focal = FocalLoss(gamma=gamma, alpha=focal_alpha, num_classes=num_classes)

    def forward(self, preds: Tensor, targets: Tensor):
        # focal
        focal = self.focal(preds, targets)

        # reverse focal
        if targets.ndim < 4:
            targets = F.one_hot(targets.long(), self.num_classes).movedim(-1, 1).float()
        targets = (targets - 0.5) * 2
        preds = F.sigmoid(preds)
        reverse_focal = self.focal(-targets * self.A, preds)

        loss = self.alpha * focal + self.beta * reverse_focal
        return loss
