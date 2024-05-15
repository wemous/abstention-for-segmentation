import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target) -> torch.Tensor:
        return self.ce(pred, target)


class DACLoss(torch.nn.Module):
    """Apply deep abstaining classifier loss"""

    def __init__(
        self,
        max_epochs: int,
        warmup_epochs: int,
        alpha_final=1.0,
        alpha_init_factor=64,
        mu=0.05,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ce = CELoss()
        # fixed values
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
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
        pred,
        target,
        train: bool = False,
        epoch: int = 0,
        *args,
        **kwargs,
    ):
        assert pred.shape[1] == target.shape[1] + 1
        ce_loss = self.ce(pred[:, :-1, :, :], target)

        if train:
            abstention_rate = (
                (pred.argmax(dim=1) == pred.shape[1] - 1).float().mean().item()
            )
            abstain = torch.exp(F.log_softmax(pred, dim=1))[:, -1, :, :].mean()
            abstain = torch.min(
                abstain,
                torch.tensor(
                    [1 - self.epsilon],
                    device=abstain.device,
                ),
            )

            if epoch < self.warmup_epochs:
                alpha_threshold = (1 - abstain.item()) * ce_loss.item()
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
                        self.alpha += self.delta_alpha
                        self.alpha_update_epoch = epoch
                loss = (1 - abstain) * ce_loss - self.alpha * torch.log(1 - abstain)
                # loss = (1 - abstain) * (
                #     ce_loss + torch.log(1 - abstain)
                # ) - self.alpha * torch.log(1 - abstain)
            return loss, abstention_rate
        else:
            return ce_loss


class SCELoss(torch.nn.Module):
    """Apply symmetric cross entropy loss"""

    def __init__(self, alpha, beta):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target, *args, **kwargs):
        # CCE
        ce = self.cross_entropy(pred, target)

        # RCE
        pred = F.softmax(pred, dim=1)
        # q. why clip the predictions?
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        # label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        # labels are already in one_hot for due to binary masks
        labels = torch.clamp(target, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(labels), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class WSCELoss(torch.nn.Module):
    """Apply symmetric cross entropy loss with weights for silver interpolations"""

    def __init__(self, alpha, beta, ws=0.5, wg=1):
        super(WSCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ws = ws
        self.wg = wg
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred, target, label, *args, **kwargs):
        # CCE
        ce_tensor = self.cross_entropy(pred, target)  # B,H,W
        weight_tensor = torch.ones(size=label.size(), device=label.device)
        weight_tensor[label == 1] = self.ws
        weight_tensor[label == 0] = self.wg
        ce_tensor = ce_tensor.reshape(ce_tensor.shape[0], -1).mean()
        ce = (ce_tensor * weight_tensor).mean()

        # RCE
        pred = F.softmax(pred, dim=1)
        # q. why clip the predictions?
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        # label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        # labels are already in one_hot for due to binary masks
        target = torch.clamp(target, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(target), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
