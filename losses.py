"""
losses.py — Loss functions for noise-robust medical image segmentation.

Architecture note
-----------------
For DAC, IDAC, GAC, and SAC: the model must output k+1 channels, where the
last channel is the abstention logit. The base loss is computed on preds[:, :-1].

For ADS: the model outputs k channels as normal; a parallel abstention head
(AdaptiveAvgPool2d -> Flatten -> Linear) outputs a (batch, k) tensor of
per-class abstention logits, which is passed separately to forward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CELoss(nn.Module):
    """Standard cross entropy loss.

    A thin wrapper around nn.CrossEntropyLoss included for a consistent
    interface across all losses in this module (all accept **kwargs so they
    can be instantiated uniformly from a config dict).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        return self.ce(preds, targets)


class DACLoss(nn.Module):
    """Deep Abstaining Classifier loss (Thulasidasan et al., ICML 2019).

    Augments cross entropy with an abstention option: the model outputs k+1
    channels, where the (k+1)-th channel is the abstention logit. The loss is:

        L_DAC = (1 - p_{k+1}) * CE + α * (-log(1 - p_{k+1}))

    The penalty α is initialized from the warm-up loss statistics and then
    linearly ramped up to alpha_final over the remaining epochs.

    Args:
        max_epochs:     Total number of training epochs.
        alpha_final:    Target value for α at the end of training.
        warmup_epochs:  Epochs with no abstention penalty; used to estimate
                        the initial α from a smoothed moving average of the
                        scaled CE loss.
        mu:             Smoothing factor for the warm-up moving average.
        rho:            Divisor used to derive the initial α from the
                        smoothed warm-up statistic (α_init = β̃ / rho).
    """

    def __init__(
        self,
        max_epochs: int,
        alpha_final: float = 2.0,
        warmup_epochs: int = 18,
        mu: float = 0.05,
        rho: float = 64,
        **kwargs,
    ):
        super().__init__()

        # --- fixed hyperparameters ---
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.alpha_final = alpha_final
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-7
        self.ce = nn.CrossEntropyLoss()

        # --- stateful values updated across epochs ---
        self.alpha = None  # current penalty weight; None until warmup ends
        self.alpha_step = None  # fixed increment per epoch after warmup
        self.alpha_update_epoch = 0  # tracks the last epoch α was incremented
        self.alpha_thershold_smoothed = None  # EMA of (1 - p_{k+1}) * CE during warmup

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        training: bool = False,
        epoch: int = 0,
    ):
        """
        Args:
            preds:    (B, C+1, H, W) — logits for C classes + 1 abstention channel.
            targets:  (B, H, W)      — integer class labels in [0, C).
            training: If False, returns a plain CE scalar (abstention channel ignored).
            epoch:    Current epoch index (0-based); used to update α.

        Returns:
            training=False -> scalar Tensor (CE loss on the k class channels).
            training=True  -> dict with keys:
                "loss"            — total loss (scalar)
                "CE loss"         — base cross entropy (scalar)
                "Regularization"  — abstention penalty term (scalar)
                "Abstention"      — mean batch abstention probability p_{k+1}
                "Abstention Rate" — fraction of pixels where abstention is argmax
        """
        # Base CE computed only over the k class channels, ignoring abstention.
        ce_loss = self.ce(preds[:, :-1], targets)

        if not training:
            return ce_loss

        # Fraction of pixels for which abstention is the argmax prediction.
        abstention_rate = (preds.argmax(dim=1) == (preds.shape[1] - 1)).float().mean()

        # Mean abstention probability p_{k+1}, clamped away from 1 to keep log finite.
        abstention = F.log_softmax(preds, dim=1).exp()[:, -1].clamp_max(1 - self.epsilon).mean()

        regularization = 0

        if epoch < self.warmup_epochs:
            # During warm-up, track a smoothed moving average of (1 - p_{k+1}) * CE.
            # This is used to set a data-driven starting value for α once warmup ends.
            alpha_threshold = ((1 - abstention) * ce_loss).item()
            if not self.alpha_thershold_smoothed:
                # First step: initialize the EMA.
                self.alpha_thershold_smoothed = alpha_threshold
            else:
                self.alpha_thershold_smoothed = (
                    1 - self.mu
                ) * self.alpha_thershold_smoothed + self.mu * alpha_threshold
            loss = ce_loss

        else:
            if not self.alpha:
                # First epoch after warmup: set α_init = β̃ / rho and compute the
                # fixed per-epoch increment to reach alpha_final by the last epoch.
                self.alpha = self.alpha_thershold_smoothed / self.rho
                self.alpha_step = (self.alpha_final - self.alpha) / (self.max_epochs - self.warmup_epochs - 1)
                self.alpha_update_epoch = epoch
            else:
                # Increment α by one step at the start of each new epoch.
                if epoch > self.alpha_update_epoch:
                    self.alpha += self.alpha_step
                    self.alpha_update_epoch = epoch

            # DAC regularization: -α * log(1 - p_{k+1})
            # Penalizes abstention, pushing p_{k+1} toward 0 over time.
            regularization = -self.alpha * torch.log(1 - abstention)
            loss = (1 - abstention) * ce_loss + regularization

        return {
            "loss": loss,
            "CE loss": ce_loss,
            "Regularization": regularization,
            "Abstention": abstention,
            "Abstention Rate": abstention_rate,
        }


class IDACLoss(nn.Module):
    """Informed Deep Abstaining Classifier loss (Schneider et al., ICONIP 2025).

    Refines DAC by replacing its incremental penalty with a term that minimizes
    the squared divergence between the prior noise rate η̃ and the current
    batch-wise abstention rate η̂:

        L_IDAC = (1 - p_{k+1}) * CE + α * (η̃ - η̂)²

    This provides a more targeted supervisory signal than DAC's push toward zero:
    the model is guided to abstain on approximately η̃ of samples rather than
    being penalized for any abstention at all.

    Args:
        noise_rate:     Prior estimate η̃ of the dataset noise rate.
        alpha:          Fixed penalty weight (not annealed, unlike DAC/GAC/SAC).
        warmup_epochs:  Epochs before the regularization term is activated.
    """

    def __init__(
        self,
        noise_rate: float = 0.0,
        alpha: float = 1.0,
        warmup_epochs: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.noise_rate = noise_rate
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        training: bool = False,
        epoch: int = 0,
    ):
        """
        Args:
            preds:    (B, C+1, H, W) — logits for C classes + 1 abstention channel.
            targets:  (B, H, W)      — integer class labels in [0, C).
            training: If False, returns a plain CE scalar.
            epoch:    Current epoch index (0-based).

        Returns:
            training=False -> scalar Tensor.
            training=True  -> dict with keys:
                "loss", "CE loss", "Regularization", "Abstention", "Abstention Rate"
        """
        ce_loss = self.ce(preds[:, :-1], targets)

        if not training:
            return ce_loss

        abstention_rate = (preds.argmax(dim=1) == (preds.shape[1] - 1)).float().mean()
        # Batch-wise abstention rate η̂ = mean(p_{k+1})
        abstention = F.log_softmax(preds, dim=1).exp()[:, -1].mean()
        regularization = 0

        if epoch >= self.warmup_epochs:
            # Squared divergence between prior noise rate and observed abstention rate.
            regularization = self.alpha * (self.noise_rate - abstention) ** 2
            loss = (1 - abstention) * ce_loss + regularization
        else:
            loss = ce_loss

        return {
            "loss": loss,
            "CE loss": ce_loss,
            "Regularization": regularization,
            "Abstention": abstention,
            "Abstention Rate": abstention_rate,
        }


class GCELoss(nn.Module):
    """Generalized Cross Entropy loss (Zhang & Sabuncu, NeurIPS 2018).

    Interpolates between CE (q → 0) and MAE (q = 1) via the negative
    Box-Cox transformation applied to the model's predicted probability
    for the true class:

        L_GCE = (1 - p_y^q) / q

    For noisy labels, smaller q down-weights low-confidence predictions
    less aggressively than CE, reducing the gradient contribution of
    likely-corrupted samples.

    Args:
        q: Box-Cox exponent in (0, 1]. Smaller values approach CE behavior;
           larger values approach MAE behavior and improve noise robustness.
    """

    def __init__(self, q: float = 0.1, **kwargs) -> None:
        super().__init__()
        assert 0 < q <= 1, "q must be in (0, 1]"
        self.q = q

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            preds:   (B, C, H, W) — raw logits.
            targets: (B, H, W)    — integer class labels in [0, C).

        Returns:
            Scalar loss Tensor.
        """
        # Clamp softmax output to avoid log(0) and log(1) instability.
        preds = F.softmax(preds, dim=1).clamp(min=1e-15, max=1 - 1e-15)
        # Extract the predicted probability for the true class at each pixel.
        true_preds = torch.gather(preds, 1, targets.unsqueeze(1))
        loss = (1 - true_preds**self.q) / self.q
        return loss.mean()


class GACLoss(nn.Module):
    """Generalized Abstaining Classifier loss (ours).

    Combines the universal abstention framework with GCE as the base loss:

        L_GAC = (1 - p_{k+1}) * L_GCE + α * |log((1 - η̃) / (1 - p_{k+1}))|

    This creates a dual defence: GCE's bounded loss attenuates the gradient
    contribution of noisy samples, while abstention filters out the most
    severely corrupted ones entirely.

    α is annealed via a power-law schedule (Eq. 4 in the paper):

        α(e) = alpha_final * ((e - L) / (E - L))^gamma

    where e is the current epoch, L is the warmup duration, and E is the
    total number of epochs. gamma > 1 gives sublinear growth (slow start,
    fast finish); gamma < 1 gives superlinear growth; gamma = 1 is linear.

    Args:
        max_epochs:     Total training epochs (E).
        noise_rate:     Prior noise estimate η̃. Set to 0 to recover DAC-style
                        regularization with the power-law schedule.
        alpha_final:    Maximum value of α reached at the final epoch.
        gamma:          Power-law growth factor controlling the annealing shape.
        warmup_epochs:  Epochs before abstention is activated (L).
        q:              GCE exponent, passed to the internal GCELoss instance.
    """

    def __init__(
        self,
        max_epochs: int,
        noise_rate: float = 0.0,
        alpha_final: float = 2.0,
        gamma: float = 2.0,
        warmup_epochs: int = 15,
        q: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.noise_rate = noise_rate
        self.alpha_final = alpha_final
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.alpha_update_epoch = 0
        self.gce = GCELoss(q)
        self.epsilon = 1e-7  # numerical guard for log(1 - abstention)

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        training: bool = False,
        epoch: int = 0,
    ):
        """
        Args:
            preds:    (B, C+1, H, W) — logits for C classes + 1 abstention channel.
            targets:  (B, H, W)      — integer class labels in [0, C).
            training: If False, returns a plain GCE scalar.
            epoch:    Current epoch index (0-based).

        Returns:
            training=False -> scalar Tensor.
            training=True  -> dict with keys:
                "loss", "GCE loss", "Regularization", "Abstention", "Abstention Rate"
        """
        gce_loss = self.gce(preds[:, :-1], targets)

        if not training:
            return gce_loss

        abstention_rate = (preds.argmax(dim=1) == (preds.shape[1] - 1)).float().mean()
        abstention = F.softmax(preds, dim=1)[:, -1].clamp_max(1 - self.epsilon).mean()
        regularization = 0

        if epoch >= self.warmup_epochs:
            # Update α once per epoch via the power-law schedule.
            if epoch > self.alpha_update_epoch:
                self.alpha = (
                    self.alpha_final
                    * ((epoch - self.warmup_epochs + 1) / (self.max_epochs - self.warmup_epochs)) ** self.gamma
                )
                self.alpha_update_epoch = epoch

            # Informed regularization: |log((1 - η̃) / (1 - p_{k+1}))|
            # The absolute value ensures the penalty is symmetric around η̃:
            # both over- and under-abstention relative to η̃ are penalized.
            regularization = self.alpha * abs(torch.log((1 - self.noise_rate) / (1 - abstention)))
            loss = (1 - abstention) * gce_loss + regularization
        else:
            loss = gce_loss

        return {
            "loss": loss,
            "GCE loss": gce_loss,
            "Regularization": regularization,
            "Abstention": abstention,
            "Abstention Rate": abstention_rate,
        }


class SCELoss(nn.Module):
    """Symmetric Cross Entropy loss (Wang et al., ICCV 2019).

    Combines standard CE with a Reverse Cross Entropy (RCE) term:

        L_SCE = α * CE(p, q) + β * RCE(p, q)
              = α * CE + β * (-Σ p_i * log(q_i))

    where p is the model's softmax output and q is the (smoothed) one-hot
    target. CE handles the forward direction (label → prediction), while RCE
    handles the reverse (prediction → label), making the combined loss more
    robust to noisy labels.

    The target distribution is clamped to 1e-4 before taking the log to
    avoid log(0) on the zero entries of the one-hot vector.

    Args:
        alpha: Weight for the standard CE term.
        beta:  Weight for the RCE term.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 1.0, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            preds:   (B, C, H, W) — raw logits.
            targets: (B, H, W)    — integer class labels in [0, C).

        Returns:
            Scalar loss Tensor.
        """
        # Standard cross entropy: CE(p, q)
        ce = self.ce(preds, targets)

        # Reverse cross entropy: RCE(p, q) = -Σ p_i * log(q_i)
        # q_i is the one-hot target, clamped to avoid log(0).
        preds = F.softmax(preds, dim=1).clamp_min(1e-7)
        targets_one_hot = F.one_hot(targets, preds.shape[1]).float().movedim(-1, 1).clamp_min(1e-4)
        rce = -torch.sum(preds * torch.log(targets_one_hot), dim=1).mean()

        return self.alpha * ce + self.beta * rce


class SACLoss(nn.Module):
    """Symmetric Abstaining Classifier loss (ours).

    Combines the universal abstention framework with SCE as the base loss:

        L_SAC = (1 - p_{k+1}) * L_SCE + α * |log((1 - η̃) / (1 - p_{k+1}))|

    The SCE base loss already re-balances the influence of noisy samples via
    the RCE term; the abstention mechanism adds the ability to completely
    disengage from the most egregiously mislabelled samples.

    Uses the same power-law α schedule as GACLoss (see GACLoss docstring).

    Args:
        max_epochs:     Total training epochs (E).
        noise_rate:     Prior noise estimate η̃.
        alpha_final:    Maximum value of α.
        gamma:          Power-law growth factor.
        warmup_epochs:  Epochs before abstention is activated (L).
        sce_alpha:      α weight for the CE component of SCE.
        sce_beta:       β weight for the RCE component of SCE.
    """

    def __init__(
        self,
        max_epochs: int,
        noise_rate: float = 0.0,
        alpha_final: float = 1.0,
        gamma: float = 3.0,
        warmup_epochs: int = 20,
        sce_alpha: float = 0.5,
        sce_beta: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.noise_rate = noise_rate
        self.alpha_final = alpha_final
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.alpha_update_epoch = 0
        self.sce = SCELoss(sce_alpha, sce_beta)
        self.epsilon = 1e-7

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        training: bool = False,
        epoch: int = 0,
    ):
        """
        Args:
            preds:    (B, C+1, H, W) — logits for C classes + 1 abstention channel.
            targets:  (B, H, W)      — integer class labels in [0, C).
            training: If False, returns a plain SCE scalar.
            epoch:    Current epoch index (0-based).

        Returns:
            training=False -> scalar Tensor.
            training=True  -> dict with keys:
                "loss", "SCE loss", "Regularization", "Abstention", "Abstention Rate"
        """
        sce_loss = self.sce(preds[:, :-1], targets)

        if not training:
            return sce_loss

        abstention_rate = (preds.argmax(dim=1) == (preds.shape[1] - 1)).float().mean()
        abstention = F.softmax(preds, dim=1)[:, -1].clamp_max(1 - self.epsilon).mean()
        regularization = 0

        if epoch >= self.warmup_epochs:
            if epoch > self.alpha_update_epoch:
                self.alpha = (
                    self.alpha_final
                    * ((epoch - self.warmup_epochs + 1) / (self.max_epochs - self.warmup_epochs)) ** self.gamma
                )
                self.alpha_update_epoch = epoch

            regularization = self.alpha * abs(torch.log((1 - self.noise_rate) / (1 - abstention)))
            loss = (1 - abstention) * sce_loss + regularization
        else:
            loss = sce_loss

        return {
            "loss": loss,
            "SCE loss": sce_loss,
            "Regularization": regularization,
            "Abstention": abstention,
            "Abstention Rate": abstention_rate,
        }


class DiceLoss(nn.Module):
    """Soft Dice loss (Milletari et al., 3DV 2016).

    Computes a differentiable approximation of the Dice similarity coefficient
    by using softmax probabilities in place of hard predictions:

        Dice_c = 2 * Σ(p_c * y_c) / (Σ p_c + Σ y_c)
        L_Dice = 1 - mean_c(Dice_c)

    The loss is computed per class and then reduced across both classes and the
    batch. Note that the denominator is clamped to avoid division by zero on
    classes that are absent in a given batch.

    Args:
        reduction: How to reduce across the (batch, class) Dice scores.
                   "mean" (default) | "sum" | "none".
                   "none" is used internally by ADSLoss to apply per-class
                   abstention weights before reducing.
    """

    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            preds:   (B, C, H, W) — raw logits.
            targets: (B, H, W)    — integer class labels in [0, C).

        Returns:
            Scalar Tensor if reduction is "mean" or "sum";
            (B, C) Tensor if reduction is "none".
        """
        # Spatial dimensions only; Dice is computed per (batch item, class).
        dims = (-1, -2)
        preds = preds.softmax(dim=1)
        targets = F.one_hot(targets, preds.shape[1]).movedim(-1, 1)

        intersection = (preds * targets).sum(dims)
        # Clamp only the prediction sum to avoid division by zero; the target
        # sum is exact and can legitimately be 0 for absent classes.
        sum_preds = preds.sum(dims).clamp_min(1e-7)
        sum_targets = targets.sum(dims)

        scores = 2 * intersection / (sum_preds + sum_targets)
        loss = 1 - scores  # shape: (B, C)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


class ADSLoss(nn.Module):
    """Abstaining Dice Segmenter loss (ours).

    Extends the universal abstention framework to region-based Dice loss.
    Because Dice is inherently class-wise (it computes a score per class),
    standard pixel-wise abstention is not directly applicable. ADS resolves
    this with two architectural changes:

    1. Class-wise abstention head: a separate branch of the model outputs a
       (B, C) tensor of per-class abstention logits, produced by
       AdaptiveAvgPool2d -> Flatten -> Linear (see models/base.py).

    2. Class-specific noise rates: the regularization term accepts a vector
       η̃_c of per-class noise estimates rather than a single global scalar.
       This is important because class-wise noise variance is typically very
       high in segmentation (e.g., 9.7%–91.1% per class on CaDIS at η=25%).

    The loss is:

        L_ADS = (1 - p_c) * Dice_c + α * |log((1 - η̃_c) / (1 - p_c))|

    applied per class c and then averaged. p_c is the per-class abstention
    probability after sigmoid activation.

    Uses the same power-law α schedule as GACLoss and SACLoss.

    Args:
        max_epochs:     Total training epochs (E).
        noise_rate:     Scalar fallback noise estimate η̃, used when class_noise
                        is not provided.
        class_noise:    Per-class noise vector η̃_c of shape (C,). Preferred over
                        noise_rate when available.
        alpha_final:    Maximum value of α.
        gamma:          Power-law growth factor.
        warmup_epochs:  Epochs before abstention is activated (L).
        window_size:    Output size (s, s) of the AdaptiveAvgPool2d layer in the
                        abstention head. Must match the head defined in the model.
    """

    def __init__(
        self,
        max_epochs: int,
        noise_rate: float = 0.0,
        class_noise: Tensor = None,
        alpha_final: float = 4.0,
        gamma: float = 1.5,
        warmup_epochs: int = 10,
        window_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.noise_rate = noise_rate
        # Use per-class noise vector if available; fall back to the scalar.
        self.class_noise = class_noise if class_noise is not None else noise_rate
        self.alpha_final = alpha_final
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.window_size = window_size
        # reduction="none" keeps the (B, C) shape so we can apply per-class
        # abstention weights before reducing.
        self.dice = DiceLoss(reduction="none")
        self.alpha_update_epoch = 0

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        abstention: Tensor = None,
        epoch: int = 0,
    ):
        """
        Args:
            preds:      (B, C, H, W) — raw segmentation logits (k channels, not k+1).
            targets:    (B, H, W)    — integer class labels in [0, C).
            abstention: (B, C)       — raw per-class abstention logits from the
                                       abstention head. None during inference.
            epoch:      Current epoch index (0-based).

        Returns:
            abstention=None -> scalar Tensor (plain Dice loss, for inference).
            abstention given -> dict with keys:
                "loss"              — total loss (scalar)
                "Dice loss"         — mean Dice loss before abstention weighting
                "Regularization"    — mean regularization term
                "Abstention"        — mean abstention probability across classes
                "Class {i} Abstention" — per-class abstention probability (one key per class)
        """
        # dice_loss shape: (B, C) — unreduced, so we can weight per class.
        dice_loss = self.dice(preds, targets)

        if abstention is None:
            # Inference mode: return plain Dice loss with no abstention.
            return dice_loss.mean()

        if epoch < self.warmup_epochs:
            # Zero out abstention logits during warmup so abstention has no effect.
            # The regularization term is set to the zeroed abstention tensor so it
            # has the right shape for the output dict without extra branching.
            abstention = abstention * 0
            regularization = abstention
            loss = dice_loss
        else:
            if epoch > self.alpha_update_epoch:
                self.alpha = (
                    self.alpha_final
                    * ((epoch - self.warmup_epochs + 1) / (self.max_epochs - self.warmup_epochs)) ** self.gamma
                )
                self.alpha_update_epoch = epoch

            # Sigmoid gives per-class abstention probabilities p_c in (0, 1).
            abstention = F.sigmoid(abstention).clamp_max(1 - 1e-7)

            # Informed regularization per class: α * |log((1 - η̃_c) / (1 - p_c))|
            # class_noise broadcasts correctly whether it is a scalar or a (C,) vector.
            regularization = self.alpha * abs(torch.log((1 - self.class_noise) / (1 - abstention)))

            # Per-class weighted loss: (1 - p_c) * Dice_c + reg_c
            # abstention shape (B, C) broadcasts over dice_loss shape (B, C).
            loss = (1 - abstention) * dice_loss + regularization

        # Average abstention over the batch for logging.
        abstention = abstention.mean(0)  # shape: (C,)
        class_abstention = {f"Class {i} Abstention": p for i, p in enumerate(abstention)}

        return {
            "loss": loss.mean(),
            "Dice loss": dice_loss.mean(),
            "Regularization": regularization.mean(),
            "Abstention": abstention.mean(),
            **class_abstention,
        }
