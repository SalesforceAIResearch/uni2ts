import abc
from typing import Callable, Optional

import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int

from uni2ts.common.core import abstract_class_property

from ._base import PackedQuantileLoss


@abstract_class_property("error_func")
class PackedQuantileLoss(PackedQuantileLoss, abc.ABC):
    error_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = NotImplemented

    def __init__(
        self,
        quantile_levels: tuple[Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ):
        super().__init__()
        self.quantile_levels = quantile_levels

    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len num_quantiles*patch_size"],
        target: Float[torch.Tensor, "*batch seq_len patch_size"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch_size"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len patch_size"]:

        quantile_levels = torch.tensor(self.quantile_levels, device=pred.device).view(
            1, 1, -1, 1
        )
        pred = rearrange(
            pred,
            "... (num_quantiles patch_size) -> ... num_quantiles patch_size",
            num_quantiles=len(self.quantile_levels),
        )
        target = repeat(
            target,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=len(self.quantile_levels),
        )
        errors = self.error_func(pred, target)
        indicator = target > pred

        quantile_loss = torch.where(
            indicator, quantile_levels * errors, (1 - quantile_levels) * errors
        )
        # aggregated by num_quantile axis
        return quantile_loss.mean(dim=-2)


class PackedQuantileMAELoss(PackedQuantileLoss):
    error_func = torch.nn.L1Loss(reduction="none")
