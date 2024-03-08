from dataclasses import dataclass
from functools import partial
from typing import Optional

from gluonts.ev.aggregations import Mean
from gluonts.ev.metrics import BaseMetricDefinition, DirectMetric
from gluonts.ev.stats import squared_error


@dataclass
class MedianMSE(BaseMetricDefinition):
    """Mean Squared Error"""

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"MSE[{self.forecast_type}]",
            stat=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )
