from collections import defaultdict
from functools import partial

from uni2ts.data.dataset import MultiSampleTimeSeriesDataset
from .lotsa_v1._base import LOTSADatasetBuilder


from pathlib import Path


class FinancialDatasetBuilder(LOTSADatasetBuilder):
    """
    A dataset builder for financial time series data. It extends the `LOTSADatasetBuilder`
    and is configured to work with the `financial_dataset_btc_2015` dataset.

    This builder uses the `MultiSampleTimeSeriesDataset` to load the data, which allows
    for combining multiple time series into a single sample.
    """
    dataset_list = [
        "financial_dataset_btc_2015",
    ]
    """ A list of the financial datasets that this builder can handle. """
    dataset_type_map = defaultdict(lambda: MultiSampleTimeSeriesDataset)
    """ A mapping from dataset names to their corresponding dataset types. """
    dataset_load_func_map = defaultdict(
        lambda: partial(
            MultiSampleTimeSeriesDataset,
            max_ts=128,
            combine_fields=("target",),
        )
    )
    """ A mapping from dataset names to their corresponding loading functions. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_path = Path("./")

    def build_dataset(self, *args, **kwargs):
        """
        This method is not implemented for `FinancialDatasetBuilder`.
        """
        pass
