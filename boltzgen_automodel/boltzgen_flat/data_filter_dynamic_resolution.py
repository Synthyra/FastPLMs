from boltzgen_flat.data_data import Record
from boltzgen_flat.data_filter_dynamic_filter import DynamicFilter


class ResolutionFilter(DynamicFilter):
    """A filter that filters complexes based on their resolution."""

    def __init__(
        self, resolution: float = 9.0, minimum_resolution: float = 0.0
    ) -> None:
        """Initialize the filter.

        Parameters
        ----------
        resolution : float, optional
            The maximum allowed resolution.

        """
        self.resolution = resolution
        self.minimum_resolution = minimum_resolution

    def filter(self, record: Record) -> bool:
        """Filter complexes based on their resolution.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        structure = record.structure
        return self.minimum_resolution <= structure.resolution <= self.resolution
