from boltzgen_flat import data_const as const
from boltzgen_flat.data_data import Record
from boltzgen_flat.data_filter_dynamic_filter import DynamicFilter


class MinProteinResiduesFilter(DynamicFilter):
    """A filter that filters structures based on their size."""

    def __init__(self, min_residues: int = 1, max_residues: int = 100000) -> None:
        """Initialize the filter.

        Parameters
        ----------
        min_chains : int
            The minimum number of chains allowed.
        max_chains : int
            The maximum number of chains allowed.

        """
        self.min_residues = min_residues
        self.max_residues = max_residues

    def filter(self, record: Record) -> bool:
        """Filter structures based on their resolution.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        num_residues = 0
        for chain in record.chains:
            if chain.mol_type == const.chain_type_ids["PROTEIN"]:
                num_residues += chain.num_residues

        return num_residues <= self.max_residues and num_residues >= self.min_residues
