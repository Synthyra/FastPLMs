from datetime import datetime
from typing import Literal

from boltzgen_flat.data_data import Record
from boltzgen_flat.data_filter_dynamic_filter import DynamicFilter


class DateFilter(DynamicFilter):
    """A filter that filters complexes based on their date.

    The date can be the deposition, release, or revision date.
    If the date is not available, the previous date is used.

    If no date is available, the complex is rejected.

    """

    def __init__(
        self,
        date: str,
        ref: Literal["deposited", "revised", "released"],
        accept_on_empty: bool = False,
    ) -> None:
        """Initialize the filter.

        Parameters
        ----------
        date : str, optional
            The maximum date of PDB entries to filter
        ref : Literal["deposited", "revised", "released"]
            The reference date to use.

        """
        self.filter_date = datetime.fromisoformat(date)
        self.ref = ref
        self.accept_on_empty = accept_on_empty
        if accept_on_empty:
            print("Warning: Accepting structures if date is empty.")

        if ref not in ["deposited", "revised", "released"]:
            msg = (
                "Invalid reference date. Must be ",
                "deposited, revised, or released",
            )
            raise ValueError(msg)

    def filter(self, record: Record) -> bool:
        """Filter a record based on its date.

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

        if self.ref == "deposited":
            date = structure.deposited
        elif self.ref == "released":
            date = structure.released
            if not date:
                date = structure.deposited
        elif self.ref == "revised":
            date = structure.revised
            if not date:
                date = structure.released if structure.released else structure.deposited

        if date is None or date in {"", "None"}:
            return self.accept_on_empty

        date = datetime.fromisoformat(date).replace(tzinfo=None)
        return date <= self.filter_date
