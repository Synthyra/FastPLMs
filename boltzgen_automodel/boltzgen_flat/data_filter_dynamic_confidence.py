from typing import List, Optional, Union

from boltzgen_flat.data_data import Record
from boltzgen_flat.data_filter_dynamic_filter import DynamicFilter


class ConfidenceFilter(DynamicFilter):
    """Filters based on confidence metrics."""

    def __init__(
        self,
        metrics: Union[str, List[str]] = "complex_pde",
        thresholds: Optional[List[float]] = None,
        compare_ops: Union[str, List[str]] = "lesser",
        composition_op: str = "OR",
    ) -> None:
        super().__init__()

        if isinstance(metrics, str):
            metrics = [metrics]

        if isinstance(compare_ops, str):
            compare_ops = [compare_ops]

        self.metrics = metrics
        self.thresholds = thresholds
        self.compare_ops = compare_ops
        self.composition_op = composition_op

        assert len(thresholds) == len(metrics) == len(compare_ops)

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
        if self.thresholds is None:
            return True

        confidence_info = getattr(record, "confidence", None)

        # Non-distillation examples
        if confidence_info is None:
            return True

        filter_results = []
        for idx, metric_name in enumerate(self.metrics):
            metric = getattr(confidence_info, metric_name, None)
            if metric is None:
                filter_results.append(True)
                continue

            compare_op_str = self.compare_ops[idx]
            if compare_op_str == "lesser":
                filter_results.append(metric <= self.thresholds[idx])
            elif compare_op_str == "greater":
                filter_results.append(metric >= self.thresholds[idx])

        assert len(filter_results) == len(self.metrics), "More filters than metrics"

        if self.composition_op == "OR":
            return any(filter_results)

        if self.composition_op == "AND":
            return all(filter_results)

        msg = f"Composition of type {self.composition_op} is not supported yet."
        raise ValueError(msg)
