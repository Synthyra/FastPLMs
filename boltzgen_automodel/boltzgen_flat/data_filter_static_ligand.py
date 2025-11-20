import numpy as np

from boltzgen_flat import data_const as const
from boltzgen_flat.data_data import Structure
from boltzgen_flat.data_filter_static_filter import StaticFilter


class ExcludedLigands(StaticFilter):
    """Filter excluded ligands."""

    def filter(self, structure: Structure) -> np.ndarray:
        """Filter excluded ligands.

        Parameters
        ----------
        structure : Structure
            The structure to filter chains from.

        Returns
        -------
        np.ndarray
            The chains to keep, as a boolean mask.

        """
        valid = np.ones(len(structure.chains), dtype=bool)

        for i, chain in enumerate(structure.chains):
            if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]:
                continue

            res_start = chain["res_idx"]
            res_end = res_start + chain["res_num"]
            residues = structure.residues[res_start:res_end]
            if any(res["name"] in const.ligand_exclusion for res in residues):
                valid[i] = 0

        return valid
