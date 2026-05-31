from dataclasses import dataclass
from typing import List


@dataclass
class ProteinStructureTemplate:
    sequence: str
    residue_names: List[str]
    atom_names: List[str]
    atom_elements: List[str]
    atom_residue_index: List[int]
    atom_chain_id: List[str]

    @property
    def num_atoms(self) -> int:
        return len(self.atom_names)

    @property
    def num_residues(self) -> int:
        return len(self.residue_names)
