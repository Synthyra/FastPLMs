from dataclasses import dataclass


@dataclass
class ProteinStructureTemplate:
    sequence: str
    residue_names: list[str]
    atom_names: list[str]
    atom_elements: list[str]
    atom_residue_index: list[int]
    atom_chain_id: list[str]

    @property
    def num_atoms(self) -> int:
        return len(self.atom_names)

    @property
    def num_residues(self) -> int:
        return len(self.residue_names)
