import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import numpy as np
import math
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module, ModuleList, Linear, Sequential

from math import pi, sqrt, exp
from scipy.stats import beta, norm
from einops.layers.torch import Rearrange
from einops import rearrange
from functools import partial, partialmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PretrainedConfig


### Constants
# Type alias for device
Device = Union[str, torch.device]


chain_types = [
    "PROTEIN",
    "DNA",
    "RNA",
    "NONPOLYMER",
]
chain_type_ids = {chain: i for i, chain in enumerate(chain_types)}

out_types = [
    "dna_protein",
    "rna_protein",
    "ligand_protein",
    "dna_ligand",
    "rna_ligand",
    "intra_ligand",
    "intra_dna",
    "intra_rna",
    "intra_protein",
    "protein_protein",
    "design_protein",
    "design_ligand",
    "design_dna",
    "design_rna",
    "intra_design",
    "design_design",
    "modified",
]

out_types_weights_af3 = {
    "dna_protein": 10.0,
    "rna_protein": 10.0,
    "ligand_protein": 10.0,
    "dna_ligand": 5.0,
    "rna_ligand": 5.0,
    "intra_ligand": 20.0,
    "intra_dna": 4.0,
    "intra_rna": 16.0,
    "intra_protein": 20.0,
    "protein_protein": 20.0,
    "modified": 0.0,
}

out_types_weights = {
    "dna_protein": 5.0,
    "rna_protein": 5.0,
    "ligand_protein": 20.0,
    "dna_ligand": 2.0,
    "rna_ligand": 2.0,
    "intra_ligand": 20.0,
    "intra_dna": 2.0,
    "intra_rna": 8.0,
    "intra_protein": 20.0,
    "protein_protein": 20.0,
    "design_protein": 20.0,
    "design_ligand": 20.0,
    "design_dna": 5.0,
    "design_rna": 5.0,
    "intra_design": 20.0,
    "design_design": 20.0,
    "modified": 0.0,
}


out_single_types = ["protein", "ligand", "dna", "rna"]

clash_types = [
    "dna_protein",
    "rna_protein",
    "ligand_protein",
    "protein_protein",
    "dna_ligand",
    "rna_ligand",
    "ligand_ligand",
    "rna_dna",
    "dna_dna",
    "rna_rna",
]

chain_types_to_clash_type = {
    frozenset(("PROTEIN", "DNA")): "dna_protein",
    frozenset(("PROTEIN", "RNA")): "rna_protein",
    frozenset(("PROTEIN", "NONPOLYMER")): "ligand_protein",
    frozenset(("PROTEIN",)): "protein_protein",
    frozenset(("NONPOLYMER", "DNA")): "dna_ligand",
    frozenset(("NONPOLYMER", "RNA")): "rna_ligand",
    frozenset(("NONPOLYMER",)): "ligand_ligand",
    frozenset(("DNA", "RNA")): "rna_dna",
    frozenset(("DNA",)): "dna_dna",
    frozenset(("RNA",)): "rna_rna",
}

chain_type_to_out_single_type = {
    "PROTEIN": "protein",
    "DNA": "dna",
    "RNA": "rna",
    "NONPOLYMER": "ligand",
}

canonical_tokens = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

non_canonincal_tokens = [
    "<pad>",
    "-",
    "UNK",  # unknown protein token
    "A",
    "G",
    "C",
    "U",
    "N",  # unknown rna token
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # unknown dna token
]

canonicals_offset = 2

tokens = (
    non_canonincal_tokens[:canonicals_offset]
    + canonical_tokens
    + non_canonincal_tokens[canonicals_offset:]
)

token_ids = {token: i for i, token in enumerate(tokens)}
num_tokens = len(tokens)
unk_token = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
unk_token_ids = {m: token_ids[t] for m, t in unk_token.items()}
mask_token = {"PROTEIN": "UNK", "DNA": "UNK", "RNA": "UNK", "NONPOLYMER": "UNK"}
mask_token_ids = {m: token_ids[t] for m, t in mask_token.items()}

prot_letter_to_token = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
    "J": "UNK",
    "B": "UNK",
    "Z": "UNK",
    "O": "UNK",
    "U": "UNK",
    "-": "-",
}

prot_token_to_letter = {v: k for k, v in prot_letter_to_token.items()}
prot_token_to_letter["UNK"] = "X"

rna_letter_to_token = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "N": "N",
}
rna_token_to_letter = {v: k for k, v in rna_letter_to_token.items()}

dna_letter_to_token = {
    "A": "DA",
    "G": "DG",
    "C": "DC",
    "T": "DT",
    "N": "DN",
}
dna_token_to_letter = {v: k for k, v in dna_letter_to_token.items()}

binding_types = [
    "UNSPECIFIED",
    "BINDING",
    "NOT_BINDING",
]
binding_type_ids = {binding_type: i for i, binding_type in enumerate(binding_types)}

ss_types = [
    "UNSPECIFIED",
    "LOOP",
    "HELIX",
    "SHEET",
]
ss_type_ids = {ss_type: i for i, ss_type in enumerate(ss_types)}


element_to_atomic_num = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "SC": 21,
    "TI": 22,
    "V": 23,
    "CR": 24,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "GA": 31,
    "GE": 32,
    "AS": 33,
    "SE": 34,
    "BR": 35,
    "KR": 36,
    "RB": 37,
    "SR": 38,
    "Y": 39,
    "ZR": 40,
    "NB": 41,
    "MO": 42,
    "TC": 43,
    "RU": 44,
    "RH": 45,
    "PD": 46,
    "AG": 47,
    "CD": 48,
    "IN": 49,
    "SN": 50,
    "SB": 51,
    "TE": 52,
    "I": 53,
    "XE": 54,
    "CS": 55,
    "BA": 56,
    "LA": 57,
    "CE": 58,
    "PR": 59,
    "ND": 60,
    "PM": 61,
    "SM": 62,
    "EU": 63,
    "GD": 64,
    "TB": 65,
    "DY": 66,
    "HO": 67,
    "ER": 68,
    "TM": 69,
    "YB": 70,
    "LU": 71,
    "HF": 72,
    "TA": 73,
    "W": 74,
    "RE": 75,
    "OS": 76,
    "IR": 77,
    "PT": 78,
    "AU": 79,
    "HG": 80,
    "TL": 81,
    "PB": 82,
    "BI": 83,
    "PO": 84,
    "AT": 85,
    "RN": 86,
    "FR": 87,
    "RA": 88,
    "AC": 89,
    "TH": 90,
    "PA": 91,
    "U": 92,
    "NP": 93,
    "PU": 94,
    "AM": 95,
    "CM": 96,
    "BK": 97,
    "CF": 98,
    "ES": 99,
    "FM": 100,
    "MD": 101,
    "NO": 102,
    "LR": 103,
    "RF": 104,
    "DB": 105,
    "SG": 106,
    "BH": 107,
    "HS": 108,
    "MT": 109,
    "DS": 110,
    "RG": 111,
    "CN": 112,
    "NH": 113,
    "FL": 114,
    "MC": 115,
    "LV": 116,
    "TS": 117,
    "OG": 118,
}
atomic_num_to_element = {v: k for k, v in element_to_atomic_num.items()}

num_elements = 128

mask_element = "FL"
mask_element_id = element_to_atomic_num[mask_element]


fake_element = "LV"
fake_element_id = element_to_atomic_num[fake_element]

chirality_types = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_SQUAREPLANAR",
    "CHI_OCTAHEDRAL",
    "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_OTHER",
]
chirality_type_ids = {chirality: i for i, chirality in enumerate(chirality_types)}
unk_chirality_type = "CHI_OTHER"

hybridization_map = [
    "S",
    "SP",
    "SP2",
    "SP2D",
    "SP3",
    "SP3D",
    "SP3D2",
    "OTHER",
    "UNSPECIFIED",
]
hybridization_type_ids = {hybrid: i for i, hybrid in enumerate(hybridization_map)}
unk_hybridization_type = "UNSPECIFIED"

atom_types = [  # Note: CB and O order flipped to be consistent with atom14 order
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

# fmt: off
ref_atoms = {
    "PAD": [],
    "UNK": ["N", "CA", "C", "O", "CB"],
    "-": [],
    "GLY": ["N", "CA", "C", "O"],  # 0
    "ALA": ["N", "CA", "C", "O", "CB"], # 1
    "CYS": ["N", "CA", "C", "O", "CB", "SG"], # 2
    "SER": ["N", "CA", "C", "O", "CB", "OG"], # 2
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],# 3
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],# 3
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],# 3
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],# 4
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],# 4
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],# 4
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],# 4
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],# 4
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],# 5
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],# 5
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],# 5
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],# 6
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],# 7
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],# 7
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"], # 8
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  # 10 noqa: E501
    "A": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],  # noqa: E501
    "G": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],  # noqa: E501
    "C": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],  # noqa: E501
    "U": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],  # noqa: E501
    "N": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],  # noqa: E501
    "DA": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],  # noqa: E501
    "DG": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],  # noqa: E501
    "DC": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],  # noqa: E501
    "DT": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],  # noqa: E501
    "DN": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
}

protein_backbone_atom_names = ["N", "CA", "C", "O"]
nucleic_backbone_atom_names = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]

protein_backbone_atom_index = {name: i for i, name in enumerate(protein_backbone_atom_names)}
nucleic_backbone_atom_index = {name: i for i, name in enumerate(nucleic_backbone_atom_names)}


# number of side chain atoms per residue
# 0: GLY
# 1: ALA
# 2: CYS SER
# 3: PRO THR VAL
# 4: ASN ASP ILE LEU MET
# 5: GLN GLU LYS
# 6: HIS
# 7: ARG PHE
# 8: TYR
# 9:
# 10: TRP

# Amino acid code based on proximity to backbone atoms
fake_atom_placements = {
    "UNK": [".", ".", ".", ".", ".", "N", "N", "N", "N", "N", "N", "N", "N", "N"], # 0
    "GLY": [".", ".", ".", ".", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], # 0
    "ALA": [".", ".", ".", ".", ".", "O", "O", "O", "O", "O", "O", "O", "O", "O"], # 1
    "CYS": [".", ".", ".", ".", ".", ".", "O", "O", "O", "O", "O", "O", "O", "O"], # 2
    "SER": [".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "N", "N", "N", "N"], # 2
    "PRO": [".", ".", ".", ".", ".", ".", ".", "O", "O", "O", "O", "O", "O", "O"], # 3
    "THR": [".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "O", "O", "O", "O"], # 3
    "VAL": [".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "N", "N", "N"], # 3
    "ILE": [".", ".", ".", ".", ".", ".", ".", ".", "O", "O", "O", "O", "O", "O"], # 4
    "ASN": [".", ".", ".", ".", ".", ".", ".", ".", "N", "O", "O", "O", "O", "O"], # 4
    "ASP": [".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "O", "O", "O", "O"], # 4
    "LEU": [".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "O", "O"], # 4
    "MET": [".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "N", "N"], # 4
    "GLN": [".", ".", ".", ".", ".", ".", ".", ".", ".", "O", "O", "O", "O", "O"], # 5
    "GLU": [".", ".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "O", "O", "O"], # 5
    "LYS": [".", ".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "N"], # 5
    "HIS": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "O", "O", "O", "O"], # 6
    "PHE": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "O", "O", "O"], # 7
    "ARG": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "N"], # 7
    "TYR": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "O", "O"], # 8
    "TRP": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."], # 10
}

# {
#  'UNK': [10, 0, 0, 0],
#  'GLY': [0, 0, 0, 10],
#  'ALA': [0, 0, 0, 9],
#  'CYS': [0, 0, 0, 8],
# ...
# }
token_to_placement_count = {ttype:[placement.count(atom_name) for atom_name in ref_atoms["GLY"]] for ttype, placement in fake_atom_placements.items()}
placement_count_to_token = {tuple(v):k for k,v in token_to_placement_count.items()}


fake_atom_placements_N_C = {
    "UNK": [".", ".", ".", ".", ".", "N", "N", "N", "N", "N", "N", "N", "N", "N"], # 0
    "GLY": [".", ".", ".", ".", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"], # 0
    "ALA": [".", ".", ".", ".", ".", "C", "C", "C", "C", "C", "C", "C", "C", "C"], # 1
    "CYS": [".", ".", ".", ".", ".", ".", "C", "C", "C", "C", "C", "C", "C", "C"], # 2
    "SER": [".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "N", "N", "N", "N"], # 2
    "PRO": [".", ".", ".", ".", ".", ".", ".", "C", "C", "C", "C", "C", "C", "C"], # 3
    "THR": [".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "C", "C", "C", "C"], # 3
    "VAL": [".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "N", "N", "N"], # 3
    "ILE": [".", ".", ".", ".", ".", ".", ".", ".", "C", "C", "C", "C", "C", "C"], # 4
    "ASN": [".", ".", ".", ".", ".", ".", ".", ".", "N", "C", "C", "C", "C", "C"], # 4
    "ASP": [".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "C", "C", "C", "C"], # 4
    "LEU": [".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "C", "C"], # 4
    "MET": [".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "N", "N"], # 4
    "GLN": [".", ".", ".", ".", ".", ".", ".", ".", ".", "C", "C", "C", "C", "C"], # 5
    "GLU": [".", ".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "C", "C", "C"], # 5
    "LYS": [".", ".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "N", "N", "N"], # 5
    "HIS": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "C", "C", "C", "C"], # 6
    "PHE": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "C", "C", "C"], # 7
    "ARG": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "N", "N", "N"], # 7
    "TYR": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "C", "C"], # 8
    "TRP": [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."], # 10
}



ref_symmetries = {
    "PAD": [],
    "ALA": [],
    "ARG": [],
    "ASN": [],
    "ASP": [[(6, 7), (7, 6)]],
    "CYS": [],
    "GLN": [],
    "GLU": [[(7, 8), (8, 7)]],
    "GLY": [],
    "HIS": [],
    "ILE": [],
    "LEU": [],
    "LYS": [],
    "MET": [],
    "PHE": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "PRO": [],
    "SER": [],
    "THR": [],
    "TRP": [],
    "TYR": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "VAL": [],
    "A": [[(1, 2), (2, 1)]],
    "G": [[(1, 2), (2, 1)]],
    "C": [[(1, 2), (2, 1)]],
    "U": [[(1, 2), (2, 1)]],
    #"N": [[(1, 2), (2, 1)]],
    "DA": [[(1, 2), (2, 1)]],
    "DG": [[(1, 2), (2, 1)]],
    "DC": [[(1, 2), (2, 1)]],
    "DT": [[(1, 2), (2, 1)]],
    #"DN": [[(1, 2), (2, 1)]]
}


res_to_center_atom = {
    "UNK": "CA",
    "ALA": "CA",
    "ARG": "CA",
    "ASN": "CA",
    "ASP": "CA",
    "CYS": "CA",
    "GLN": "CA",
    "GLU": "CA",
    "GLY": "CA",
    "HIS": "CA",
    "ILE": "CA",
    "LEU": "CA",
    "LYS": "CA",
    "MET": "CA",
    "PHE": "CA",
    "PRO": "CA",
    "SER": "CA",
    "THR": "CA",
    "TRP": "CA",
    "TYR": "CA",
    "VAL": "CA",
    "A": "C1'",
    "G": "C1'",
    "C": "C1'",
    "U": "C1'",
    "N": "C1'",
    "DA": "C1'",
    "DG": "C1'",
    "DC": "C1'",
    "DT": "C1'",
    "DN": "C1'"
}

res_to_disto_atom = {
    "UNK": "CB",
    "ALA": "CB",
    "ARG": "CB",
    "ASN": "CB",
    "ASP": "CB",
    "CYS": "CB",
    "GLN": "CB",
    "GLU": "CB",
    "GLY": "CA",
    "HIS": "CB",
    "ILE": "CB",
    "LEU": "CB",
    "LYS": "CB",
    "MET": "CB",
    "PHE": "CB",
    "PRO": "CB",
    "SER": "CB",
    "THR": "CB",
    "TRP": "CB",
    "TYR": "CB",
    "VAL": "CB",
    "A": "C4",
    "G": "C4",
    "C": "C2",
    "U": "C2",
    "N": "C1'",
    "DA": "C4",
    "DG": "C4",
    "DC": "C2",
    "DT": "C2",
    "DN": "C1'"
}

res_to_center_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_center_atom.items()
}

res_to_disto_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_disto_atom.items()
}

# fmt: on

res_type_weight = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    2.0,
    1.0,
    1.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    10.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]

atom_interface_cutoff = 5.0
interface_cutoff = 15.0

bond_types = [
    "OTHER",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
    "COVALENT",
]
bond_type_ids = {bond: i for i, bond in enumerate(bond_types)}
unk_bond_type = "OTHER"


pocket_contact_info = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET": 2,
    "BINDER": 3,
}

contact_conditioning_info = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET>BINDER": 2,
    "BINDER>POCKET": 3,
    "CONTACT": 4,
}

activity_types = {
    "Ki": 0,
    "Kd": 0,
    "kd": 0,
    "ki_microm": 0,
    "IC50": 1,
    "AC50": 1,
    "EC50": 1,
    "ac50_um": 1,
    "avgec50": 1,
    "ec50": 1,
    "ec50_microm": 1,
    "ec50_um": 1,
    "ic50": 1,
    "ic50 (um)": 1,
    "ic50(um)": 1,
    "ic50_microm": 1,
    "mean ic50": 1,
    "DEL": 1,
}


chunk_size_threshold = 512

# Methods
method_types_ids = {
    "MD": 0,
    "X-RAY DIFFRACTION": 1,
    "ELECTRON MICROSCOPY": 2,
    "SOLUTION NMR": 3,
    "SOLID-STATE NMR": 4,
    "NEUTRON DIFFRACTION": 4,
    "ELECTRON CRYSTALLOGRAPHY": 4,
    "FIBER DIFFRACTION": 4,
    "POWDER DIFFRACTION": 4,
    "INFRARED SPECTROSCOPY": 4,
    "FLUORESCENCE TRANSFER": 4,
    "EPR": 4,
    "THEORETICAL MODEL": 4,
    "SOLUTION SCATTERING": 4,
    "OTHER": 4,
    "AFDB": 5,
    "BOLTZ-1": 6,
    "FUTURE1": 7,  # Placeholder for future supervision sources
    "FUTURE2": 8,
    "FUTURE3": 9,
    "FUTURE4": 10,
    "FUTURE5": 11,
}
method_types_ids = {k.lower(): v for k, v in method_types_ids.items()}
num_method_types = len(set(method_types_ids.values()))

# Temperature
temperature_bins = [(265, 280), (280, 295), (295, 310)]
temperature_bins_ids = {temp: i for i, temp in enumerate(temperature_bins)}
temperature_bins_ids["other"] = len(temperature_bins)
num_temp_bins = len(temperature_bins_ids)


# pH
ph_bins = [(0, 6), (6, 8), (8, 14)]
ph_bins_ids = {ph: i for i, ph in enumerate(ph_bins)}
ph_bins_ids["other"] = len(ph_bins)
num_ph_bins = len(ph_bins_ids)

vdw_radii = [
    1.2,
    1.4,
    2.2,
    1.9,
    1.8,
    1.7,
    1.6,
    1.55,
    1.5,
    1.54,
    2.4,
    2.2,
    2.1,
    2.1,
    1.95,
    1.8,
    1.8,
    1.88,
    2.8,
    2.4,
    2.3,
    2.15,
    2.05,
    2.05,
    2.05,
    2.05,
    2.0,
    2.0,
    2.0,
    2.1,
    2.1,
    2.1,
    2.05,
    1.9,
    1.9,
    2.02,
    2.9,
    2.55,
    2.4,
    2.3,
    2.15,
    2.1,
    2.05,
    2.05,
    2.0,
    2.05,
    2.1,
    2.2,
    2.2,
    2.25,
    2.2,
    2.1,
    2.1,
    2.16,
    3.0,
    2.7,
    2.5,
    2.48,
    2.47,
    2.45,
    2.43,
    2.42,
    2.4,
    2.38,
    2.37,
    2.35,
    2.33,
    2.32,
    2.3,
    2.28,
    2.27,
    2.25,
    2.2,
    2.1,
    2.05,
    2.0,
    2.0,
    2.05,
    2.1,
    2.05,
    2.2,
    2.3,
    2.3,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.4,
    2.0,
    2.3,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
]

protein_letters_3to1_extended = {
    "A5N": "N",
    "A8E": "V",
    "A9D": "S",
    "AA3": "A",
    "AA4": "A",
    "AAR": "R",
    "ABA": "A",
    "ACL": "R",
    "AEA": "C",
    "AEI": "D",
    "AFA": "N",
    "AGM": "R",
    "AGQ": "Y",
    "AGT": "C",
    "AHB": "N",
    "AHL": "R",
    "AHO": "A",
    "AHP": "A",
    "AIB": "A",
    "AKL": "D",
    "AKZ": "D",
    "ALA": "A",
    "ALC": "A",
    "ALM": "A",
    "ALN": "A",
    "ALO": "T",
    "ALS": "A",
    "ALT": "A",
    "ALV": "A",
    "ALY": "K",
    "AME": "M",
    "AN6": "L",
    "AN8": "A",
    "API": "K",
    "APK": "K",
    "AR2": "R",
    "AR4": "E",
    "AR7": "R",
    "ARG": "R",
    "ARM": "R",
    "ARO": "R",
    "AS7": "N",
    "ASA": "D",
    "ASB": "D",
    "ASI": "D",
    "ASK": "D",
    "ASL": "D",
    "ASN": "N",
    "ASP": "D",
    "ASQ": "D",
    "AYA": "A",
    "AZH": "A",
    "AZK": "K",
    "AZS": "S",
    "AZY": "Y",
    "AVJ": "H",
    "A30": "Y",
    "A3U": "F",
    "ECC": "Q",
    "ECX": "C",
    "EFC": "C",
    "EHP": "F",
    "ELY": "K",
    "EME": "E",
    "EPM": "M",
    "EPQ": "Q",
    "ESB": "Y",
    "ESC": "M",
    "EXY": "L",
    "EXA": "K",
    "E0Y": "P",
    "E9V": "H",
    "E9M": "W",
    "EJA": "C",
    "EUP": "T",
    "EZY": "G",
    "E9C": "Y",
    "EW6": "S",
    "EXL": "W",
    "I2M": "I",
    "I4G": "G",
    "I58": "K",
    "IAM": "A",
    "IAR": "R",
    "ICY": "C",
    "IEL": "K",
    "IGL": "G",
    "IIL": "I",
    "ILE": "I",
    "ILG": "E",
    "ILM": "I",
    "ILX": "I",
    "ILY": "K",
    "IML": "I",
    "IOR": "R",
    "IPG": "G",
    "IT1": "K",
    "IYR": "Y",
    "IZO": "M",
    "IC0": "G",
    "M0H": "C",
    "M2L": "K",
    "M2S": "M",
    "M30": "G",
    "M3L": "K",
    "M3R": "K",
    "MA ": "A",
    "MAA": "A",
    "MAI": "R",
    "MBQ": "Y",
    "MC1": "S",
    "MCL": "K",
    "MCS": "C",
    "MD3": "C",
    "MD5": "C",
    "MD6": "G",
    "MDF": "Y",
    "ME0": "M",
    "MEA": "F",
    "MEG": "E",
    "MEN": "N",
    "MEQ": "Q",
    "MET": "M",
    "MEU": "G",
    "MFN": "E",
    "MGG": "R",
    "MGN": "Q",
    "MGY": "G",
    "MH1": "H",
    "MH6": "S",
    "MHL": "L",
    "MHO": "M",
    "MHS": "H",
    "MHU": "F",
    "MIR": "S",
    "MIS": "S",
    "MK8": "L",
    "ML3": "K",
    "MLE": "L",
    "MLL": "L",
    "MLY": "K",
    "MLZ": "K",
    "MME": "M",
    "MMO": "R",
    "MNL": "L",
    "MNV": "V",
    "MP8": "P",
    "MPQ": "G",
    "MSA": "G",
    "MSE": "M",
    "MSL": "M",
    "MSO": "M",
    "MT2": "M",
    "MTY": "Y",
    "MVA": "V",
    "MYK": "K",
    "MYN": "R",
    "QCS": "C",
    "QIL": "I",
    "QMM": "Q",
    "QPA": "C",
    "QPH": "F",
    "Q3P": "K",
    "QVA": "C",
    "QX7": "A",
    "Q2E": "W",
    "Q75": "M",
    "Q78": "F",
    "QM8": "L",
    "QMB": "A",
    "QNQ": "C",
    "QNT": "C",
    "QNW": "C",
    "QO2": "C",
    "QO5": "C",
    "QO8": "C",
    "QQ8": "Q",
    "U2X": "Y",
    "U3X": "F",
    "UF0": "S",
    "UGY": "G",
    "UM1": "A",
    "UM2": "A",
    "UMA": "A",
    "UQK": "A",
    "UX8": "W",
    "UXQ": "F",
    "YCM": "C",
    "YOF": "Y",
    "YPR": "P",
    "YPZ": "Y",
    "YTH": "T",
    "Y1V": "L",
    "Y57": "K",
    "YHA": "K",
    "200": "F",
    "23F": "F",
    "23P": "A",
    "26B": "T",
    "28X": "T",
    "2AG": "A",
    "2CO": "C",
    "2FM": "M",
    "2GX": "F",
    "2HF": "H",
    "2JG": "S",
    "2KK": "K",
    "2KP": "K",
    "2LT": "Y",
    "2LU": "L",
    "2ML": "L",
    "2MR": "R",
    "2MT": "P",
    "2OR": "R",
    "2P0": "P",
    "2QZ": "T",
    "2R3": "Y",
    "2RA": "A",
    "2RX": "S",
    "2SO": "H",
    "2TY": "Y",
    "2VA": "V",
    "2XA": "C",
    "2ZC": "S",
    "6CL": "K",
    "6CW": "W",
    "6GL": "A",
    "6HN": "K",
    "60F": "C",
    "66D": "I",
    "6CV": "A",
    "6M6": "C",
    "6V1": "C",
    "6WK": "C",
    "6Y9": "P",
    "6DN": "K",
    "DA2": "R",
    "DAB": "A",
    "DAH": "F",
    "DBS": "S",
    "DBU": "T",
    "DBY": "Y",
    "DBZ": "A",
    "DC2": "C",
    "DDE": "H",
    "DDZ": "A",
    "DI7": "Y",
    "DHA": "S",
    "DHN": "V",
    "DIR": "R",
    "DLS": "K",
    "DM0": "K",
    "DMH": "N",
    "DMK": "D",
    "DNL": "K",
    "DNP": "A",
    "DNS": "K",
    "DNW": "A",
    "DOH": "D",
    "DON": "L",
    "DP1": "R",
    "DPL": "P",
    "DPP": "A",
    "DPQ": "Y",
    "DYS": "C",
    "D2T": "D",
    "DYA": "D",
    "DJD": "F",
    "DYJ": "P",
    "DV9": "E",
    "H14": "F",
    "H1D": "M",
    "H5M": "P",
    "HAC": "A",
    "HAR": "R",
    "HBN": "H",
    "HCM": "C",
    "HGY": "G",
    "HHI": "H",
    "HIA": "H",
    "HIC": "H",
    "HIP": "H",
    "HIQ": "H",
    "HIS": "H",
    "HL2": "L",
    "HLU": "L",
    "HMR": "R",
    "HNC": "C",
    "HOX": "F",
    "HPC": "F",
    "HPE": "F",
    "HPH": "F",
    "HPQ": "F",
    "HQA": "A",
    "HR7": "R",
    "HRG": "R",
    "HRP": "W",
    "HS8": "H",
    "HS9": "H",
    "HSE": "S",
    "HSK": "H",
    "HSL": "S",
    "HSO": "H",
    "HT7": "W",
    "HTI": "C",
    "HTR": "W",
    "HV5": "A",
    "HVA": "V",
    "HY3": "P",
    "HYI": "M",
    "HYP": "P",
    "HZP": "P",
    "HIX": "A",
    "HSV": "H",
    "HLY": "K",
    "HOO": "H",
    "H7V": "A",
    "L5P": "K",
    "LRK": "K",
    "L3O": "L",
    "LA2": "K",
    "LAA": "D",
    "LAL": "A",
    "LBY": "K",
    "LCK": "K",
    "LCX": "K",
    "LDH": "K",
    "LE1": "V",
    "LED": "L",
    "LEF": "L",
    "LEH": "L",
    "LEM": "L",
    "LEN": "L",
    "LET": "K",
    "LEU": "L",
    "LEX": "L",
    "LGY": "K",
    "LLO": "K",
    "LLP": "K",
    "LLY": "K",
    "LLZ": "K",
    "LME": "E",
    "LMF": "K",
    "LMQ": "Q",
    "LNE": "L",
    "LNM": "L",
    "LP6": "K",
    "LPD": "P",
    "LPG": "G",
    "LPS": "S",
    "LSO": "K",
    "LTR": "W",
    "LVG": "G",
    "LVN": "V",
    "LWY": "P",
    "LYF": "K",
    "LYK": "K",
    "LYM": "K",
    "LYN": "K",
    "LYO": "K",
    "LYP": "K",
    "LYR": "K",
    "LYS": "K",
    "LYU": "K",
    "LYX": "K",
    "LYZ": "K",
    "LAY": "L",
    "LWI": "F",
    "LBZ": "K",
    "P1L": "C",
    "P2Q": "Y",
    "P2Y": "P",
    "P3Q": "Y",
    "PAQ": "Y",
    "PAS": "D",
    "PAT": "W",
    "PBB": "C",
    "PBF": "F",
    "PCA": "Q",
    "PCC": "P",
    "PCS": "F",
    "PE1": "K",
    "PEC": "C",
    "PF5": "F",
    "PFF": "F",
    "PG1": "S",
    "PGY": "G",
    "PHA": "F",
    "PHD": "D",
    "PHE": "F",
    "PHI": "F",
    "PHL": "F",
    "PHM": "F",
    "PKR": "P",
    "PLJ": "P",
    "PM3": "F",
    "POM": "P",
    "PPN": "F",
    "PR3": "C",
    "PR4": "P",
    "PR7": "P",
    "PR9": "P",
    "PRJ": "P",
    "PRK": "K",
    "PRO": "P",
    "PRS": "P",
    "PRV": "G",
    "PSA": "F",
    "PSH": "H",
    "PTH": "Y",
    "PTM": "Y",
    "PTR": "Y",
    "PVH": "H",
    "PXU": "P",
    "PYA": "A",
    "PYH": "K",
    "PYX": "C",
    "PH6": "P",
    "P9S": "C",
    "P5U": "S",
    "POK": "R",
    "T0I": "Y",
    "T11": "F",
    "TAV": "D",
    "TBG": "V",
    "TBM": "T",
    "TCQ": "Y",
    "TCR": "W",
    "TEF": "F",
    "TFQ": "F",
    "TH5": "T",
    "TH6": "T",
    "THC": "T",
    "THR": "T",
    "THZ": "R",
    "TIH": "A",
    "TIS": "S",
    "TLY": "K",
    "TMB": "T",
    "TMD": "T",
    "TNB": "C",
    "TNR": "S",
    "TNY": "T",
    "TOQ": "W",
    "TOX": "W",
    "TPJ": "P",
    "TPK": "P",
    "TPL": "W",
    "TPO": "T",
    "TPQ": "Y",
    "TQI": "W",
    "TQQ": "W",
    "TQZ": "C",
    "TRF": "W",
    "TRG": "K",
    "TRN": "W",
    "TRO": "W",
    "TRP": "W",
    "TRQ": "W",
    "TRW": "W",
    "TRX": "W",
    "TRY": "W",
    "TS9": "I",
    "TSY": "C",
    "TTQ": "W",
    "TTS": "Y",
    "TXY": "Y",
    "TY1": "Y",
    "TY2": "Y",
    "TY3": "Y",
    "TY5": "Y",
    "TY8": "Y",
    "TY9": "Y",
    "TYB": "Y",
    "TYC": "Y",
    "TYE": "Y",
    "TYI": "Y",
    "TYJ": "Y",
    "TYN": "Y",
    "TYO": "Y",
    "TYQ": "Y",
    "TYR": "Y",
    "TYS": "Y",
    "TYT": "Y",
    "TYW": "Y",
    "TYY": "Y",
    "T8L": "T",
    "T9E": "T",
    "TNQ": "W",
    "TSQ": "F",
    "TGH": "W",
    "X2W": "E",
    "XCN": "C",
    "XPR": "P",
    "XSN": "N",
    "XW1": "A",
    "XX1": "K",
    "XYC": "A",
    "XA6": "F",
    "11Q": "P",
    "11W": "E",
    "12L": "P",
    "12X": "P",
    "12Y": "P",
    "143": "C",
    "1AC": "A",
    "1L1": "A",
    "1OP": "Y",
    "1PA": "F",
    "1PI": "A",
    "1TQ": "W",
    "1TY": "Y",
    "1X6": "S",
    "56A": "H",
    "5AB": "A",
    "5CS": "C",
    "5CW": "W",
    "5HP": "E",
    "5OH": "A",
    "5PG": "G",
    "51T": "Y",
    "54C": "W",
    "5CR": "F",
    "5CT": "K",
    "5FQ": "A",
    "5GM": "I",
    "5JP": "S",
    "5T3": "K",
    "5MW": "K",
    "5OW": "K",
    "5R5": "S",
    "5VV": "N",
    "5XU": "A",
    "55I": "F",
    "999": "D",
    "9DN": "N",
    "9NE": "E",
    "9NF": "F",
    "9NR": "R",
    "9NV": "V",
    "9E7": "K",
    "9KP": "K",
    "9WV": "A",
    "9TR": "K",
    "9TU": "K",
    "9TX": "K",
    "9U0": "K",
    "9IJ": "F",
    "B1F": "F",
    "B27": "T",
    "B2A": "A",
    "B2F": "F",
    "B2I": "I",
    "B2V": "V",
    "B3A": "A",
    "B3D": "D",
    "B3E": "E",
    "B3K": "K",
    "B3U": "H",
    "B3X": "N",
    "B3Y": "Y",
    "BB6": "C",
    "BB7": "C",
    "BB8": "F",
    "BB9": "C",
    "BBC": "C",
    "BCS": "C",
    "BCX": "C",
    "BFD": "D",
    "BG1": "S",
    "BH2": "D",
    "BHD": "D",
    "BIF": "F",
    "BIU": "I",
    "BL2": "L",
    "BLE": "L",
    "BLY": "K",
    "BMT": "T",
    "BNN": "F",
    "BOR": "R",
    "BP5": "A",
    "BPE": "C",
    "BSE": "S",
    "BTA": "L",
    "BTC": "C",
    "BTK": "K",
    "BTR": "W",
    "BUC": "C",
    "BUG": "V",
    "BYR": "Y",
    "BWV": "R",
    "BWB": "S",
    "BXT": "S",
    "F2F": "F",
    "F2Y": "Y",
    "FAK": "K",
    "FB5": "A",
    "FB6": "A",
    "FC0": "F",
    "FCL": "F",
    "FDL": "K",
    "FFM": "C",
    "FGL": "G",
    "FGP": "S",
    "FH7": "K",
    "FHL": "K",
    "FHO": "K",
    "FIO": "R",
    "FLA": "A",
    "FLE": "L",
    "FLT": "Y",
    "FME": "M",
    "FOE": "C",
    "FP9": "P",
    "FPK": "P",
    "FT6": "W",
    "FTR": "W",
    "FTY": "Y",
    "FVA": "V",
    "FZN": "K",
    "FY3": "Y",
    "F7W": "W",
    "FY2": "Y",
    "FQA": "K",
    "F7Q": "Y",
    "FF9": "K",
    "FL6": "D",
    "JJJ": "C",
    "JJK": "C",
    "JJL": "C",
    "JLP": "K",
    "J3D": "C",
    "J9Y": "R",
    "J8W": "S",
    "JKH": "P",
    "N10": "S",
    "N7P": "P",
    "NA8": "A",
    "NAL": "A",
    "NAM": "A",
    "NBQ": "Y",
    "NC1": "S",
    "NCB": "A",
    "NEM": "H",
    "NEP": "H",
    "NFA": "F",
    "NIY": "Y",
    "NLB": "L",
    "NLE": "L",
    "NLN": "L",
    "NLO": "L",
    "NLP": "L",
    "NLQ": "Q",
    "NLY": "G",
    "NMC": "G",
    "NMM": "R",
    "NNH": "R",
    "NOT": "L",
    "NPH": "C",
    "NPI": "A",
    "NTR": "Y",
    "NTY": "Y",
    "NVA": "V",
    "NWD": "A",
    "NYB": "C",
    "NYS": "C",
    "NZH": "H",
    "N80": "P",
    "NZC": "T",
    "NLW": "L",
    "N0A": "F",
    "N9P": "A",
    "N65": "K",
    "R1A": "C",
    "R4K": "W",
    "RE0": "W",
    "RE3": "W",
    "RGL": "R",
    "RGP": "E",
    "RT0": "P",
    "RVX": "S",
    "RZ4": "S",
    "RPI": "R",
    "RVJ": "A",
    "VAD": "V",
    "VAF": "V",
    "VAH": "V",
    "VAI": "V",
    "VAL": "V",
    "VB1": "K",
    "VH0": "P",
    "VR0": "R",
    "V44": "C",
    "V61": "F",
    "VPV": "K",
    "V5N": "H",
    "V7T": "K",
    "Z01": "A",
    "Z3E": "T",
    "Z70": "H",
    "ZBZ": "C",
    "ZCL": "F",
    "ZU0": "T",
    "ZYJ": "P",
    "ZYK": "P",
    "ZZD": "C",
    "ZZJ": "A",
    "ZIQ": "W",
    "ZPO": "P",
    "ZDJ": "Y",
    "ZT1": "K",
    "30V": "C",
    "31Q": "C",
    "33S": "F",
    "33W": "A",
    "34E": "V",
    "3AH": "H",
    "3BY": "P",
    "3CF": "F",
    "3CT": "Y",
    "3GA": "A",
    "3GL": "E",
    "3MD": "D",
    "3MY": "Y",
    "3NF": "Y",
    "3O3": "E",
    "3PX": "P",
    "3QN": "K",
    "3TT": "P",
    "3XH": "G",
    "3YM": "Y",
    "3WS": "A",
    "3WX": "P",
    "3X9": "C",
    "3ZH": "H",
    "7JA": "I",
    "73C": "S",
    "73N": "R",
    "73O": "Y",
    "73P": "K",
    "74P": "K",
    "7N8": "F",
    "7O5": "A",
    "7XC": "F",
    "7ID": "D",
    "7OZ": "A",
    "C1S": "C",
    "C1T": "C",
    "C1X": "K",
    "C22": "A",
    "C3Y": "C",
    "C4R": "C",
    "C5C": "C",
    "C6C": "C",
    "CAF": "C",
    "CAS": "C",
    "CAY": "C",
    "CCS": "C",
    "CEA": "C",
    "CGA": "E",
    "CGU": "E",
    "CGV": "C",
    "CHP": "G",
    "CIR": "R",
    "CLE": "L",
    "CLG": "K",
    "CLH": "K",
    "CME": "C",
    "CMH": "C",
    "CML": "C",
    "CMT": "C",
    "CR5": "G",
    "CS0": "C",
    "CS1": "C",
    "CS3": "C",
    "CS4": "C",
    "CSA": "C",
    "CSB": "C",
    "CSD": "C",
    "CSE": "C",
    "CSJ": "C",
    "CSO": "C",
    "CSP": "C",
    "CSR": "C",
    "CSS": "C",
    "CSU": "C",
    "CSW": "C",
    "CSX": "C",
    "CSZ": "C",
    "CTE": "W",
    "CTH": "T",
    "CWD": "A",
    "CWR": "S",
    "CXM": "M",
    "CY0": "C",
    "CY1": "C",
    "CY3": "C",
    "CY4": "C",
    "CYA": "C",
    "CYD": "C",
    "CYF": "C",
    "CYG": "C",
    "CYJ": "K",
    "CYM": "C",
    "CYQ": "C",
    "CYR": "C",
    "CYS": "C",
    "CYW": "C",
    "CZ2": "C",
    "CZZ": "C",
    "CG6": "C",
    "C1J": "R",
    "C4G": "R",
    "C67": "R",
    "C6D": "R",
    "CE7": "N",
    "CZS": "A",
    "G01": "E",
    "G8M": "E",
    "GAU": "E",
    "GEE": "G",
    "GFT": "S",
    "GHC": "E",
    "GHG": "Q",
    "GHW": "E",
    "GL3": "G",
    "GLH": "Q",
    "GLJ": "E",
    "GLK": "E",
    "GLN": "Q",
    "GLQ": "E",
    "GLU": "E",
    "GLY": "G",
    "GLZ": "G",
    "GMA": "E",
    "GME": "E",
    "GNC": "Q",
    "GPL": "K",
    "GSC": "G",
    "GSU": "E",
    "GT9": "C",
    "GVL": "S",
    "G3M": "R",
    "G5G": "L",
    "G1X": "Y",
    "G8X": "P",
    "K1R": "C",
    "KBE": "K",
    "KCX": "K",
    "KFP": "K",
    "KGC": "K",
    "KNB": "A",
    "KOR": "M",
    "KPI": "K",
    "KPY": "K",
    "KST": "K",
    "KYN": "W",
    "KYQ": "K",
    "KCR": "K",
    "KPF": "K",
    "K5L": "S",
    "KEO": "K",
    "KHB": "K",
    "KKD": "D",
    "K5H": "C",
    "K7K": "S",
    "OAR": "R",
    "OAS": "S",
    "OBS": "K",
    "OCS": "C",
    "OCY": "C",
    "OHI": "H",
    "OHS": "D",
    "OLD": "H",
    "OLT": "T",
    "OLZ": "S",
    "OMH": "S",
    "OMT": "M",
    "OMX": "Y",
    "OMY": "Y",
    "ONH": "A",
    "ORN": "A",
    "ORQ": "R",
    "OSE": "S",
    "OTH": "T",
    "OXX": "D",
    "OYL": "H",
    "O7A": "T",
    "O7D": "W",
    "O7G": "V",
    "O2E": "S",
    "O6H": "W",
    "OZW": "F",
    "S12": "S",
    "S1H": "S",
    "S2C": "C",
    "S2P": "A",
    "SAC": "S",
    "SAH": "C",
    "SAR": "G",
    "SBG": "S",
    "SBL": "S",
    "SCH": "C",
    "SCS": "C",
    "SCY": "C",
    "SD4": "N",
    "SDB": "S",
    "SDP": "S",
    "SEB": "S",
    "SEE": "S",
    "SEG": "A",
    "SEL": "S",
    "SEM": "S",
    "SEN": "S",
    "SEP": "S",
    "SER": "S",
    "SET": "S",
    "SGB": "S",
    "SHC": "C",
    "SHP": "G",
    "SHR": "K",
    "SIB": "C",
    "SLL": "K",
    "SLZ": "K",
    "SMC": "C",
    "SME": "M",
    "SMF": "F",
    "SNC": "C",
    "SNN": "N",
    "SOY": "S",
    "SRZ": "S",
    "STY": "Y",
    "SUN": "S",
    "SVA": "S",
    "SVV": "S",
    "SVW": "S",
    "SVX": "S",
    "SVY": "S",
    "SVZ": "S",
    "SXE": "S",
    "SKH": "K",
    "SNM": "S",
    "SNK": "H",
    "SWW": "S",
    "WFP": "F",
    "WLU": "L",
    "WPA": "F",
    "WRP": "W",
    "WVL": "V",
    "02K": "A",
    "02L": "N",
    "02O": "A",
    "02Y": "A",
    "033": "V",
    "037": "P",
    "03Y": "C",
    "04U": "P",
    "04V": "P",
    "05N": "P",
    "07O": "C",
    "0A0": "D",
    "0A1": "Y",
    "0A2": "K",
    "0A8": "C",
    "0A9": "F",
    "0AA": "V",
    "0AB": "V",
    "0AC": "G",
    "0AF": "W",
    "0AG": "L",
    "0AH": "S",
    "0AK": "D",
    "0AR": "R",
    "0BN": "F",
    "0CS": "A",
    "0E5": "T",
    "0EA": "Y",
    "0FL": "A",
    "0LF": "P",
    "0NC": "A",
    "0PR": "Y",
    "0QL": "C",
    "0TD": "D",
    "0UO": "W",
    "0WZ": "Y",
    "0X9": "R",
    "0Y8": "P",
    "4AF": "F",
    "4AR": "R",
    "4AW": "W",
    "4BF": "F",
    "4CF": "F",
    "4CY": "M",
    "4DP": "W",
    "4FB": "P",
    "4FW": "W",
    "4HL": "Y",
    "4HT": "W",
    "4IN": "W",
    "4MM": "M",
    "4PH": "F",
    "4U7": "A",
    "41H": "F",
    "41Q": "N",
    "42Y": "S",
    "432": "S",
    "45F": "P",
    "4AK": "K",
    "4D4": "R",
    "4GJ": "C",
    "4KY": "P",
    "4L0": "P",
    "4LZ": "Y",
    "4N7": "P",
    "4N8": "P",
    "4N9": "P",
    "4OG": "W",
    "4OU": "F",
    "4OV": "S",
    "4OZ": "S",
    "4PQ": "W",
    "4SJ": "F",
    "4WQ": "A",
    "4HH": "S",
    "4HJ": "S",
    "4J4": "C",
    "4J5": "R",
    "4II": "F",
    "4VI": "R",
    "823": "N",
    "8SP": "S",
    "8AY": "A",
}

# Nucleic Acids
nucleic_letters_3to1_extended = {
    "A  ": "A",
    "A23": "A",
    "A2L": "A",
    "A2M": "A",
    "A34": "A",
    "A35": "A",
    "A38": "A",
    "A39": "A",
    "A3A": "A",
    "A3P": "A",
    "A40": "A",
    "A43": "A",
    "A44": "A",
    "A47": "A",
    "A5L": "A",
    "A5M": "C",
    "A5O": "A",
    "A6A": "A",
    "A6C": "C",
    "A6G": "G",
    "A6U": "U",
    "A7E": "A",
    "A9Z": "A",
    "ABR": "A",
    "ABS": "A",
    "AD2": "A",
    "ADI": "A",
    "ADP": "A",
    "AET": "A",
    "AF2": "A",
    "AFG": "G",
    "AMD": "A",
    "AMO": "A",
    "AP7": "A",
    "AS ": "A",
    "ATD": "T",
    "ATL": "T",
    "ATM": "T",
    "AVC": "A",
    "AI5": "C",
    "E  ": "A",
    "E1X": "A",
    "EDA": "A",
    "EFG": "G",
    "EHG": "G",
    "EIT": "T",
    "EXC": "C",
    "E3C": "C",
    "E6G": "G",
    "E7G": "G",
    "EQ4": "G",
    "EAN": "T",
    "I5C": "C",
    "IC ": "C",
    "IG ": "G",
    "IGU": "G",
    "IMC": "C",
    "IMP": "G",
    "IU ": "U",
    "I4U": "U",
    "IOO": "G",
    "M1G": "G",
    "M2G": "G",
    "M4C": "C",
    "M5M": "C",
    "MA6": "A",
    "MA7": "A",
    "MAD": "A",
    "MCY": "C",
    "ME6": "C",
    "MEP": "U",
    "MG1": "G",
    "MGQ": "A",
    "MGT": "G",
    "MGV": "G",
    "MIA": "A",
    "MMT": "T",
    "MNU": "U",
    "MRG": "G",
    "MTR": "T",
    "MTU": "A",
    "MFO": "G",
    "M7A": "A",
    "MHG": "G",
    "MMX": "C",
    "QUO": "G",
    "QCK": "T",
    "QSQ": "A",
    "U  ": "U",
    "U25": "U",
    "U2L": "U",
    "U2P": "U",
    "U31": "U",
    "U34": "U",
    "U36": "U",
    "U37": "U",
    "U8U": "U",
    "UAR": "U",
    "UBB": "U",
    "UBD": "U",
    "UD5": "U",
    "UPV": "U",
    "UR3": "U",
    "URD": "U",
    "US3": "T",
    "US5": "U",
    "UZR": "U",
    "UMO": "U",
    "U23": "U",
    "U48": "C",
    "U7B": "C",
    "Y  ": "A",
    "YCO": "C",
    "YG ": "G",
    "YYG": "G",
    "23G": "G",
    "26A": "A",
    "2AR": "A",
    "2AT": "T",
    "2AU": "U",
    "2BT": "T",
    "2BU": "A",
    "2DA": "A",
    "2DT": "T",
    "2EG": "G",
    "2GT": "T",
    "2JV": "G",
    "2MA": "A",
    "2MG": "G",
    "2MU": "U",
    "2NT": "T",
    "2OM": "U",
    "2OT": "T",
    "2PR": "G",
    "2SG": "G",
    "2ST": "T",
    "63G": "G",
    "63H": "G",
    "64T": "T",
    "68Z": "G",
    "6CT": "T",
    "6HA": "A",
    "6HB": "A",
    "6HC": "C",
    "6HG": "G",
    "6HT": "T",
    "6IA": "A",
    "6MA": "A",
    "6MC": "A",
    "6MP": "A",
    "6MT": "A",
    "6MZ": "A",
    "6OG": "G",
    "6PO": "G",
    "6FK": "G",
    "6NW": "A",
    "6OO": "C",
    "D00": "C",
    "D3T": "T",
    "D4M": "T",
    "DA ": "A",
    "DC ": "C",
    "DCG": "G",
    "DCT": "C",
    "DDG": "G",
    "DFC": "C",
    "DFG": "G",
    "DG ": "G",
    "DG8": "G",
    "DGI": "G",
    "DGP": "G",
    "DHU": "U",
    "DNR": "C",
    "DOC": "C",
    "DPB": "T",
    "DRT": "T",
    "DT ": "T",
    "DZM": "A",
    "D4B": "C",
    "H2U": "U",
    "HN0": "G",
    "HN1": "G",
    "LC ": "C",
    "LCA": "A",
    "LCG": "G",
    "LG ": "G",
    "LGP": "G",
    "LHU": "U",
    "LSH": "T",
    "LST": "T",
    "LDG": "G",
    "L3X": "A",
    "LHH": "C",
    "LV2": "C",
    "L1J": "G",
    "P  ": "G",
    "P2T": "T",
    "P5P": "A",
    "PG7": "G",
    "PGN": "G",
    "PGP": "G",
    "PMT": "C",
    "PPU": "A",
    "PPW": "G",
    "PR5": "A",
    "PRN": "A",
    "PST": "T",
    "PSU": "U",
    "PU ": "A",
    "PVX": "C",
    "PYO": "U",
    "PZG": "G",
    "P4U": "U",
    "P7G": "G",
    "T  ": "T",
    "T2S": "T",
    "T31": "U",
    "T32": "T",
    "T36": "T",
    "T37": "T",
    "T38": "T",
    "T39": "T",
    "T3P": "T",
    "T41": "T",
    "T48": "T",
    "T49": "T",
    "T4S": "T",
    "T5S": "T",
    "T64": "T",
    "T6A": "A",
    "TA3": "T",
    "TAF": "T",
    "TBN": "A",
    "TC1": "C",
    "TCP": "T",
    "TCY": "A",
    "TDY": "T",
    "TED": "T",
    "TFE": "T",
    "TFF": "T",
    "TFO": "A",
    "TFT": "T",
    "TGP": "G",
    "TCJ": "C",
    "TLC": "T",
    "TP1": "T",
    "TPC": "C",
    "TPG": "G",
    "TSP": "T",
    "TTD": "T",
    "TTM": "T",
    "TXD": "A",
    "TXP": "A",
    "TC ": "C",
    "TG ": "G",
    "T0N": "G",
    "T0Q": "G",
    "X  ": "G",
    "XAD": "A",
    "XAL": "A",
    "XCL": "C",
    "XCR": "C",
    "XCT": "C",
    "XCY": "C",
    "XGL": "G",
    "XGR": "G",
    "XGU": "G",
    "XPB": "G",
    "XTF": "T",
    "XTH": "T",
    "XTL": "T",
    "XTR": "T",
    "XTS": "G",
    "XUA": "A",
    "XUG": "G",
    "102": "G",
    "10C": "C",
    "125": "U",
    "126": "U",
    "127": "U",
    "12A": "A",
    "16B": "C",
    "18M": "G",
    "1AP": "A",
    "1CC": "C",
    "1FC": "C",
    "1MA": "A",
    "1MG": "G",
    "1RN": "U",
    "1SC": "C",
    "5AA": "A",
    "5AT": "T",
    "5BU": "U",
    "5CG": "G",
    "5CM": "C",
    "5FA": "A",
    "5FC": "C",
    "5FU": "U",
    "5HC": "C",
    "5HM": "C",
    "5HT": "T",
    "5IC": "C",
    "5IT": "T",
    "5MC": "C",
    "5MU": "U",
    "5NC": "C",
    "5PC": "C",
    "5PY": "T",
    "9QV": "U",
    "94O": "T",
    "9SI": "A",
    "9SY": "A",
    "B7C": "C",
    "BGM": "G",
    "BOE": "T",
    "B8H": "U",
    "B8K": "G",
    "B8Q": "C",
    "B8T": "C",
    "B8W": "G",
    "B9B": "G",
    "B9H": "C",
    "BGH": "G",
    "F3H": "T",
    "F3N": "A",
    "F4H": "T",
    "FA2": "A",
    "FDG": "G",
    "FHU": "U",
    "FMG": "G",
    "FNU": "U",
    "FOX": "G",
    "F2T": "U",
    "F74": "G",
    "F4Q": "G",
    "F7H": "C",
    "F7K": "G",
    "JDT": "T",
    "JMH": "C",
    "J0X": "C",
    "N5M": "C",
    "N6G": "G",
    "N79": "A",
    "NCU": "C",
    "NMS": "T",
    "NMT": "T",
    "NTT": "T",
    "N7X": "C",
    "R  ": "A",
    "RBD": "A",
    "RDG": "G",
    "RIA": "A",
    "RMP": "A",
    "RPC": "C",
    "RSP": "C",
    "RSQ": "C",
    "RT ": "T",
    "RUS": "U",
    "RFJ": "G",
    "V3L": "A",
    "VC7": "G",
    "Z  ": "C",
    "ZAD": "A",
    "ZBC": "C",
    "ZBU": "U",
    "ZCY": "C",
    "ZGU": "G",
    "31H": "A",
    "31M": "A",
    "3AU": "U",
    "3DA": "A",
    "3ME": "U",
    "3MU": "U",
    "3TD": "U",
    "70U": "U",
    "7AT": "A",
    "7DA": "A",
    "7GU": "G",
    "7MG": "G",
    "7BG": "G",
    "73W": "C",
    "75B": "U",
    "7OK": "C",
    "7S3": "G",
    "7SN": "G",
    "C  ": "C",
    "C25": "C",
    "C2L": "C",
    "C2S": "C",
    "C31": "C",
    "C32": "C",
    "C34": "C",
    "C36": "C",
    "C37": "C",
    "C38": "C",
    "C42": "C",
    "C43": "C",
    "C45": "C",
    "C46": "C",
    "C49": "C",
    "C4S": "C",
    "C5L": "C",
    "C6G": "G",
    "CAR": "C",
    "CB2": "C",
    "CBR": "C",
    "CBV": "C",
    "CCC": "C",
    "CDW": "C",
    "CFL": "C",
    "CFZ": "C",
    "CG1": "G",
    "CH ": "C",
    "CMR": "C",
    "CNU": "U",
    "CP1": "C",
    "CSF": "C",
    "CSL": "C",
    "CTG": "T",
    "CX2": "C",
    "C7S": "C",
    "C7R": "C",
    "G  ": "G",
    "G1G": "G",
    "G25": "G",
    "G2L": "G",
    "G2S": "G",
    "G31": "G",
    "G32": "G",
    "G33": "G",
    "G36": "G",
    "G38": "G",
    "G42": "G",
    "G46": "G",
    "G47": "G",
    "G48": "G",
    "G49": "G",
    "G7M": "G",
    "GAO": "G",
    "GCK": "C",
    "GDO": "G",
    "GDP": "G",
    "GDR": "G",
    "GF2": "G",
    "GFL": "G",
    "GH3": "G",
    "GMS": "G",
    "GN7": "G",
    "GNG": "G",
    "GOM": "G",
    "GRB": "G",
    "GS ": "G",
    "GSR": "G",
    "GSS": "G",
    "GTP": "G",
    "GX1": "G",
    "KAG": "G",
    "KAK": "G",
    "O2G": "G",
    "OGX": "G",
    "OMC": "C",
    "OMG": "G",
    "OMU": "U",
    "ONE": "U",
    "O2Z": "A",
    "OKN": "C",
    "OKQ": "C",
    "S2M": "T",
    "S4A": "A",
    "S4C": "C",
    "S4G": "G",
    "S4U": "U",
    "S6G": "G",
    "SC ": "C",
    "SDE": "A",
    "SDG": "G",
    "SDH": "G",
    "SMP": "A",
    "SMT": "T",
    "SPT": "T",
    "SRA": "A",
    "SSU": "U",
    "SUR": "U",
    "00A": "A",
    "0AD": "G",
    "0AM": "A",
    "0AP": "C",
    "0AV": "A",
    "0R8": "C",
    "0SP": "A",
    "0UH": "G",
    "47C": "C",
    "4OC": "C",
    "4PC": "C",
    "4PD": "C",
    "4PE": "C",
    "4SC": "C",
    "4SU": "U",
    "45A": "A",
    "4U3": "C",
    "8AG": "G",
    "8AN": "A",
    "8BA": "A",
    "8FG": "G",
    "8MG": "G",
    "8OG": "G",
    "8PY": "G",
    "8AA": "G",
    "85Y": "U",
    "8OS": "G",
}


ligand_exclusion = {
    "144",
    "15P",
    "1PE",
    "2F2",
    "2JC",
    "3HR",
    "3SY",
    "7N5",
    "7PE",
    "9JE",
    "AAE",
    "ABA",
    "ACE",
    "ACN",
    "ACT",
    "ACY",
    "AZI",
    "BAM",
    "BCN",
    "BCT",
    "BDN",
    "BEN",
    "BME",
    "BO3",
    "BTB",
    "BTC",
    "BU1",
    "C8E",
    "CAD",
    "CAQ",
    "CBM",
    "CCN",
    "CIT",
    "CL",
    "CLR",
    "CM",
    "CMO",
    "CO3",
    "CPT",
    "CXS",
    "D10",
    "DEP",
    "DIO",
    "DMS",
    "DN",
    "DOD",
    "DOX",
    "EDO",
    "EEE",
    "EGL",
    "EOH",
    "EOX",
    "EPE",
    "ETF",
    "FCY",
    "FJO",
    "FLC",
    "FMT",
    "FW5",
    "GOL",
    "GSH",
    "GTT",
    "GYF",
    "HED",
    "IHP",
    "IHS",
    "IMD",
    "IOD",
    "IPA",
    "IPH",
    "LDA",
    "MB3",
    "MEG",
    "MES",
    "MLA",
    "MLI",
    "MOH",
    "MPD",
    "MRD",
    "MSE",
    "MYR",
    "N",
    "NA",
    "NH2",
    "NH4",
    "NHE",
    "NO3",
    "O4B",
    "OHE",
    "OLA",
    "OLC",
    "OMB",
    "OME",
    "OXA",
    "P6G",
    "PE3",
    "PE4",
    "PEG",
    "PEO",
    "PEP",
    "PG0",
    "PG4",
    "PGE",
    "PGR",
    "PLM",
    "PO4",
    "POL",
    "POP",
    "PVO",
    "SAR",
    "SCN",
    "SEO",
    "SEP",
    "SIN",
    "SO4",
    "SPD",
    "SPM",
    "SR",
    "STE",
    "STO",
    "STU",
    "TAR",
    "TBU",
    "TME",
    "TPO",
    "TRS",
    "UNK",
    "UNL",
    "UNX",
    "UPL",
    "URE",
}


ambiguous_atoms = {
    "LV": "LV",
    "FL": "FL",
    "CA": {
        "*": "C",
        "OEX": "CA",
        "OEC": "CA",
        "543": "CA",
        "OC6": "CA",
        "OC1": "CA",
        "OC7": "CA",
        "OEY": "CA",
        "OC4": "CA",
        "OC3": "CA",
        "ICA": "CA",
        "CA": "CA",
        "OC2": "CA",
        "OC5": "CA",
    },
    "CD": {"*": "C", "CD": "CD", "CD3": "CD", "CD5": "CD", "CD1": "CD"},
    "BR": "BR",
    "CL": {
        "*": "CL",
        "C8P": "C",
        "L3T": "C",
        "TLC": "C",
        "TZ0": "C",
        "471": "C",
        "NLK": "C",
        "PGM": "C",
        "PNE": "C",
        "RCY": "C",
        "11F": "C",
        "PII": "C",
        "C1Q": "C",
        "4MD": "C",
        "R5A": "C",
        "KW2": "C",
        "I7M": "C",
        "R48": "C",
        "FC3": "C",
        "55V": "C",
        "KPF": "C",
        "SPZ": "C",
        "0TT": "C",
        "R9A": "C",
        "5NA": "C",
        "C55": "C",
        "NIX": "C",
        "5PM": "C",
        "PP8": "C",
        "544": "C",
        "812": "C",
        "NPM": "C",
        "KU8": "C",
        "A1AMM": "C",
        "4S0": "C",
        "AQC": "C",
        "2JK": "C",
        "WJR": "C",
        "A1AAW": "C",
        "85E": "C",
        "MB0": "C",
        "ZAB": "C",
        "85K": "C",
        "GBP": "C",
        "A1H80": "C",
        "A1AFR": "C",
        "L9M": "C",
        "MYK": "C",
        "MB9": "C",
        "38R": "C",
        "EKB": "C",
        "NKF": "C",
        "UMQ": "C",
        "T4K": "C",
        "3PT": "C",
        "A1A7S": "C",
        "1Q9": "C",
        "11R": "C",
        "D2V": "C",
        "SM8": "C",
        "IFC": "C",
        "DB5": "C",
        "L2T": "C",
        "GNB": "C",
        "PP7": "C",
        "072": "C",
        "P88": "C",
        "DRL": "C",
        "C9W": "C",
        "NTP": "C",
        "4HJ": "C",
        "7NA": "C",
        "LPC": "C",
        "T8W": "C",
        "63R": "C",
        "570": "C",
        "R4A": "C",
        "3BG": "C",
        "4RB": "C",
        "GSO": "C",
        "BQ6": "C",
        "R4P": "C",
        "5CP": "C",
        "TTR": "C",
        "6UZ": "C",
        "SPJ": "C",
        "0SA": "C",
        "ZL1": "C",
        "BYG": "C",
        "F0E": "C",
        "PC0": "C",
        "B2Q": "C",
        "KV6": "C",
        "NTO": "C",
        "CLG": "C",
        "R7U": "C",
        "SMQ": "C",
        "GM2": "C",
        "Z7P": "C",
        "NXF": "C",
        "C6Q": "C",
        "A1G": "C",
        "433": "C",
        "L9N": "C",
        "7OX": "C",
        "A1H84": "C",
        "97L": "C",
        "HDV": "C",
        "LUO": "C",
        "R6A": "C",
        "1PC": "C",
        "4PT": "C",
        "SBZ": "C",
        "EAB": "C",
        "FL4": "C",
        "OPS": "C",
        "C2X": "C",
        "SLL": "C",
        "BFC": "C",
        "GIP": "C",
        "7CP": "C",
        "CLH": "C",
        "34E": "C",
        "5NE": "C",
        "PBF": "C",
        "ABD": "C",
        "ABC": "C",
        "LPF": "C",
        "TIZ": "C",
        "4HH": "C",
        "AFC": "C",
        "WQH": "C",
        "9JL": "C",
        "CS3": "C",
        "NL0": "C",
        "KPY": "C",
        "DNA": "C",
        "B3C": "C",
        "TKL": "C",
        "KVS": "C",
        "HO6": "C",
        "NLH": "C",
        "1PB": "C",
        "CYF": "C",
        "G4M": "C",
        "R5B": "C",
        "N4S": "C",
        "N11": "C",
        "C8F": "C",
        "PIJ": "C",
        "WIN": "C",
        "NT1": "C",
        "WJW": "C",
        "HF7": "C",
        "TY1": "C",
        "VM1": "C",
    },
    "OS": {"*": "O", "DWC": "OS", "OHX": "OS", "OS": "OS", "8WV": "OS", "OS4": "OS"},
    "PB": {"*": "P", "ZN9": "PB", "ZN7": "PB", "PBM": "PB", "PB": "PB", "CSB": "PB"},
    "CE": {"*": "C", "CE": "CE"},
    "FE": {"*": "FE", "TFR": "F", "PF5": "F", "IFC": "F", "F5C": "F"},
    "NA": {"*": "N", "CGO": "NA", "R2K": "NA", "LVQ": "NA", "NA": "NA"},
    "ND": {"*": "N", "ND": "ND"},
    "CF": {"*": "C", "CF": "CF"},
    "RU": "RU",
    "BRAF": "BR",
    "EU": "EU",
    "CLAA": "CL",
    "CLBQ": "CL",
    "CM": {"*": "C", "ZCM": "CM"},
    "SN": {"*": "SN", "TAP": "S", "SND": "S", "TAD": "S", "XPT": "S"},
    "AG": "AG",
    "CLN": "CL",
    "CLM": "CL",
    "CLA": {"*": "CL", "PII": "C", "TDL": "C", "D0J": "C", "GM2": "C", "PIJ": "C"},
    "CLB": {
        "*": "CL",
        "TD5": "C",
        "PII": "C",
        "TDL": "C",
        "GM2": "C",
        "TD7": "C",
        "TD6": "C",
        "PIJ": "C",
    },
    "CR": {
        "*": "C",
        "BW9": "CR",
        "CQ4": "CR",
        "AC9": "CR",
        "TIL": "CR",
        "J7U": "CR",
        "CR": "CR",
    },
    "CLAY": "CL",
    "CLBC": "CL",
    "PD": {
        "*": "P",
        "F6Q": "PD",
        "SVP": "PD",
        "SXC": "PD",
        "U5U": "PD",
        "PD": "PD",
        "PLL": "PD",
    },
    "CO": {
        "*": "C",
        "J1S": "CO",
        "OCN": "CO",
        "OL3": "CO",
        "OL4": "CO",
        "B12": "CO",
        "XCO": "CO",
        "UFU": "CO",
        "CON": "CO",
        "OL5": "CO",
        "B13": "CO",
        "7KI": "CO",
        "PL1": "CO",
        "OCO": "CO",
        "J1R": "CO",
        "COH": "CO",
        "SIR": "CO",
        "6KI": "CO",
        "NCO": "CO",
        "9CO": "CO",
        "PC3": "CO",
        "BWU": "CO",
        "B1Z": "CO",
        "J83": "CO",
        "CO": "CO",
        "COY": "CO",
        "CNC": "CO",
        "3CO": "CO",
        "OCL": "CO",
        "R5Q": "CO",
        "X5Z": "CO",
        "CBY": "CO",
        "OLS": "CO",
        "F0X": "CO",
        "I2A": "CO",
        "OCM": "CO",
    },
    "CU": {
        "*": "C",
        "8ZR": "CU",
        "K7E": "CU",
        "CU3": "CU",
        "SI9": "CU",
        "35N": "CU",
        "C2O": "CU",
        "SI7": "CU",
        "B15": "CU",
        "SI0": "CU",
        "CUP": "CU",
        "SQ1": "CU",
        "CUK": "CU",
        "CUL": "CU",
        "SI8": "CU",
        "IC4": "CU",
        "CUM": "CU",
        "MM2": "CU",
        "B30": "CU",
        "S32": "CU",
        "V79": "CU",
        "IMF": "CU",
        "CUN": "CU",
        "MM1": "CU",
        "MP1": "CU",
        "IME": "CU",
        "B17": "CU",
        "C2C": "CU",
        "1CU": "CU",
        "CU6": "CU",
        "C1O": "CU",
        "CU1": "CU",
        "B22": "CU",
        "CUS": "CU",
        "RUQ": "CU",
        "CUF": "CU",
        "CUA": "CU",
        "CU": "CU",
        "CUO": "CU",
        "0TE": "CU",
        "SI4": "CU",
    },
    "CS": {"*": "C", "CS": "CS"},
    "CLQ": "CL",
    "CLR": "CL",
    "CLU": "CL",
    "TE": "TE",
    "NI": {
        "*": "N",
        "USN": "NI",
        "NFO": "NI",
        "NI2": "NI",
        "NFS": "NI",
        "NFR": "NI",
        "82N": "NI",
        "R5N": "NI",
        "NFU": "NI",
        "A1ICD": "NI",
        "NI3": "NI",
        "M43": "NI",
        "MM5": "NI",
        "BF8": "NI",
        "TCN": "NI",
        "NIK": "NI",
        "CUV": "NI",
        "MM6": "NI",
        "J52": "NI",
        "NI": "NI",
        "SNF": "NI",
        "XCC": "NI",
        "F0L": "NI",
        "UWE": "NI",
        "NFC": "NI",
        "3NI": "NI",
        "HNI": "NI",
        "F43": "NI",
        "RQM": "NI",
        "NFE": "NI",
        "NFB": "NI",
        "B51": "NI",
        "NI1": "NI",
        "WCC": "NI",
        "NUF": "NI",
    },
    "SB": {"*": "S", "UJI": "SB", "SB": "SB", "118": "SB", "SBO": "SB", "3CG": "SB"},
    "MO": "MO",
    "SEG": "SE",
    "CLL": "CL",
    "CLAH": "CL",
    "CLC": {
        "*": "CL",
        "TD5": "C",
        "PII": "C",
        "TDL": "C",
        "GM2": "C",
        "TD7": "C",
        "TD6": "C",
        "PIJ": "C",
    },
    "CLD": {"*": "CL", "PII": "C", "GM2": "C", "PIJ": "C"},
    "CLAD": "CL",
    "CLAE": "CL",
    "LA": "LA",
    "RH": "RH",
    "BRAC": "BR",
    "BRAD": "BR",
    "CLBN": "CL",
    "CLAC": "CL",
    "BRAB": "BR",
    "BRAE": "BR",
    "MG": "MG",
    "IR": "IR",
    "SE": {
        "*": "SE",
        "HII": "S",
        "NT2": "S",
        "R2P": "S",
        "S2P": "S",
        "0IU": "S",
        "QMB": "S",
        "81S": "S",
        "0QB": "S",
        "UB4": "S",
        "OHS": "S",
        "Q78": "S",
        "0Y2": "S",
        "B3M": "S",
        "NT1": "S",
        "81R": "S",
    },
    "BRAG": "BR",
    "CLF": {"*": "CL", "PII": "C", "GM2": "C", "PIJ": "C"},
    "CLE": {"*": "CL", "PII": "C", "GM2": "C", "PIJ": "C"},
    "BRAX": "BR",
    "CLK": "CL",
    "ZN": "ZN",
    "AS": "AS",
    "AU": "AU",
    "PT": "PT",
    "CLAS": "CL",
    "MN": "MN",
    "CLBE": "CL",
    "CLBF": "CL",
    "CLAF": "CL",
    "NA'": {"*": "N", "CGO": "NA"},
    "BRAH": "BR",
    "BRAI": "BR",
    "BRA": "BR",
    "BRB": "BR",
    "BRAV": "BR",
    "HG": {
        "*": "HG",
        "BBA": "H",
        "MID": "H",
        "APM": "H",
        "4QQ": "H",
        "0ZG": "H",
        "APH": "H",
    },
    "AR": "AR",
    "D": "H",
    "CLAN": "CL",
    "SI": "SI",
    "CLS": "CL",
    "ZR": "ZR",
    "CLAR": {"*": "CL", "ZM4": "C"},
    "HO": "HO",
    "CLI": {"*": "CL", "GM2": "C"},
    "CLH": {"*": "CL", "GM2": "C"},
    "CLAP": "CL",
    "CLBL": "CL",
    "CLBM": "CL",
    "PR": {"*": "PR", "UF0": "P", "252": "P"},
    "IN": "IN",
    "CLJ": "CL",
    "BRU": "BR",
    "SC": {"*": "S", "SFL": "SC"},
    "CLG": {"*": "CL", "GM2": "C"},
    "BRAT": "BR",
    "BRAR": "BR",
    "CLAG": "CL",
    "CLAB": "CL",
    "CLV": "CL",
    "TI": "TI",
    "CLAX": "CL",
    "CLAJ": "CL",
    "CL'": {"*": "CL", "BNR": "C", "25A": "C", "BDA": "C"},
    "CLAW": "CL",
    "BRF": "BR",
    "BRE": "BR",
    "RE": "RE",
    "GD": "GD",
    "SM": {"*": "S", "SM": "SM"},
    "CLBH": "CL",
    "CLBI": "CL",
    "CLAI": "CL",
    "CLY": "CL",
    "CLZ": "CL",
    "AC": "AC",
    "BR'": "BR",
    "CLT": "CL",
    "CLO": "CL",
    "CLP": "CL",
    "LU": "LU",
    "BA": {"*": "B", "BA": "BA"},
    "CLAU": "CL",
    "RB": "RB",
    "LI": "LI",
    "MOM": "MO",
    "BRAQ": "BR",
    "SR": {"*": "S", "SR": "SR", "OER": "SR"},
    "CLAT": "CL",
    "BRAL": "BR",
    "SEB": "SE",
    "CLW": "CL",
    "CLX": "CL",
    "BE": "BE",
    "BRG": "BR",
    "SEA": "SE",
    "BRAW": "BR",
    "BRBB": "BR",
    "ER": "ER",
    "TH": "TH",
    "BRR": "BR",
    "CLBV": "CL",
    "AL": "AL",
    "CLAV": "CL",
    "BRH": "BR",
    "CLAQ": "CL",
    "GA": "GA",
    "X": "*",
    "TL": "TL",
    "CLBB": "CL",
    "TB": "TB",
    "CLAK": "CL",
    "XE": {"*": "*", "XE": "XE"},
    "SEL": "SE",
    "PU": {"*": "P", "4PU": "PU"},
    "CLAZ": "CL",
    "SE'": "SE",
    "CLBA": "CL",
    "SEN": "SE",
    "SNN": "SN",
    "MOB": "MO",
    "YB": "YB",
    "BRC": "BR",
    "BRD": "BR",
    "CLAM": "CL",
    "DA": "H",
    "DB": "H",
    "DC": "H",
    "DXT": "H",
    "DXU": "H",
    "DXX": "H",
    "DXY": "H",
    "DXZ": "H",
    "DY": "DY",
    "TA": "TA",
    "XD": "*",
    "SED": "SE",
    "CLAL": "CL",
    "BRAJ": "BR",
    "AM": "AM",
    "CLAO": "CL",
    "BI": "BI",
    "KR": "KR",
    "BRBJ": "BR",
    "UNK": "*",
}

hydrophobicity_info = {
    "W": {"Rc": 12.25, "Rc1": 11.1, "Rc2": 11.8, "Rn": 12.25, "Rn1": 12.1},
    "F": {"Rc": 10.90, "Rc1": 7.5, "Rc2": 9.5, "Rn": 10.90, "Rn1": 10.3},
    "L": {"Rc": 9.30, "Rc1": 5.55, "Rc2": 7.4, "Rn": 9.30, "Rn1": 9.3},
    "I": {"Rc": 8.00, "Rc1": 5.2, "Rc2": 6.6, "Rn": 8.00, "Rn1": 7.7},
    "M": {"Rc": 6.20, "Rc1": 4.4, "Rc2": 5.7, "Rn": 6.20, "Rn1": 6.0},
    "V": {"Rc": 5.00, "Rc1": 2.9, "Rc2": 3.4, "Rn": 5.00, "Rn1": 4.2},
    "Y": {"Rc": 4.85, "Rc1": 3.7, "Rc2": 4.5, "Rn": 4.85, "Rn1": 4.4},
    "C": {
        "Rc": 0.45,
        "Rc1": 0.9,
        "Rc2": 0.2,
        "Rn": 0.45,
        "Rn1": -0.5,
    },  # carbamidomethylated Cys
    "P": {"Rc": 2.10, "Rc1": 2.1, "Rc2": 2.1, "Rn": 2.10, "Rn1": 2.1},
    "A": {"Rc": 1.10, "Rc1": 0.35, "Rc2": 0.5, "Rn": 1.10, "Rn1": -0.1},
    "E": {"Rc": 0.95, "Rc1": 1.0, "Rc2": 0.0, "Rn": 0.95, "Rn1": -0.1},
    "T": {"Rc": 0.65, "Rc1": 0.8, "Rc2": 0.6, "Rn": 0.65, "Rn1": 0.0},
    "D": {"Rc": 0.15, "Rc1": 0.5, "Rc2": 0.4, "Rn": 0.15, "Rn1": -0.5},
    "Q": {"Rc": -0.40, "Rc1": -0.7, "Rc2": -0.2, "Rn": -0.40, "Rn1": -1.1},
    "S": {"Rc": -0.15, "Rc1": 0.8, "Rc2": -0.1, "Rn": -0.15, "Rn1": -1.2},
    "G": {"Rc": -0.35, "Rc1": 0.2, "Rc2": 0.15, "Rn": -0.35, "Rn1": -0.7},
    "R": {"Rc": -1.40, "Rc1": 0.5, "Rc2": -1.1, "Rn": -1.30, "Rn1": -1.1},
    "N": {"Rc": -0.85, "Rc1": 0.2, "Rc2": -0.2, "Rn": -0.85, "Rn1": -1.1},
    "H": {"Rc": -1.45, "Rc1": -0.1, "Rc2": -0.2, "Rn": -1.45, "Rn1": -1.7},
    "K": {"Rc": -2.05, "Rc1": -0.6, "Rc2": -1.5, "Rn": -1.90, "Rn1": -1.45},
}

# Nearest-neighbor penalty for hydrophobics adjacent to H/R/K
nn_penalty = {"W": 0.15, "F": 0.10, "L": 0.30, "I": 0.15, "V": 0.20, "Y": 0.05}

# Severity scoring
liability_severity = {
    "UnpairedCys": 10,  # Unpaired cysteine
    # antibody severities
    "DeAmdH": 10,  # High-risk deamidation
    "FragH": 10,  # High fragmentation risk
    "Isom": 10,  # Aspartate isomerization hotspot
    "TrpOx": 10,  # Tryptophan oxidation
    "IntBind": 10,  # Integrin-binding motif
    "DeAmdM": 5,  # Medium-risk deamidation
    "FragM": 5,  # Medium fragmentation risk
    "DeAmdL": 1,  # Low-risk deamidation
    # peptide severities
    "AspBridge": 10,  # Deamidation hotspot (Asn-X)
    "AspCleave": 10,  # Aspartate cleavage site
    "ProtTryp": 10,  # Trypsin cleavage site
    "DPP4": 5,  # DPP4 cleavage site
    # cyclic peptide specific liabilities
    "LowHydrophilic": 7,  # Low overall hydrophilicity
    "ConsecIdentical": 7,  # Consecutive identical residues
    "LongHydrophobic": 7,  # Long hydrophobic stretch
}
default_severity = 5

training_task_probabilities = {
    "select_all": [
        (0, "select_none"),
        (0, "select_scaffold"),
        (0, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (1, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    "0prot_>=0nonprot": [
        (1, "select_none"),
        (0, "select_scaffold"),
        (0, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    "1prot_0nonprot": [
        (0.1, "select_none"),
        (0.5, "select_scaffold"),
        (0.3, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0.1, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    "1prot_>0nonprot": [
        (0.05, "select_none"),
        (0.2, "select_scaffold"),
        (0.15, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0.2, "select_nonprot_interface"),
        (0.4, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    ">1prot_0nonprot": [
        (0.05, "select_none"),
        (0.2, "select_scaffold"),
        (0.15, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0.1, "select_standard_prot"),
        (0.1, "select_protein_intefaces"),
        (0.4, "select_protein_chains"),
    ],
    ">1prot_>0nonprot": [
        (0.05, "select_none"),
        (0.2, "select_scaffold"),
        (0.1, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0.05, "select_nonprot_interface"),
        (0.1, "select_standard_prot"),
        (0.1, "select_protein_intefaces"),
        (0.4, "select_protein_chains"),
    ],
}

training_task_probabilities_with_reindexing = {
    "select_all": [
        (0, "select_none"),
        (0, "select_scaffold"),
        (0, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (1, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    "0prot_>=0nonprot": [
        (1, "select_none"),
        (0, "select_scaffold"),
        (0, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    "1prot_0nonprot": [
        (0.1, "select_none"),
        (0.25, "select_scaffold"),
        (0.15, "select_motif"),
        (0.25, "select_scaffold_binder"),
        (0.15, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0.1, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    "1prot_>0nonprot": [
        (0.05, "select_none"),
        (0.1, "select_scaffold"),
        (0.075, "select_motif"),
        (0.1, "select_scaffold_binder"),
        (0.075, "select_motif_binder"),
        (0.2, "select_nonprot_interface"),
        (0.4, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    ">1prot_0nonprot": [
        (0.05, "select_none"),
        (0.1, "select_scaffold"),
        (0.075, "select_motif"),
        (0.1, "select_scaffold_binder"),
        (0.075, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0.1, "select_standard_prot"),
        (0.1, "select_protein_intefaces"),
        (0.4, "select_protein_chains"),
    ],
    ">1prot_>0nonprot": [
        (0.05, "select_none"),
        (0.1, "select_scaffold"),
        (0.05, "select_motif"),
        (0.1, "select_scaffold_binder"),
        (0.05, "select_motif_binder"),
        (0.05, "select_nonprot_interface"),
        (0.1, "select_standard_prot"),
        (0.1, "select_protein_intefaces"),
        (0.4, "select_protein_chains"),
    ],
}
training_task_probabilities_simple = {
    "select_all": [
        (0, "select_none"),
        (0, "select_scaffold"),
        (0, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (1, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    "0prot_>=0nonprot": [
        (1, "select_none"),
        (0, "select_scaffold"),
        (0, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    "1prot_0nonprot": [
        (0, "select_none"),
        (0.6, "select_scaffold"),
        (0.4, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    "1prot_>0nonprot": [
        (0, "select_none"),
        (0.1, "select_scaffold"),
        (0.1, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0.8, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0, "select_protein_chains"),
    ],
    ">1prot_0nonprot": [
        (0, "select_none"),
        (0.1, "select_scaffold"),
        (0.1, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0.8, "select_protein_chains"),
    ],
    ">1prot_>0nonprot": [
        (0, "select_none"),
        (0, "select_scaffold"),
        (0, "select_motif"),
        (0, "select_scaffold_binder"),
        (0, "select_motif_binder"),
        (0, "select_nonprot_interface"),
        (0.4, "select_standard_prot"),
        (0, "select_protein_intefaces"),
        (0.6, "select_protein_chains"),
    ],
}

eval_keys_confidence = [
    "ligand_iptm",
    "interaction_pae",
    "min_interaction_pae",
    "min_design_to_target_pae",
    "iptm",
    "ptm",
    "protein_iptm",
    "design_iptm",
    "design_iiptm",
    "design_to_target_iptm",
    "target_ptm",
    "design_ptm",
    "ligand_iptm",
]

eval_keys_affinity = [
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_probability_binary2",
    "affinity_pred_value2",
    "affinity_probability_binary1",
    "affinity_pred_value1",
]

eval_keys = (
    eval_keys_confidence
    + eval_keys_affinity
    + [  # additional keys needed to compute folding metrics
        "coords",
        "res_type",
        "input_coords",
        "token_index",
        "atom_resolved_mask",
        "atom_to_token",
        "mol_type",
        "backbone_mask",
    ]
)

folding_dirname = "fold_out_npz"
folding_design_dirname = "fold_out_design_npz"
refold_cif_dirname = "refold_cif"
refold_design_cif_dirname = "refold_design_cif"
affinity_dirname = "affinity_out_npz"
molecules_dirname = "molecules_out_dir"
metrics_dirname = "metrics_tmp"

token_features = [
    "token_index",
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "res_type",
    "res_type_clone",
    "is_standard",
    "design_mask",
    "binding_type",
    "structure_group",
    "token_bonds",
    "type_bonds",
    "token_pad_mask",
    "token_resolved_mask",
    "token_disto_mask",
    "token_pair_mask",
    "contact_conditioning",
    "contact_threshold",
    "method_feature",
    "temp_feature",
    "ph_feature",
    "modified",
    "ccd",
    "cyclic",
    "center_coords",
    "token_distance_mask",
    "target_msa_mask",
    "design_ss_mask",
    "feature_residue_index",
    "feature_asym_id",
    "ligand_affinity_mask",
    "token_to_res",
    "ss_type",
]

atom_features = [
    "ref_pos",
    "atom_resolved_mask",
    "ref_atom_name_chars",
    "ref_element",
    "ref_charge",
    "ref_chirality",
    "atom_backbone_feat",
    "ref_space_uid",
    "coords",
    "atom_pad_mask",
    "atom_to_token",
    "new_to_old_atomidx",
    "bfactor",
    "plddt",
    "masked_ref_atom_name_chars",
    "backbone_mask",
    "fake_atom_mask",
]

# Formal charges (because CCD ones are missing negative charges???)
formal_charges = {k: defaultdict(int) for k in prot_token_to_letter.keys()}
formal_charges["ASP"]["OD2"] = -1
formal_charges["GLU"]["OE2"] = -1
formal_charges["LYS"]["NZ"] = 1
formal_charges["ARG"]["NH2"] = 1
# Note: Histidine is protonated in CCD, but usually isn't at neutral pH


### Kernel details
@torch.compiler.disable
def kernel_triangular_attn(q, k, v, tri_bias, mask, scale):
    from cuequivariance_torch.primitives.triangle import triangle_attention
    return triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)


@torch.compiler.disable  # noqa: E402 – decorator must follow import of torch
def _kernel_triangular_mult(
    x: Tensor,
    *,
    direction: str,
    mask: Tensor,
    norm_in_weight: Tensor,
    norm_in_bias: Tensor,
    p_in_weight: Tensor,
    g_in_weight: Tensor,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    p_out_weight: Tensor,
    g_out_weight: Tensor,
    eps: float,
):
    try:
        from cuequivariance_torch.primitives.triangle import (
            triangle_multiplicative_update as _triangle_multiplicative_update,
        )
    except ModuleNotFoundError:
        raise RuntimeError(
            "cuEquivariance kernels requested via use_kernels=True but the package is not available."
        )

    return _triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


### Utils
LinearNoBias = partial(nn.Linear, bias=False)


def optionally_tqdm(iterable, use_tqdm=False, desc=""):
    if use_tqdm:
        return tqdm(iterable, desc=desc)
    return iterable


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def one_hot(x, num_classes):
    """One-hot encoding."""
    return F.one_hot(x.long(), num_classes=num_classes)


def sigmoid(x):
    """Sigmoid activation."""
    return torch.sigmoid(x)


def pad(x, padding, value=0):
    """Pad tensor."""
    return F.pad(x, padding, value=value)


def compute_aggregated_metric(logits, start=0, end=None):
    """Compute aggregated metric from logits."""
    if end is None:
        end = logits.shape[-1]
    probs = F.softmax(logits, dim=-1)
    bin_values = torch.arange(start, end, device=logits.device, dtype=logits.dtype)
    return (probs[..., start:end] * bin_values).sum(dim=-1)


def weighted_rigid_align(x, y, x_mask, y_mask):
    """Weighted rigid alignment (Kabsch algorithm)."""
    # Compute centroids
    x_mean = (x * x_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / x_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
    y_mean = (y * y_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / y_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
    
    # Center
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    # Compute covariance matrix
    mask = (x_mask * y_mask).unsqueeze(-1).unsqueeze(-1)
    H = torch.matmul(
        (x_centered * mask.squeeze(-1)).transpose(-1, -2),
        y_centered * mask.squeeze(-1)
    )
    
    # SVD
    U, S, Vh = torch.linalg.svd(H)
    
    # Rotation matrix
    d = torch.det(torch.matmul(Vh.transpose(-1, -2), U.transpose(-1, -2)))
    V = Vh.transpose(-1, -2)
    U_t = U.transpose(-1, -2)
    
    # Handle reflection
    eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
    eye = eye.clone()
    eye[..., -1, -1] = d
    R = torch.matmul(torch.matmul(V, eye), U_t)
    
    # Apply transformation
    x_aligned = torch.matmul(x_centered, R) + y_mean
    
    return x_aligned


def compute_ptms(pae_logits, x_pred, feats, multiplicity):
    """Compute PTM and iPTM metrics - placeholder implementation."""
    batch_size = pae_logits.shape[0]
    device = pae_logits.device
    
    # Placeholder - full implementation requires proper PAE computation
    ptm = torch.ones(batch_size, device=device) * 0.8
    iptm = torch.ones(batch_size, device=device) * 0.7
    ligand_iptm = torch.ones(batch_size, device=device) * 0.7
    protein_iptm = torch.ones(batch_size, device=device) * 0.7
    pair_chains_iptm = {}
    design_to_target_iptm = torch.ones(batch_size, device=device) * 0.7
    design_iptm = torch.ones(batch_size, device=device) * 0.7
    design_iiptm = torch.ones(batch_size, device=device) * 0.7
    target_ptm = torch.ones(batch_size, device=device) * 0.8
    design_ptm = torch.ones(batch_size, device=device) * 0.8
    
    return (
        ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm,
        design_to_target_iptm, design_iptm, design_iiptm,
        target_ptm, design_ptm
    )


def minimum_lddt_symmetry_dist(pred_distogram, feats, index_batch):
    """Compute minimum LDDT symmetry distance - placeholder."""
    pass


class SwiGLU(Module):
    def forward(
        self,
        x,  #: Float['... d']
    ):  # -> Float[' ... (d//2)']:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


def center(atom_coords, atom_mask):
    atom_mean = torch.sum(
        atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
    ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
    atom_coords = atom_coords - atom_mean
    return atom_coords


def compute_random_augmentation(
    multiplicity,
    s_trans=1.0,
    device=None,
    dtype=torch.float32
):
    R = random_rotations(multiplicity, dtype=dtype, device=device)
    random_trans = torch.randn((multiplicity, 1, 3), dtype=dtype, device=device) * s_trans
    return R, random_trans


def randomly_rotate(coords, return_second_coords=False, second_coords=None):
    R = random_rotations(len(coords), coords.dtype, coords.device)

    if return_second_coords:
        return torch.einsum("bmd,bds->bms", coords, R), torch.einsum(
            "bmd,bds->bms", second_coords, R
        ) if second_coords is not None else None

    return torch.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
):
    """Algorithm 19"""
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords, return_second_coords=True, second_coords=second_coords
        )
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords


class GaussianRandom3DEncodings(Module):
    def __init__(self, dim=50):
        super().__init__()
        center = torch.randn((dim, 3))
        std = torch.rand((dim))
        self.dim = dim
        self.register_buffer("center", center)
        self.register_buffer("std", std)

    def forward(self, coords):
        B, N, _ = coords.shape
        dist2 = (
            (coords.view(B, N, 1, 3) - self.center.view(1, 1, self.dim, 3)) ** 2
        ).sum(dim=-1)
        emb = torch.exp(-dist2 / (2 * self.std.view(1, 1, self.dim)))
        return emb


# the following is copied from Torch3D, BSD License, Copyright (c) Meta Platforms, Inc. and affiliates.


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)


def scatter_sum(
    src: Tensor, 
    index: Tensor, 
    dim: int = -1,
    dim_size: Optional[int] = None
) -> Tensor:
    """
    Replacement for torch_scatter.scatter_sum using native PyTorch.
    
    Args:
        src: Source tensor to scatter
        index: Index tensor for scattering
        dim: Dimension along which to scatter
        dim_size: Optional size of the output dimension
        
    Returns:
        Scattered sum tensor
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    
    # Create output shape
    shape = list(src.shape)
    shape[dim] = dim_size
    
    # Initialize output tensor with zeros
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    # Expand index to match src dimensions
    expanded_index = index
    for _ in range(len(src.shape) - len(index.shape)):
        expanded_index = expanded_index.unsqueeze(-1)
    expanded_index = expanded_index.expand_as(src)
    
    # Use scatter_add_ for summation
    out.scatter_add_(dim, expanded_index, src)
    
    return out


def scatter_softmax(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None
) -> Tensor:
    """
    Replacement for torch_scatter.scatter_softmax using native PyTorch.
    
    Args:
        src: Source tensor to apply softmax
        index: Index tensor for grouping
        dim: Dimension along which to apply softmax
        dim_size: Optional size of the output dimension
        
    Returns:
        Tensor with softmax applied within each group
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    
    # Get max values for numerical stability
    max_value_per_index = scatter_max(src, index, dim, dim_size)
    
    # Expand index to match src dimensions for gather
    expanded_index = index
    for _ in range(len(src.shape) - len(index.shape)):
        expanded_index = expanded_index.unsqueeze(-1)
    expanded_index = expanded_index.expand_as(src)
    
    max_src = max_value_per_index.gather(dim, expanded_index)
    
    # Compute exp(src - max)
    exp_src = torch.exp(src - max_src)
    
    # Sum exp values per group
    sum_exp = scatter_sum(exp_src, index, dim, dim_size)
    
    # Gather sum for each element
    sum_exp_per_src = sum_exp.gather(dim, expanded_index)
    
    # Compute softmax
    out = exp_src / (sum_exp_per_src + 1e-10)  # Add small epsilon for numerical stability
    
    return out


def scatter_max(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None
) -> Tensor:
    """
    Helper function to compute scatter max using native PyTorch.
    
    Args:
        src: Source tensor
        index: Index tensor for grouping
        dim: Dimension along which to compute max
        dim_size: Optional size of the output dimension
        
    Returns:
        Tensor with max values per group
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Create output shape
    shape = list(src.shape)
    shape[dim] = dim_size
    
    # Initialize with -inf for max operation
    out = torch.full(shape, float('-inf'), dtype=src.dtype, device=src.device)
    
    # Create expanded index for scatter
    expanded_index = index
    for _ in range(len(src.shape) - len(index.shape)):
        expanded_index = expanded_index.unsqueeze(-1)
    expanded_index = expanded_index.expand_as(src)
    
    # Use scatter_reduce for max operation (PyTorch >= 1.12)
    if hasattr(out, 'scatter_reduce_'):
        out.scatter_reduce_(dim, expanded_index, src, reduce='amax', include_self=False)
    else:
        # Fallback for older PyTorch versions
        out.scatter_(dim, expanded_index, src, reduce='max')
    
    # Replace -inf with 0 for indices that were never written to
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    
    return out


def softmax_dropout(
    attn_weight: Tensor, softmax_dropout: float, dst_idx: Tensor = None
):
    for _ in range(10):
        dropout_mask = torch.rand_like(attn_weight) < softmax_dropout
        not_all_drop_mask = scatter_sum(~dropout_mask, index=dst_idx, dim=0)
        if not_all_drop_mask.all():
            attn_weight = attn_weight.masked_fill(dropout_mask, -float("inf"))
            return attn_weight
    raise RuntimeError(
        "Softmax dropout failed to keep at least one edge for each node after 10 attempts."
    )


def get_dropout_mask(
    dropout: float,
    z: torch.Tensor,
    training: bool,
    columnwise: bool = False,
) -> torch.Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    Tensor
        The dropout mask

    """
    dropout = dropout * training
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    d = torch.rand_like(v) > dropout
    d = d * 1.0 / (1.0 - dropout)
    return d


def get_dropout_mask_columnwise(
    dropout: float,
    z: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    Tensor
        The dropout mask

    """
    dropout = dropout * training
    v = z[:, 0:1, :, 0:1]
    d = torch.rand_like(v) > dropout
    d = d * 1.0 / (1.0 - dropout)
    return d


def get_dropout_mask_rowise(
    dropout: float,
    z: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    Tensor
        The dropout mask

    """
    dropout = dropout * training
    v = z[:, :, 0:1, 0:1]
    d = torch.rand_like(v) > dropout
    d = d * 1.0 / (1.0 - dropout)
    return d


def add(m1, m2, inplace):
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def is_fp16_enabled():
    # Autocast world
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()

    return fp16_enabled


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Tree of type {type(tree)} not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def _fetch_dims(tree):
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError("Not supported")

    return shapes


@torch.jit.ignore
def _flat_idx_to_idx(
    flat_idx: int,
    dims: Tuple[int],
) -> Tuple[int]:
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d

    return tuple(reversed(idx))


@torch.jit.ignore
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: int,
    start_edges: Optional[Sequence[bool]] = None,
    end_edges: Optional[Sequence[bool]] = None,
) -> Sequence[Tuple[int]]:
    """
    Produces an ordered sequence of tensor slices that, when used in
    sequence on a tensor with shape dims, yields tensors that contain every
    leaf in the contiguous range [start, end]. Care is taken to yield a
    short sequence of slices, and perhaps even the shortest possible (I'm
    pretty sure it's the latter).

    end is INCLUSIVE.
    """

    # start_edges and end_edges both indicate whether, starting from any given
    # dimension, the start/end index is at the top/bottom edge of the
    # corresponding tensor, modeled as a tree
    def reduce_edge_list(l):
        tally = 1
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] *= tally
            tally = l[reversed_idx]

    if start_edges is None:
        start_edges = [s == 0 for s in start]
        reduce_edge_list(start_edges)
    if end_edges is None:
        end_edges = [e == (d - 1) for e, d in zip(end, dims)]
        reduce_edge_list(end_edges)

        # Base cases. Either start/end are empty and we're done, or the final,
    # one-dimensional tensor can be simply sliced
    if len(start) == 0:
        return [tuple()]
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]

    slices = []
    path = []

    # Dimensions common to start and end can be selected directly
    for s, e in zip(start, end):
        if s == e:
            path.append(slice(s, s + 1))
        else:
            break

    path = tuple(path)
    divergence_idx = len(path)

    # start == end, and we're done
    if divergence_idx == len(dims):
        return [tuple(path)]

    def upper():
        sdi = start[divergence_idx]
        return [
            path + (slice(sdi, sdi + 1),) + s
            for s in _get_minimal_slice_set(
                start[divergence_idx + 1 :],
                [d - 1 for d in dims[divergence_idx + 1 :]],
                dims[divergence_idx + 1 :],
                start_edges=start_edges[divergence_idx + 1 :],
                end_edges=[1 for _ in end_edges[divergence_idx + 1 :]],
            )
        ]

    def lower():
        edi = end[divergence_idx]
        return [
            path + (slice(edi, edi + 1),) + s
            for s in _get_minimal_slice_set(
                [0 for _ in start[divergence_idx + 1 :]],
                end[divergence_idx + 1 :],
                dims[divergence_idx + 1 :],
                start_edges=[1 for _ in start_edges[divergence_idx + 1 :]],
                end_edges=end_edges[divergence_idx + 1 :],
            )
        ]

    # If both start and end are at the edges of the subtree rooted at
    # divergence_idx, we can just select the whole subtree at once
    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx] + 1),))
    # If just start is at the edge, we can grab almost all of the subtree,
    # treating only the ragged bottom edge as an edge case
    elif start_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx]),))
        slices.extend(lower())
    # Analogous to the previous case, but the top is ragged this time
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(
            path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),)
        )
    # If both sides of the range are ragged, we need to handle both sides
    # separately. If there's contiguous meat in between them, we can index it
    # in one big chunk
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(
                path + (slice(start[divergence_idx] + 1, end[divergence_idx]),)
            )
        slices.extend(lower())

    return [tuple(s) for s in slices]


@torch.jit.ignore
def _chunk_slice(
    t: torch.Tensor,
    flat_start: int,
    flat_end: int,
    no_batch_dims: int,
) -> torch.Tensor:
    """
    Equivalent to

        t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

    but without the need for the initial reshape call, which can be
    memory-intensive in certain situations. The only reshape operations
    in this function are performed on sub-tensors that scale with
    (flat_end - flat_start), the chunk size.
    """

    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    # _get_minimal_slice_set is inclusive
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))

    # Get an ordered list of slices to perform
    slices = _get_minimal_slice_set(
        start_idx,
        end_idx,
        batch_dims,
    )

    sliced_tensors = [t[s] for s in slices]

    return torch.cat([s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors])


def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
    _out: Any = None,
    _add_into_out: bool = False,
) -> Any:
    """
    Implements the "chunking" procedure described in section 1.11.8.

    Layer outputs and inputs are assumed to be simple "pytrees,"
    consisting only of (arbitrarily nested) lists, tuples, and dicts with
    torch.Tensor leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (non-nested) dictionary of keyworded inputs. All leaves must
            be tensors and must share the same batch dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch
            dimensions are specified, a "sub-batch" is defined as a single
            indexing of all batch dimensions simultaneously (s.t. the
            number of sub-batches is the product of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can
            be considered batch dimensions.
        low_mem:
            Avoids flattening potentially large input tensors. Unnecessary
            in most cases, and is ever so slightly slower than the default
            setting.
    Returns:
        The reassembled output of the layer on the inputs.
    """
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    def _prep_inputs(t):
        if not low_mem:
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t

    prepped_inputs = tensor_tree_map(_prep_inputs, inputs)
    prepped_outputs = None
    if _out is not None:
        reshape_fn = lambda t: t.view([-1] + list(t.shape[no_batch_dims:]))
        prepped_outputs = tensor_tree_map(reshape_fn, _out)

    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d

    no_chunks = flat_batch_dim // chunk_size + (flat_batch_dim % chunk_size != 0)

    i = 0
    out = prepped_outputs
    for _ in range(no_chunks):
        # Chunk the input
        if not low_mem:
            select_chunk = lambda t: t[i : i + chunk_size] if t.shape[0] != 1 else t
        else:
            select_chunk = partial(
                _chunk_slice,
                flat_start=i,
                flat_end=min(flat_batch_dim, i + chunk_size),
                no_batch_dims=len(orig_batch_dims),
            )

        chunks = tensor_tree_map(select_chunk, prepped_inputs)

        # Run the layer on the chunk
        output_chunk = layer(**chunks)

        # Allocate space for the output
        if out is None:
            allocate = lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:])
            out = tensor_tree_map(allocate, output_chunk)

        # Put the chunk in its pre-allocated space
        out_type = type(output_chunk)
        if out_type is dict:

            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    else:
                        if _add_into_out:
                            v[i : i + chunk_size] += d2[k]
                        else:
                            v[i : i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                if _add_into_out:
                    x1[i : i + chunk_size] += x2
                else:
                    x1[i : i + chunk_size] = x2
        elif out_type is torch.Tensor:
            if _add_into_out:
                out[i : i + chunk_size] += output_chunk
            else:
                out[i : i + chunk_size] = output_chunk
        else:
            raise ValueError("Not supported")

        i += chunk_size

    reshape = lambda t: t.view(orig_batch_dims + t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def compute_relative_distribution_perfect_correlation(
    binned_distribution_1, binned_distribution_2
):
    """
    Compute the relative distribution between two binned distributions with perfect correlation.

    Parameters
    ----------
    binned_distribution_1 : torch.Tensor
        The first binned distribution, shape (..., K).
    binned_distribution_2 : torch.Tensor
        The second binned distribution, shape (..., K).

    Returns
    -------
    torch.Tensor
        The relative distribution, shape (..., 2K - 1).

    """
    K = binned_distribution_1.shape[-1]
    relative_distribution = torch.zeros(
        binned_distribution_1.shape[:-1] + (2 * K - 1,),
        device=binned_distribution_1.device,
    )
    zero = torch.zeros(
        binned_distribution_1.shape[:-1] + (1,), device=binned_distribution_1.device
    )

    binned_distribution_1 = torch.cat([zero, binned_distribution_1], dim=-1)
    binned_distribution_2 = torch.cat([zero, binned_distribution_2], dim=-1)

    cumulative_1 = torch.cumsum(binned_distribution_1, dim=-1)
    cumulative_2 = torch.cumsum(binned_distribution_2, dim=-1)

    for i in range(K):
        relative_distribution[..., K - 1 + i] = torch.sum(
            torch.relu(
                torch.minimum(
                    cumulative_1[..., 1 + i :], cumulative_2[..., 1 : K + 1 - i]
                )
                - torch.maximum(cumulative_1[..., i:-1], cumulative_2[..., : K - i]),
            )
        )

    for i in range(1, K):
        relative_distribution[..., K - 1 - i] = torch.sum(
            torch.relu(
                torch.minimum(
                    cumulative_2[..., 1 + i :], cumulative_1[..., 1 : K + 1 - i]
                )
                - torch.maximum(cumulative_2[..., i:-1], cumulative_1[..., : K - i]),
            )
        )

    return relative_distribution


class OuterProductMean(Module):
    """Outer product mean layer."""

    def __init__(self, c_in: int, c_hidden: int, c_out: int) -> None:
        """Initialize the pair weighted averaging layer.

        Parameters
        ----------
        c_in : int
            The input dimension.
        c_hidden : int
            The hidden dimension.
        c_out : int
            The output dimension.

        """
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)

    def forward(
        self,
        m: Tensor,
        mask: Tensor,
        chunk_size: int = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        m : torch.Tensor
            The sequence tensor (B, S, N, c_in).
        mask : torch.Tensor
            The mask tensor (B, S, N).

        Returns
        -------
        torch.Tensor
            The output tensor (B, N, N, c_out).

        """
        # Set mask to dtype
        mask = mask.to(m)

        # Compute projections
        m = self.norm(m)
        a = self.proj_a(m) * mask[..., None]
        b = self.proj_b(m) * mask[..., None]

        # Compute outer product mean
        if chunk_size is not None and not self.training:
            # Compute pairwise mask
            mask = mask.unsqueeze(-1)

            for i in range(0, mask.shape[1], 64):
                if i == 0:
                    num_mask = (
                        mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
                    ).sum(1)
                else:
                    num_mask += (
                        mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
                    ).sum(1)
            num_mask = num_mask.clamp(min=1)

            for i in range(0, self.c_hidden, chunk_size):
                a_chunk = a[:, :, :, i : i + chunk_size]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, i * self.c_hidden : (i + chunk_size) * self.c_hidden
                ]
                with torch.autocast("cuda", enabled=False):
                    z = torch.einsum("bsic,bsjd->bijcd", a_chunk.float(), b.float())

                z = z.reshape(*z.shape[:3], -1)
                z = (z / num_mask).to(m)

                # Project to output
                if i == 0:
                    z_out = z @ sliced_weight_proj_o.T
                else:
                    z_out = z_out + z @ sliced_weight_proj_o.T
            z_out = z_out + self.proj_o.bias
            return z_out
        else:
            # Compute outer product
            with torch.autocast("cuda", enabled=False):
                z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())

            # Compute mask sum
            mask_sum = torch.einsum("bsi,bsj->bij", mask, mask)
            mask_sum = mask_sum.clamp(min=1)

            # Reshape and normalize
            z = z.reshape(*z.shape[:3], -1)
            z = (z / mask_sum[..., None]).to(m)

            # Project to output
            z = self.proj_o(z)
            return z


class PairWeightedAveraging(Module):
    """Pair weighted averaging layer."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_h: int,
        num_heads: int,
        inf: float = 1e6,
    ) -> None:
        """Initialize the pair weighted averaging layer."""
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf

        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)

        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)

    def forward(
        self, m: Tensor, z: Tensor, mask: Tensor, chunk_heads: False = bool
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        m : torch.Tensor
            The input sequence tensor (B, S, N, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor (B, S, N, D)

        """
        # Compute layer norms
        m = self.norm_m(m)
        z = self.norm_z(z)

        if chunk_heads and not self.training:
            o_chunks = []
            for head_idx in range(self.num_heads):
                # NOTE: Linear layers have no bias so no need to slice them
                sliced_weight_proj_m = self.proj_m.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_g = self.proj_g.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_z = self.proj_z.weight[head_idx : (head_idx + 1), :]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, head_idx * self.c_h : (head_idx + 1) * self.c_h
                ]
                v: Tensor = m @ sliced_weight_proj_m.T
                v = v.reshape(*v.shape[:3], 1, self.c_h)
                v = v.permute(0, 3, 1, 2, 4)

                # Compute weights
                b: Tensor = z @ sliced_weight_proj_z.T
                b = b.permute(0, 3, 1, 2)
                b = b + (1 - mask[:, None]) * -self.inf
                w = torch.softmax(b, dim=-1)

                # Compute gating
                g: Tensor = m @ sliced_weight_proj_g.T
                g = g.sigmoid()

                # Compute output
                o = torch.einsum("bhij,bhsjd->bhsid", w, v)
                o = o.permute(0, 2, 3, 1, 4)
                o = o.reshape(*o.shape[:3], 1 * self.c_h)
                o_chunks = g * o
                if head_idx == 0:
                    o_out = o_chunks @ sliced_weight_proj_o.T
                else:
                    o_out += o_chunks @ sliced_weight_proj_o.T
            return o_out
        else:
            # Project input tensors
            v: Tensor = self.proj_m(m)
            v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
            v = v.permute(0, 3, 1, 2, 4)

            # Compute weights
            b: Tensor = self.proj_z(z)
            b = b.permute(0, 3, 1, 2)
            b = b + (1 - mask[:, None]) * -self.inf
            w = torch.softmax(b, dim=-1)

            # Compute gating
            g: Tensor = self.proj_g(m)
            g = g.sigmoid()

            # Compute output
            o = torch.einsum("bhij,bhsjd->bhsid", w, v)
            o = o.permute(0, 2, 3, 1, 4)
            o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
            concat_o = g * o

            o = self.proj_o(concat_o)
            return o


### Basic layers
class Transition(Module):
    """Perform a two-layer MLP."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        out_dim: int = None,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        hidden: int
            The dimension of the hidden, default 512

        """
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

    def forward(self, x: Tensor, chunk_size: int = None) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        x = self.norm(x)

        if chunk_size is None or self.training:
            x = self.silu(self.fc1(x)) * self.fc2(x)
            x = self.fc3(x)
            return x
        else:
            for i in range(0, self.hidden, chunk_size):
                fc1_slice = self.fc1.weight[i : i + chunk_size, :]
                fc2_slice = self.fc2.weight[i : i + chunk_size, :]
                fc3_slice = self.fc3.weight[:, i : i + chunk_size]
                x_chunk = self.silu((x @ fc1_slice.T)) * (x @ fc2_slice.T)
                if i == 0:
                    x_out = x_chunk @ fc3_slice.T
                else:
                    x_out = x_out + x_chunk @ fc3_slice.T
            return x_out


class LayerNorm(Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        if d is torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                out = nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=d),
                    self.bias.to(dtype=d),
                    self.eps,
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    d = t.dtype
    if d is torch.bfloat16:
        with torch.autocast("cuda", enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


# @torch.jit.script
def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


class Attention(Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = nn.Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_k = nn.Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_v = nn.Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_o = nn.Linear(
            self.c_hidden * self.no_heads, self.c_q, bias=False
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = nn.Linear(
                self.c_q, self.c_hidden * self.no_heads, bias=False
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        tri_bias: torch.Tensor,
        mask_bias: torch.Tensor,
        mask: torch.Tensor,
        use_kernels: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_kernels:
                Whether to use the cuEquivariance triangle kernel.

        Returns
            [*, Q, C_q] attention update
        """
        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=not use_kernels)

        if use_kernels:
            scale = 1.0 / math.sqrt(self.c_hidden)
            o = kernel_triangular_attn(
                q,
                k,
                v,
                tri_bias=tri_bias,
                mask=mask.bool(),
                scale=scale,
            )
            o = o.transpose(-2, -3)

        else:
            biases = [mask_bias, tri_bias]
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


class AttentionPairBias(Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: int = None,
        num_heads: int = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
        use_qk_norm: bool = False,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6

        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        self.compute_pair_bias = compute_pair_bias
        if compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )
        else:
            self.proj_z = Rearrange("b ... h -> b h ...")

        self.proj_o = nn.Linear(c_s, c_s, bias=False)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        k_in: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor or bias (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """

        B = s.shape[0]

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
        """
        TODO
        The k and v part should be done like this instead for efficiency reasons in the next version of boltz
        self.proj_kv = nn.Linear(c_s, 2*c_s, bias=False)
        kv = self.proj_kv(k_in).view(B, -1, self.num_heads, 2*self.head_dim).permute(0, 2, 1, 3)
        k,v = torch.chunk(kv, chunks=2, dim=3) # chunking (B,H,N,2C) into 2x (B,H,N,C)
        """

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        bias = self.proj_z(z)
        bias = bias.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s)
        g.sigmoid_()

        attn_mask = (1 - mask[:, None, None].float()) * -self.inf
        attn_mask = attn_mask + bias.float()

        with torch.autocast("cuda", enabled=False):
            # Compute attention weights
            o = torch.nn.functional.scaled_dot_product_attention(
                q.float(),
                k.float(),
                v.float(),
                attn_mask=attn_mask,
            )

        o = o.permute(0, 2, 1, 3).reshape(B, -1, self.c_s)
        o = o * g
        o = self.proj_o(o)
        return o


class TriangleAttention(Module):
    """Implement Algorithm 12."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ) -> None:
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = nn.Linear(c_in, self.no_heads, bias=False)

        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        tri_bias: torch.Tensor,
        mask_bias: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
        use_kernels: bool = False,
    ) -> torch.Tensor:
        """Compute triangle attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [*, I, J, C_in]
        biases : list[torch.Tensor]
            List of bias tensors of shape [*, H, I, J]
        chunk_size : int
            Size of chunks for memory efficient computation
        use_kernels : bool, default=False
            Whether to use optimized CUDA kernels

        Returns
        -------
        torch.Tensor
            Output tensor of shape [*, I, J, C_in]

        """
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "tri_bias": tri_bias,
            "mask_bias": mask_bias,
            "mask": mask,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_kernels=use_kernels,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_kernels: bool = False,
    ) -> torch.Tensor:
        """Compute triangle attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [*, I, J, C_in]
        mask : torch.Tensor, optional
            Attention mask of shape [*, I, J]
        chunk_size : int, optional
            Size of chunks for memory efficient computation
        use_kernels : bool, default=False
            Whether to use optimized CUDA kernels

        Returns
        -------
        torch.Tensor
            Output tensor of shape [*, I, J, C_in]

        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask = mask[..., :, None, None, :]
        mask_bias = self.inf * (mask - 1)

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        if chunk_size is not None and not use_kernels:
            x = self._chunk(
                x,
                triangle_bias,
                mask_bias,
                mask,
                chunk_size,
                use_kernels=use_kernels,
            )
        else:
            x = self.mha(
                x,
                x,
                triangle_bias,
                mask_bias,
                mask,
                use_kernels=use_kernels,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """Implement Algorithm 14."""

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)


class MiniTriangularUpdate(Module):
    """Perform a bi-directional triangular update.

    This module differs from the original multiplicative
    update introduced in AlphaFold2 in several ways. First,
    we merge the incoming and outgoing layers in a single
    update. Second, and related to the  above change, we
    down-project the input to D // 4. This allows us to keep
    memory constant. Third, we modify the output gate to be
    a function of the output instead of the intput, which
    allows us to use the same gating kernel for both the
    input and output gates, and thereby save some more memory.

    """

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, dim, bias=False)
        self.g_in = nn.Linear(dim, dim, bias=False)

        self.norm_out = nn.LayerNorm(dim // 2)
        self.p_out = nn.Linear(dim // 2, dim, bias=False)
        self.g_out = nn.Linear(dim // 2, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor, *, use_kernels: bool = False) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """

        # Input gating: D -> D
        x = self.norm_in(x)
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        with torch.autocast("cuda", enabled=False):
            a1, b1, a2, b2 = torch.chunk(x.float(), 4, dim=-1)

            # Triangular projection
            x1 = torch.einsum("bikd,bjkd->bijd", a1, b1)
            x2 = torch.einsum("bkid,bkjd->bijd", a2, b2)

            # Merge outputs
            x = torch.cat([x1, x2], dim=-1).to(x.dtype)

        # Output gating: D / 2 -> D
        x = self.norm_out(x)
        x = self.p_out(x) * self.g_out(x).sigmoid()

        return x


class TriangleMultiplicationOutgoing(Module):
    """TriangleMultiplicationOutgoing."""

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if use_kernels:
            return _kernel_triangular_mult(
                x,
                direction="outgoing",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        with torch.autocast("cuda", enabled=False):
            a, b = torch.chunk(x.float(), 2, dim=-1)

            # Triangular projection
            x = torch.einsum("bikd,bjkd->bijd", a, b).to(x.dtype)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x


class TriangleMultiplicationIncoming(Module):
    """TriangleMultiplicationIncoming."""

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if use_kernels:
            return _kernel_triangular_mult(
                x,
                direction="incoming",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        with torch.autocast("cuda", enabled=False):
            a, b = torch.chunk(x.float(), 2, dim=-1)

            # Triangular projection
            x = torch.einsum("bkid,bkjd->bijd", a, b).to(x.dtype)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x


### Transformers
class MiniformerModule(Module):
    """Miniformer module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        use_s_to_z: bool = False,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.use_s_to_z = use_s_to_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm
        self.layers = ModuleList()
        self.activation_checkpointing = activation_checkpointing

        for i in range(num_blocks):
            self.layers.append(
                MiniformerLayer(
                    token_s,
                    token_z,
                    num_heads,
                    dropout,
                    post_layer_norm,
                    use_s_to_z,
                ),
            )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:
            if self.activation_checkpointing:
                s, z = torch.utils.checkpoint.checkpoint(layer, s, z, mask, pair_mask)
            else:
                s, z = layer(s, z, mask, pair_mask)
        return s, z


class MiniformerLayer(Module):
    """Miniformer layer."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        use_s_to_z: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm
        self.use_s_to_z = use_s_to_z

        self.pre_norm_s = nn.LayerNorm(token_s)
        self.attention = AttentionPairBias(token_s, token_z, num_heads)

        self.triangular = MiniTriangularUpdate(token_z)

        self.transition_s = Transition(token_s, token_s * 4)
        self.transition_z = Transition(token_z, token_z * 4)

        if self.post_layer_norm:
            self.s_post_norm = nn.LayerNorm(token_s)
        else:
            self.s_post_norm = nn.Identity()

        if self.use_s_to_z:
            self.s_to_z_1 = nn.Linear(token_s, token_z, bias=False)
            self.s_to_z_2 = nn.Linear(token_s, token_z, bias=False)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        # Compute s to z messages
        if self.use_s_to_z:
            s_to_z = self.s_to_z_1(s)[:, :, None] + self.s_to_z_2(s)[:, None, :]
            z = z + s_to_z

        # Compute pairwise stack
        dropout = get_dropout_mask_rowise(self.dropout, z, self.training)
        z = z + dropout * self.triangular(z, mask=pair_mask)
        z = z + self.transition_z(z)

        # Compute sequence stack
        with torch.autocast("cuda", enabled=False):
            s_normed = self.pre_norm_s(s.float())
            s = s.float() + self.attention(
                s=s_normed, z=z.float(), mask=mask.float(), k_in=s_normed
            )
            s = s + self.transition_s(s)
            s = self.s_post_norm(s)

        return s, z


class MiniformerNoSeqModule(Module):
    """Miniformer module without sequence track."""

    def __init__(
        self,
        token_z: int,
        num_blocks: int,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm
        self.layers = ModuleList()
        self.activation_checkpointing = activation_checkpointing

        for i in range(num_blocks):
            self.layers.append(
                MiniformerNoSeqLayer(
                    token_z,
                    dropout,
                    post_layer_norm,
                ),
            )

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        for layer in self.layers:
            if self.activation_checkpointing:
                z = torch.utils.checkpoint.checkpoint(layer, z, pair_mask)
            else:
                z = layer(z, pair_mask)
        return z


class MiniformerNoSeqLayer(Module):
    """Miniformer layer without sequence track."""

    def __init__(
        self,
        token_z: int,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm

        self.triangular = MiniTriangularUpdate(token_z)
        self.transition_z = Transition(token_z, token_z * 4)

        if self.post_layer_norm:
            self.z_post_norm = nn.LayerNorm(token_z)
        else:
            self.z_post_norm = nn.Identity()

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: int = None,
        use_kernels: bool = False,
    ) -> Tensor:
        # Compute pairwise stack
        dropout = get_dropout_mask_rowise(self.dropout, z, self.training)
        z = z + dropout * self.triangular(z, mask=pair_mask)
        z = z + self.transition_z(z)

        # Post-LN
        z = self.z_post_norm(z)
        return z


class PairformerModule(Module):
    """Pairformer module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm
        self.activation_checkpointing = activation_checkpointing

        self.layers = ModuleList()
        for i in range(num_blocks):
            self.layers.append(
                PairformerLayer(
                    token_s,
                    token_z,
                    num_heads,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                    post_layer_norm,
                ),
            )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        *,
        use_kernels: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        if not self.training:
            if z.shape[1] > chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            if self.activation_checkpointing:
                s, z = torch.utils.checkpoint.checkpoint(
                    layer,
                    s,
                    z,
                    mask,
                    pair_mask,
                    chunk_size_tri_attn,
                    use_kernels=use_kernels,
                )
            else:
                s, z = layer(
                    s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels=use_kernels
                )
        return s, z


class PairformerLayer(Module):
    """Pairformer module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm

        self.pre_norm_s = nn.LayerNorm(token_s)
        self.attention = AttentionPairBias(token_s, token_z, num_heads)

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)

        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )

        self.transition_s = Transition(token_s, token_s * 4)
        self.transition_z = Transition(token_z, token_z * 4)

        self.s_post_norm = (
            nn.LayerNorm(token_s) if self.post_layer_norm else nn.Identity()
        )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: int = None,
        *,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        # Compute pairwise stack
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(
            z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(
            z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        z = z + self.transition_z(z)

        # Compute sequence stack
        with torch.autocast("cuda", enabled=False):
            s_normed = self.pre_norm_s(s.float())
            s = s.float() + self.attention(
                s=s_normed, z=z.float(), mask=mask.float(), k_in=s_normed
            )
            s = s + self.transition_s(s)
            s = self.s_post_norm(s)

        return s, z


class PairformerNoSeqModule(Module):
    """Pairformer module without sequence track."""

    def __init__(
        self,
        token_z: int,
        num_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm
        self.activation_checkpointing = activation_checkpointing

        self.layers = ModuleList()
        for i in range(num_blocks):
            self.layers.append(
                PairformerNoSeqLayer(
                    token_z,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                    post_layer_norm,
                ),
            )

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        *,
        use_kernels: bool = False,
    ) -> Tensor:
        if not self.training:
            if z.shape[1] > chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            if self.activation_checkpointing:
                # torch.utils.checkpoint does not accept keyword arguments, wrap the
                # call in a lambda that closes over use_kernels.
                z = torch.utils.checkpoint.checkpoint(
                    partial(layer, use_kernels=use_kernels),
                    z,
                    pair_mask,
                    chunk_size_tri_attn,
                )
            else:
                z = layer(
                    z,
                    pair_mask,
                    chunk_size_tri_attn,
                    use_kernels=use_kernels,
                )
        return z


class PairformerNoSeqLayer(Module):
    """Pairformer module without sequence track."""

    def __init__(
        self,
        token_z: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)

        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )

        self.transition_z = Transition(token_z, token_z * 4)

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: int = None,
        *,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> Tensor:
        # Compute pairwise stack
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(
            z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(
            z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        z = z + self.transition_z(z)
        return z


class AdaLN(Module):
    """Algorithm 26"""

    def __init__(self, dim, dim_single_cond):
        super().__init__()
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
        self.s_scale = Linear(dim_single_cond, dim)
        self.s_bias = LinearNoBias(dim_single_cond, dim)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a


class ConditionedTransitionBlock(Module):
    """Algorithm 25"""

    def __init__(self, dim_single, dim_single_cond, expansion_factor=2):
        super().__init__()

        self.adaln = AdaLN(dim_single, dim_single_cond)

        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = Sequential(
            LinearNoBias(dim_single, dim_inner * 2),
            SwiGLU(),
        )
        self.a_to_b = LinearNoBias(dim_single, dim_inner)
        self.b_to_a = LinearNoBias(dim_inner, dim_single)

        output_projection_linear = Linear(dim_single_cond, dim_single)

        self.output_projection = nn.Sequential(output_projection_linear, nn.Sigmoid())

    def forward(
        self,
        a,  # Float['... d']
        s,
    ):  # -> Float['... d']:
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)
        a = self.output_projection(s) * self.b_to_a(b)

        return a


class DiffusionTransformer(Module):
    """Algorithm 23"""

    def __init__(
        self,
        depth,
        heads,
        dim=384,
        dim_single_cond=None,
        use_qk_norm=False,
        activation_checkpointing=False,
        post_layer_norm=False,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing

        dim_single_cond = default(dim_single_cond, dim)

        self.layers = ModuleList()
        for _ in range(depth):
            self.layers.append(
                DiffusionTransformerLayer(
                    heads,
                    dim=dim,
                    dim_single_cond=dim_single_cond,
                    post_layer_norm=post_layer_norm,
                    use_qk_norm=use_qk_norm,
                )
            )

    def forward(
        self,
        a,  # Float['bm n d'],
        s,  # Float['bm n ds'],
        bias=None,  # Float['b n n dp']
        mask=None,  # Bool['b n'] | None = None
        to_keys=None,
        multiplicity=1,
        use_uniform_bias=False,
    ):
        B, N, M, D = bias.shape
        L = len(self.layers)
        bias = bias.view(B, N, M, L, D // L) if not use_uniform_bias else bias

        for i, layer in enumerate(self.layers):
            bias_l = bias[:, :, :, i]

            if self.activation_checkpointing:
                a = torch.utils.checkpoint.checkpoint(
                    layer,
                    a,
                    s,
                    bias_l if not use_uniform_bias else bias,
                    mask,
                    to_keys,
                    multiplicity,
                )

            else:
                a = layer(
                    a,  # Float['bm n d'],
                    s,  # Float['bm n ds'],
                    bias_l if not use_uniform_bias else bias,  # Float['b n n dp']
                    mask,  # Bool['b n'] | None = None
                    to_keys,
                    multiplicity,
                )
        return a


class DiffusionTransformerLayer(Module):
    """Algorithm 23"""

    def __init__(
        self,
        heads,
        dim=384,
        dim_single_cond=None,
        post_layer_norm=False,
        use_qk_norm=False,
    ):
        super().__init__()

        dim_single_cond = default(dim_single_cond, dim)

        self.adaln = AdaLN(dim, dim_single_cond)

        self.pair_bias_attn = AttentionPairBias(
            c_s=dim,
            num_heads=heads,
            compute_pair_bias=False,
            use_qk_norm=use_qk_norm,
        )

        self.output_projection_linear = Linear(dim_single_cond, dim)

        self.output_projection = nn.Sequential(
            self.output_projection_linear, nn.Sigmoid()
        )
        self.transition = ConditionedTransitionBlock(
            dim_single=dim, dim_single_cond=dim_single_cond
        )

        if post_layer_norm:
            self.post_lnorm = nn.LayerNorm(dim)
        else:
            self.post_lnorm = nn.Identity()

    def forward(
        self,
        a,  # Float['bm n d'],
        s,  # Float['bm n ds'],
        bias=None,  # Float['b n n dp']
        mask=None,  # Bool['b n'] | None = None
        to_keys=None,
        multiplicity=1,
    ):
        b = self.adaln(a, s)

        k_in = b
        if to_keys is not None:
            k_in = to_keys(b)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)

        b = self.pair_bias_attn(
            s=b,
            z=bias,
            mask=mask,
            multiplicity=multiplicity,
            k_in=k_in,
        )
        b = self.output_projection(s) * b

        a = a + b
        a = a + self.transition(a, s)

        a = self.post_lnorm(a)
        return a


class AtomTransformer(Module):
    """Algorithm 7"""

    def __init__(
        self,
        attn_window_queries,
        attn_window_keys,
        **diffusion_transformer_kwargs,
    ):
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        self.diffusion_transformer = DiffusionTransformer(
            **diffusion_transformer_kwargs
        )

    def forward(
        self,
        q,  # Float['b m d'],
        c,  # Float['b m ds'],
        bias,  # Float['b m m dp']
        to_keys,
        mask,  # Bool['b m'] | None = None
        multiplicity=1,
    ):
        W = self.attn_window_queries
        H = self.attn_window_keys

        B, N, D = q.shape
        NW = N // W

        # reshape tokens
        q = q.view((B * NW, W, -1))
        c = c.view((B * NW, W, -1))
        mask = mask.view(B * NW, W)
        bias = bias.repeat_interleave(multiplicity, 0)
        bias = bias.view((bias.shape[0] * NW, W, H, -1))

        to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)

        # main transformer
        q = self.diffusion_transformer(
            a=q,
            s=c,
            bias=bias,
            mask=mask.float(),
            multiplicity=1,
            to_keys=to_keys_new,
        )

        q = q.view((B, NW * W, D))
        return q


class ConfidenceModule(Module):
    """Algorithm 31"""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        num_dist_bins=64,
        token_level_confidence=True,
        max_dist=22,
        add_s_to_z_prod=False,
        confidence_args: dict = None,
        return_latent_feats=False,
        conditioning_cutoff_min=None,
        conditioning_cutoff_max=None,
        bond_type_feature=False,
        add_z_input_to_z=None,  # old checkpoint compatibility
        add_s_input_to_s=None,  # old checkpoint compatibility
        **kwargs,
    ):
        super().__init__()
        self.max_num_atoms_per_token = 23
        self.no_update_s = pairformer_args.get("no_update_s", False)
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        self.token_level_confidence = token_level_confidence

        self.s_to_z = LinearNoBias(token_s, token_z)
        self.s_to_z_transpose = LinearNoBias(token_s, token_z)

        self.add_s_to_z_prod = add_s_to_z_prod
        if add_s_to_z_prod:
            self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_out = LinearNoBias(token_z, token_z)

        self.s_inputs_norm = nn.LayerNorm(token_s)
        if not self.no_update_s:
            self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        self.add_s_input_to_s = add_s_input_to_s
        if add_s_input_to_s:
            self.s_input_to_s = LinearNoBias(token_s, token_s)

        self.add_z_input_to_z = add_z_input_to_z
        if add_z_input_to_z:
            self.rel_pos = RelativePositionEncoder(token_z)
            self.token_bonds = nn.Linear(
                1,
                token_z,
                bias=False,
            )
            self.bond_type_feature = bond_type_feature
            if bond_type_feature:
                self.token_bonds_type = nn.Embedding(len(bond_types) + 1, token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=token_z,
                cutoff_min=conditioning_cutoff_min,
                cutoff_max=conditioning_cutoff_max,
            )

        self.pairformer_stack = PairformerModule(
            token_s,
            token_z,
            **pairformer_args,
        )
        self.return_latent_feats = return_latent_feats

        self.confidence_heads = ConfidenceHeads(
            token_s,
            token_z,
            token_level_confidence=token_level_confidence,
            **confidence_args,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s,  # Float['b n ts']
        z,  # Float['b n n tz']
        x_pred,  # Float['bm m 3']
        feats,
        pred_distogram_logits,
        multiplicity=1,
        s_diffusion=None,
        run_sequentially=False,
        use_kernels: bool = False,
    ):
        if run_sequentially and multiplicity > 1:
            assert z.shape[0] == 1, "Not supported with batch size > 1"
            out_dicts = []
            for sample_idx in range(multiplicity):
                out_dicts.append(  # noqa: PERF401
                    self.forward(
                        s_inputs,
                        s,
                        z,
                        x_pred[sample_idx : sample_idx + 1],
                        feats,
                        pred_distogram_logits,
                        multiplicity=1,
                        s_diffusion=(
                            s_diffusion[sample_idx : sample_idx + 1]
                            if s_diffusion is not None
                            else None
                        ),
                        run_sequentially=False,
                        use_kernels=use_kernels,
                    )
                )

            out_dict = {}
            for key in out_dicts[0]:
                if key != "pair_chains_iptm":
                    out_dict[key] = torch.cat([out[key] for out in out_dicts], dim=0)
                else:
                    pair_chains_iptm = {}
                    for chain_idx1 in out_dicts[0][key]:
                        chains_iptm = {}
                        for chain_idx2 in out_dicts[0][key][chain_idx1]:
                            chains_iptm[chain_idx2] = torch.cat(
                                [out[key][chain_idx1][chain_idx2] for out in out_dicts],
                                dim=0,
                            )
                        pair_chains_iptm[chain_idx1] = chains_iptm
                    out_dict[key] = pair_chains_iptm
            return out_dict

        s_inputs = self.s_inputs_norm(s_inputs)
        if not self.no_update_s:
            s = self.s_norm(s)

        if self.add_s_input_to_s:
            s = s + self.s_input_to_s(s_inputs)

        z = self.z_norm(z)

        s = s.repeat_interleave(multiplicity, 0)

        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )
        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
                * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        z = z.repeat_interleave(multiplicity, 0)
        s_inputs = s_inputs.repeat_interleave(multiplicity, 0)

        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        else:
            BM, N, _ = x_pred.shape
            B = BM // multiplicity
            mult = multiplicity
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)
        z = z + distogram

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s_t, z_t = self.pairformer_stack(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=use_kernels
        )

        # AF3 has residual connections, we remove them
        s = s_t
        z = z_t

        out_dict = {}

        if self.return_latent_feats:
            out_dict["s_conf"] = s
            out_dict["z_conf"] = z

        # confidence heads
        out_dict.update(
            self.confidence_heads(
                s=s,
                z=z,
                x_pred=x_pred,
                d=d,
                feats=feats,
                multiplicity=multiplicity,
                pred_distogram_logits=pred_distogram_logits,
            )
        )
        return out_dict


class DiffusionConditioning(Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
    ) -> None:
        super().__init__()

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=True,
        )

        self.atom_enc_proj_z = ModuleList()
        for _ in range(atom_encoder_depth):
            self.atom_enc_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_encoder_heads, bias=False),
                )
            )

        self.atom_dec_proj_z = ModuleList()
        for _ in range(atom_decoder_depth):
            self.atom_dec_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_decoder_heads, bias=False),
                )
            )

        self.token_trans_proj_z = ModuleList()
        for _ in range(token_transformer_depth):
            self.token_trans_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(token_z),
                    nn.Linear(token_z, token_transformer_heads, bias=False),
                )
            )

    def forward(
        self,
        s_trunk,  # Float['b n ts']
        z_trunk,  # Float['b n n tz']
        relative_position_encoding,  # Float['b n n tz']
        feats,
    ):
        z = self.pairwise_conditioner(
            z_trunk,
            relative_position_encoding,
        )

        q, c, p, to_keys = self.atom_encoder(
            feats=feats,
            s_trunk=s_trunk,  # Float['b n ts'],
            z=z,  # Float['b n n tz'],
        )

        atom_enc_bias = []
        for layer in self.atom_enc_proj_z:
            atom_enc_bias.append(layer(p))
        atom_enc_bias = torch.cat(atom_enc_bias, dim=-1)

        atom_dec_bias = []
        for layer in self.atom_dec_proj_z:
            atom_dec_bias.append(layer(p))
        atom_dec_bias = torch.cat(atom_dec_bias, dim=-1)

        token_trans_bias = []
        for layer in self.token_trans_proj_z:
            token_trans_bias.append(layer(z))
        token_trans_bias = torch.cat(token_trans_bias, dim=-1)

        return (
            q,
            c,
            to_keys,
            atom_enc_bias,
            atom_dec_bias,
            token_trans_bias,
        )


class ConfidenceHeads(Module):
    def __init__(
        self,
        token_s,
        token_z,
        num_plddt_bins=50,
        num_pde_bins=64,
        num_pae_bins=64,
        token_level_confidence=True,
        use_separate_heads: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.max_num_atoms_per_token = 23
        self.token_level_confidence = token_level_confidence
        self.use_separate_heads = use_separate_heads

        if self.use_separate_heads:
            self.to_pae_intra_logits = LinearNoBias(token_z, num_pae_bins)
            self.to_pae_inter_logits = LinearNoBias(token_z, num_pae_bins)
        else:
            self.to_pae_logits = LinearNoBias(token_z, num_pae_bins)

        if self.use_separate_heads:
            self.to_pde_intra_logits = LinearNoBias(token_z, num_pde_bins)
            self.to_pde_inter_logits = LinearNoBias(token_z, num_pde_bins)
        else:
            self.to_pde_logits = LinearNoBias(token_z, num_pde_bins)

        if self.token_level_confidence:
            self.to_plddt_logits = LinearNoBias(token_s, num_plddt_bins)
            self.to_resolved_logits = LinearNoBias(token_s, 2)
        else:
            self.to_plddt_logits = LinearNoBias(
                token_s, num_plddt_bins * self.max_num_atoms_per_token
            )
            self.to_resolved_logits = LinearNoBias(
                token_s, 2 * self.max_num_atoms_per_token
            )

    def forward(
        self,
        s,  # Float['b n ts']
        z,  # Float['b n n tz']
        x_pred,  # Float['bm m 3']
        d,
        feats,
        pred_distogram_logits,
        multiplicity=1,
    ):
        if self.use_separate_heads:
            asym_id_token = feats["asym_id"]
            is_same_chain = asym_id_token.unsqueeze(-1) == asym_id_token.unsqueeze(-2)
            is_different_chain = ~is_same_chain

        if self.use_separate_heads:
            pae_intra_logits = self.to_pae_intra_logits(z)
            pae_intra_logits = pae_intra_logits * is_same_chain.float().unsqueeze(-1)

            pae_inter_logits = self.to_pae_inter_logits(z)
            pae_inter_logits = pae_inter_logits * is_different_chain.float().unsqueeze(
                -1
            )

            pae_logits = pae_inter_logits + pae_intra_logits
        else:
            pae_logits = self.to_pae_logits(z)

        if self.use_separate_heads:
            pde_intra_logits = self.to_pde_intra_logits(z + z.transpose(1, 2))
            pde_intra_logits = pde_intra_logits * is_same_chain.float().unsqueeze(-1)

            pde_inter_logits = self.to_pde_inter_logits(z + z.transpose(1, 2))
            pde_inter_logits = pde_inter_logits * is_different_chain.float().unsqueeze(
                -1
            )

            pde_logits = pde_inter_logits + pde_intra_logits
        else:
            pde_logits = self.to_pde_logits(z + z.transpose(1, 2))
        resolved_logits = self.to_resolved_logits(s)
        plddt_logits = self.to_plddt_logits(s)

        ligand_weight = 20
        non_interface_weight = 1
        interface_weight = 10

        token_type = feats["mol_type"]
        token_type = token_type.repeat_interleave(multiplicity, 0)
        is_ligand_token = (token_type == chain_type_ids["NONPOLYMER"]).float()
        is_protein_token = (token_type == chain_type_ids["PROTEIN"]).float()

        if self.token_level_confidence:
            plddt = compute_aggregated_metric(plddt_logits)
            token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
            complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(
                dim=-1
            )

            is_contact = (d < 8).float()
            is_different_chain = (
                feats["asym_id"].unsqueeze(-1) != feats["asym_id"].unsqueeze(-2)
            ).float()
            is_different_chain = is_different_chain.repeat_interleave(multiplicity, 0)
            token_interface_mask = torch.max(
                is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
                dim=-1,
            ).values
            token_non_interface_mask = (1 - token_interface_mask) * (
                1 - is_ligand_token
            )
            iplddt_weight = (
                is_ligand_token * ligand_weight
                + token_interface_mask * interface_weight
                + token_non_interface_mask * non_interface_weight
            )
            complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(
                dim=-1
            ) / torch.sum(token_pad_mask * iplddt_weight, dim=-1)

        else:
            # token to atom conversion for resolved logits
            B, N, _ = resolved_logits.shape
            resolved_logits = resolved_logits.reshape(
                B, N, self.max_num_atoms_per_token, 2
            )

            arange_max_num_atoms = (
                torch.arange(self.max_num_atoms_per_token)
                .reshape(1, 1, -1)
                .to(resolved_logits.device)
            )
            max_num_atoms_mask = (
                feats["atom_to_token"].sum(1).unsqueeze(-1) > arange_max_num_atoms
            )
            resolved_logits = resolved_logits[:, max_num_atoms_mask.squeeze(0)]
            resolved_logits = pad(
                resolved_logits,
                (
                    0,
                    0,
                    0,
                    int(
                        feats["atom_pad_mask"].shape[1]
                        - feats["atom_pad_mask"].sum().item()
                    ),
                ),
                value=0,
            )
            plddt_logits = plddt_logits.reshape(B, N, self.max_num_atoms_per_token, -1)
            plddt_logits = plddt_logits[:, max_num_atoms_mask.squeeze(0)]
            plddt_logits = pad(
                plddt_logits,
                (
                    0,
                    0,
                    0,
                    int(
                        feats["atom_pad_mask"].shape[1]
                        - feats["atom_pad_mask"].sum().item()
                    ),
                ),
                value=0,
            )
            atom_pad_mask = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0)
            plddt = compute_aggregated_metric(plddt_logits)

            complex_plddt = (plddt * atom_pad_mask).sum(dim=-1) / atom_pad_mask.sum(
                dim=-1
            )
            token_type = feats["mol_type"].float()
            atom_to_token = feats["atom_to_token"].float()
            chain_id_token = feats["asym_id"].float()
            atom_type = torch.bmm(atom_to_token, token_type.unsqueeze(-1)).squeeze(-1)
            is_ligand_atom = (atom_type == chain_type_ids["NONPOLYMER"]).float()
            d_atom = torch.cdist(x_pred, x_pred)
            is_contact = (d_atom < 8).float()
            chain_id_atom = torch.bmm(
                atom_to_token, chain_id_token.unsqueeze(-1)
            ).squeeze(-1)
            is_different_chain = (
                chain_id_atom.unsqueeze(-1) != chain_id_atom.unsqueeze(-2)
            ).float()

            atom_interface_mask = torch.max(
                is_contact * is_different_chain * (1 - is_ligand_atom).unsqueeze(-1),
                dim=-1,
            ).values
            atom_non_interface_mask = (1 - atom_interface_mask) * (1 - is_ligand_atom)
            iplddt_weight = (
                is_ligand_atom * ligand_weight
                + atom_interface_mask * interface_weight
                + atom_non_interface_mask * non_interface_weight
            )

            complex_iplddt = (plddt * feats["atom_pad_mask"] * iplddt_weight).sum(
                dim=-1
            ) / torch.sum(feats["atom_pad_mask"] * iplddt_weight, dim=-1)

        # Compute the gPDE and giPDE
        pde = compute_aggregated_metric(pde_logits, end=32)
        pae = compute_aggregated_metric(pae_logits, end=32)
        pred_distogram_prob = nn.functional.softmax(
            pred_distogram_logits, dim=-1
        ).repeat_interleave(multiplicity, 0)
        contacts = torch.zeros((1, 1, 1, 64), dtype=pred_distogram_prob.dtype).to(
            pred_distogram_prob.device
        )
        contacts[:, :, :, :20] = 1.0
        prob_contact = (pred_distogram_prob * contacts).sum(-1)
        token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1)
            * token_pad_mask.unsqueeze(-2)
            * (
                1
                - torch.eye(
                    token_pad_mask.shape[1], device=token_pad_mask.device
                ).unsqueeze(0)
            )
        )
        # added code
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        selected_ligand_mask = is_ligand_token * (asym_id == 1).float()
        protein_pair_mask = token_pad_pair_mask * (
            is_protein_token.unsqueeze(-1) * is_protein_token.unsqueeze(-2)
        )
        protein_pair_mask_weighted = protein_pair_mask * prob_contact
        protein_pae_value_weighted = (pae * protein_pair_mask_weighted).sum(
            dim=(1, 2)
        ) / (protein_pair_mask_weighted.sum(dim=(1, 2)) + 1e-5)
        protein_pae_value_unweighted = (pae * protein_pair_mask).sum(dim=(1, 2)) / (
            protein_pair_mask.sum(dim=(1, 2)) + 1e-5
        )

        ligand_pair_mask = token_pad_pair_mask * (
            selected_ligand_mask.unsqueeze(-1) * selected_ligand_mask.unsqueeze(-2)
        )

        ligand_pair_mask_weighted = ligand_pair_mask * prob_contact
        ligand_pae_value_weighted = (pae * ligand_pair_mask_weighted).sum(
            dim=(1, 2)
        ) / (ligand_pair_mask_weighted.sum(dim=(1, 2)) + 1e-5)
        ligand_pae_value_unweighted = (pae * ligand_pair_mask).sum(dim=(1, 2)) / (
            ligand_pair_mask.sum(dim=(1, 2)) + 1e-5
        )

        interface_protein_mask = torch.max(
            (d < 8).float()
            * is_protein_token.unsqueeze(-1)
            * selected_ligand_mask.unsqueeze(-2),
            dim=-1,
        ).values
        protein_ligand_interface_pair_mask = (
            token_pad_pair_mask
            * interface_protein_mask.unsqueeze(-1)
            * selected_ligand_mask.unsqueeze(-2)
            * (d < 8).float()
        )
        protein_ligand_interface_pair_mask_weighted = (
            protein_ligand_interface_pair_mask * prob_contact
        )
        interface_pae_value_weighted = (
            pae * protein_ligand_interface_pair_mask_weighted
        ).sum(dim=(1, 2)) / (
            protein_ligand_interface_pair_mask_weighted.sum(dim=(1, 2)) + 1e-5
        )
        interface_pae_value_unweighted = (pae * protein_ligand_interface_pair_mask).sum(
            dim=(1, 2)
        ) / (protein_ligand_interface_pair_mask.sum(dim=(1, 2)) + 1e-5)
        # finish
        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(dim=(1, 2)) / token_pair_mask.sum(
            dim=(1, 2)
        )
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        token_interface_pair_mask = token_pair_mask * (
            asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        )
        complex_ipde = (pde * token_interface_pair_mask).sum(dim=(1, 2)) / (
            token_interface_pair_mask.sum(dim=(1, 2)) + 1e-5
        )

        # Compute design metric PDE
        is_chain_design_token = (
            feats["chain_design_mask"].float()
            if feats["design_mask"].sum() > 0
            else feats["design_mask"].float()
        )
        is_target_token = (
            1 - feats["chain_design_mask"].float()
            if feats["design_mask"].sum() > 0
            else feats["design_mask"].float()
        )
        target_designchain_inter_mask = token_pad_pair_mask * (
            is_chain_design_token[:, :, None] * is_target_token[:, None, :]
            + is_target_token[:, :, None] * is_chain_design_token[:, None, :]
        )
        interaction_pae = (pae * target_designchain_inter_mask).sum(dim=(1, 2)) / (
            target_designchain_inter_mask.sum(dim=(1, 2)) + 1e-5
        )
        min_interaction_pae = torch.min(
            pae + (1 - target_designchain_inter_mask) * 100000
        )[None]

        is_design_token = feats["design_mask"].float()
        target_design_inter_mask = token_pad_pair_mask * (
            is_design_token[:, :, None] * is_target_token[:, None, :]
            + is_target_token[:, :, None] * is_design_token[:, None, :]
        )
        interaction_pae = (pae * target_design_inter_mask).sum(dim=(1, 2)) / (
            target_design_inter_mask.sum(dim=(1, 2)) + 1e-5
        )
        min_design_to_target_pae = torch.min(
            pae + (1 - target_design_inter_mask) * 100000
        )[None]

        out_dict = dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
            interaction_pae=interaction_pae,
            min_design_to_target_pae=min_design_to_target_pae
            if feats["design_mask"].sum() > 0
            else torch.nan,
            min_interaction_pae=min_interaction_pae,
        )

        # weighted
        out_dict["protein_pae_value_weighted"] = protein_pae_value_weighted
        out_dict["ligand_pae_value_weighted"] = ligand_pae_value_weighted
        out_dict["interface_pae_value_weighted"] = interface_pae_value_weighted
        # unweighted
        out_dict["protein_pae_value_unweighted"] = protein_pae_value_unweighted
        out_dict["ligand_pae_value_unweighted"] = ligand_pae_value_unweighted
        out_dict["interface_pae_value_unweighted"] = interface_pae_value_unweighted

        out_dict["pae_logits"] = pae_logits
        out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)

        try:
            (
                ptm,
                iptm,
                ligand_iptm,
                protein_iptm,
                pair_chains_iptm,
                design_to_target_iptm,
                design_iptm,
                design_iiptm,
                target_ptm,
                design_ptm,
            ) = compute_ptms(pae_logits, x_pred, feats, multiplicity)
            out_dict["ptm"] = ptm
            out_dict["iptm"] = iptm
            out_dict["ligand_iptm"] = ligand_iptm
            out_dict["protein_iptm"] = protein_iptm
            out_dict["pair_chains_iptm"] = pair_chains_iptm
            out_dict["design_to_target_iptm"] = design_to_target_iptm
            out_dict["design_iptm"] = design_iptm
            out_dict["design_iiptm"] = design_iiptm
            out_dict["target_ptm"] = target_ptm
            out_dict["design_ptm"] = design_ptm
        except Exception as e:
            print(f"Error in compute_ptms: {e}")
            out_dict["ptm"] = torch.zeros_like(complex_plddt)
            out_dict["iptm"] = torch.zeros_like(complex_plddt)
            out_dict["ligand_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["protein_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["pair_chains_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_to_target_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_iiptm"] = torch.zeros_like(complex_plddt)
            out_dict["target_ptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_ptm"] = torch.zeros_like(complex_plddt)

        return out_dict


class FourierEmbedding(Module):
    """Algorithm 22."""

    def __init__(self, dim):
        super().__init__()
        self.proj = Module()
        self.proj.register_buffer("weight", torch.randn(dim, 1))
        self.proj.register_buffer("bias", torch.randn(dim))

    def forward(
        self,
        times,  # Float[' b'],
    ):  # -> Float['b d']:
        times = rearrange(times, "b -> b 1")
        rand_proj = torch.addmm(self.proj.bias, times, self.proj.weight.t())
        return torch.cos(2 * pi * rand_proj)


class RelativePositionEncoder(Module):
    """Algorithm 3."""

    def __init__(self, token_z, r_max=32, s_max=2):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.linear_layer = LinearNoBias(4 * (r_max + 1) + 2 * (s_max + 1) + 1, token_z)

    def forward(self, feats):
        b_same_chain = torch.eq(
            feats["feature_asym_id"][:, :, None], feats["feature_asym_id"][:, None, :]
        )
        b_same_residue = torch.eq(
            feats["feature_residue_index"][:, :, None],
            feats["feature_residue_index"][:, None, :],
        )
        b_same_entity = torch.eq(
            feats["entity_id"][:, :, None], feats["entity_id"][:, None, :]
        )

        d_residue = (
            feats["feature_residue_index"][:, :, None]
            - feats["feature_residue_index"][:, None, :]
        )

        if torch.any(feats["cyclic"] > 0):
            period = torch.where(
                feats["cyclic"] > 0,
                feats["cyclic"],
                torch.zeros_like(feats["cyclic"]) + 10000,
            ).unsqueeze(1)
            d_residue = (d_residue - period * torch.round(d_residue / period)).long()

        d_residue = torch.clip(
            d_residue + self.r_max,
            0,
            2 * self.r_max,
        )
        d_residue = torch.where(
            b_same_chain,
            d_residue,
            torch.zeros_like(d_residue) + 2 * self.r_max + 1,
        )

        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        d_token = torch.clip(
            feats["token_index"][:, :, None]
            - feats["token_index"][:, None, :]
            + self.r_max,
            0,
            2 * self.r_max,
        )
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token = one_hot(d_token, 2 * self.r_max + 2)

        d_chain = torch.clip(
            feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + self.s_max,
            0,
            2 * self.s_max,
        )

        d_chain = torch.where(
            (~b_same_entity),
            torch.zeros_like(d_chain) + 2 * self.s_max + 1,
            d_chain,
        )
        # Note: added  | (~b_same_entity) based on observation of ProteinX manuscript
        a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)

        p = self.linear_layer(
            torch.cat(
                [
                    a_rel_pos.float(),
                    a_rel_token.float(),
                    b_same_entity.unsqueeze(-1).float(),
                    a_rel_chain.float(),
                ],
                dim=-1,
            )
        )
        return p


class SingleConditioning(Module):
    """Algorithm 21."""

    def __init__(
        self,
        sigma_data: float,
        tfmr_s: int = 768,
        token_s: int = 384,
        dim_fourier: int = 256,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
        eps: float = 1e-20,
        disable_times: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.sigma_data = sigma_data
        self.disable_times = disable_times

        self.norm_single = nn.LayerNorm(token_s * 2)

        if tfmr_s != token_s * 2:
            self.token_s_to_tfmr_s = nn.Linear(token_s * 2, tfmr_s)
        else:
            self.token_s_to_tfmr_s = None

        self.single_embed = nn.Linear(tfmr_s, tfmr_s)
        if not self.disable_times:
            self.fourier_embed = FourierEmbedding(dim_fourier)
            self.norm_fourier = nn.LayerNorm(dim_fourier)
            self.fourier_to_single = LinearNoBias(dim_fourier, tfmr_s)

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=tfmr_s, hidden=transition_expansion_factor * tfmr_s
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        times,  # Float[' b'],
        s_trunk,  # Float['b n ts'],
        s_inputs,  # Float['b n ts'],
    ):  # -> Float['b n 2ts']:
        s = torch.cat((s_trunk, s_inputs), dim=-1)
        s = self.norm_single(s)

        if self.token_s_to_tfmr_s is not None:
            s = self.token_s_to_tfmr_s(s)

        s = self.single_embed(s)
        if not self.disable_times:
            fourier_embed = self.fourier_embed(
                times
            )  # note: sigma rescaling done in diffusion module
            normed_fourier = self.norm_fourier(fourier_embed)
            fourier_to_single = self.fourier_to_single(normed_fourier)

            s = rearrange(fourier_to_single, "b d -> b 1 d") + s

        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier if not self.disable_times else None


class PairwiseConditioning(Module):
    """Algorithm 21."""

    def __init__(
        self,
        token_z,
        dim_token_rel_pos_feats,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_token_rel_pos_feats),
            LinearNoBias(token_z + dim_token_rel_pos_feats, token_z),
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=token_z, hidden=transition_expansion_factor * token_z
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        z_trunk,  # Float['b n n tz'],
        token_rel_pos_feats,  # Float['b n n 3'],
    ):  # -> Float['b n n tz']:
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)

        for transition in self.transitions:
            z = transition(z) + z

        return z


class CoordinateConditioning(Module):
    def __init__(
        self,
        sigma_data: float,
        atom_s,
        token_s,
        num_heads,
        tfmr_s: int = 768,
        dim_fourier: int = 256,
        atom_feature_dim: int = 132,
        structure_prediction=True,
        disable_times: bool = False,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.single_embed = LinearNoBias(token_s * 2, tfmr_s)

        self.disable_times = disable_times
        if not self.disable_times:
            self.fourier_embed = FourierEmbedding(dim_fourier)
            self.norm_fourier = nn.LayerNorm(dim_fourier)
            self.fourier_to_single = LinearNoBias(dim_fourier, tfmr_s)

        self.embed_atom_features = Linear(atom_feature_dim, num_heads)
        self.embed_atompair_ref_coord = LinearNoBias(3, num_heads)
        self.embed_atompair_ref_dist = LinearNoBias(1, num_heads)
        self.embed_atompair_mask = LinearNoBias(1, num_heads)

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(
                nn.LayerNorm(tfmr_s), LinearNoBias(tfmr_s, 1)
            )

    def forward(self, s_trunk, s_inputs, times, feats, atom_coords_noisy):
        s = torch.cat((s_trunk, s_inputs), dim=-1)
        s = self.single_embed(s)
        if not self.disable_times:
            fourier_embed = self.fourier_embed(
                times
            )  # note: sigma rescaling done in diffusion module
            normed_fourier = self.norm_fourier(fourier_embed)
            fourier_to_single = self.fourier_to_single(normed_fourier)
            s = rearrange(fourier_to_single, "b d -> b 1 d") + s

        with torch.autocast("cuda", enabled=False):
            atom_mask = feats["atom_pad_mask"].bool()
            atom_ref_pos = feats["ref_pos"]
            atom_uid = feats["ref_space_uid"]

            atom_feats = [
                atom_ref_pos,
                feats["ref_charge"].unsqueeze(-1),
                feats["ref_element"],
            ]

            atom_feats = torch.cat(atom_feats, dim=-1)
            c = self.embed_atom_features(atom_feats)
            B, N = atom_coords_noisy.shape[:2]

            s_to_c = self.s_to_c_trans(s.float())
            s_to_c = torch.bmm(
                feats["atom_to_token"].float().repeat_interleave(B, 0), s_to_c
            )
            c = c + s_to_c.to(c)

            atom_mask = atom_mask.repeat_interleave(B, 0)
            atom_uid = atom_uid.repeat_interleave(B, 0)

            d = atom_coords_noisy.view(B, N, 1, 3) - atom_coords_noisy.view(B, 1, N, 3)
            d_norm = torch.sum(d * d, dim=-1, keepdim=True)
            d_norm = 1 / (1 + d_norm)
            atom_mask_queries = atom_mask.view(B, N, 1)
            atom_mask_keys = atom_mask.view(B, 1, N)
            atom_uid_queries = atom_uid.view(B, N, 1)
            atom_uid_keys = atom_uid.view(B, 1, N)
            v = (
                (
                    atom_mask_queries
                    & atom_mask_keys
                    & (atom_uid_queries == atom_uid_keys)
                )
                .float()
                .unsqueeze(-1)
            )

            p = (
                self.embed_atompair_ref_coord(d) * v
                + self.embed_atompair_ref_dist(d_norm) * v
                + self.embed_atompair_mask(v) * v
            )
            p = p + c.view(B, 1, N, -1) + c.view(B, N, 1, -1)

            return p.sum(dim=0, keepdim=True)


class DistanceTokenEncoder(Module):
    def __init__(
        self,
        distance_gaussian_dim: int,
        token_z: int,
        out_dim: int,
    ):
        super().__init__()
        self.distance_gaussian_smearing = GaussianSmearing(
            start=0.0, stop=2.0, num_gaussians=distance_gaussian_dim
        )
        input_dim = distance_gaussian_dim + 1 + token_z
        self.distance_token_bias_trans = Transition(
            dim=input_dim, hidden=token_z, out_dim=out_dim
        )

    def forward(
        self,
        relative_position_encoding,
        feats,
    ):
        B, N, _, _ = relative_position_encoding.shape

        token_to_bb4_atoms = feats["token_to_bb4_atoms"]
        r = feats["coords"]

        r_repr = torch.bmm(
            token_to_bb4_atoms.float().view(B, N * 4, -1), r.view(B, -1, 3)
        )
        r_repr = r_repr.reshape(B, N, 4, 3).permute(0, 2, 1, 3)
        d = (r_repr.unsqueeze(-2) - r_repr.unsqueeze(-3)).norm(dim=-1).unsqueeze(-1)
        distance_gaussian = self.distance_gaussian_smearing(d)

        relative_position_encoding = relative_position_encoding.view(
            B, 1, N, N, -1
        ).expand(-1, 4, -1, -1, -1)
        distance_token_bias_input = torch.cat(
            (
                distance_gaussian,
                d,
                relative_position_encoding,
            ),
            dim=-1,
        )
        distance_token_bias = (
            self.distance_token_bias_trans(distance_token_bias_input)
            .permute(0, 2, 3, 4, 1)
            .reshape(B, N, N, -1)
        )
        return distance_token_bias


def get_indexing_matrix(K, W, H, device):
    assert W % 2 == 0
    assert H % (W // 2) == 0

    h = H // (W // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1
    )
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()


def single_to_keys(single, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )  # j = 2K, i = W//2, k = h * K


class AtomEncoder(Module):
    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim,
        structure_prediction=True,
    ):
        super().__init__()

        self.embed_atom_features = Linear(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(
                nn.LayerNorm(token_s), LinearNoBias(token_s, atom_s)
            )

            self.z_to_p_trans = nn.Sequential(
                nn.LayerNorm(token_z), LinearNoBias(token_z, atom_z)
            )

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )

    def forward(
        self,
        feats,
        s_trunk=None,  # Float['bm n ts'],
        z=None,  # Float['bm n n tz'],
    ):
        with torch.autocast("cuda", enabled=False):
            B, N, _ = feats["ref_pos"].shape
            atom_mask = feats["atom_pad_mask"].bool()  # Bool['b m'],

            atom_ref_pos = feats["ref_pos"]  # Float['b m 3'],
            atom_uid = feats["ref_space_uid"]  # Long['b m'],

            atom_feats = [
                atom_ref_pos,
                feats["ref_charge"].unsqueeze(-1),
                feats["ref_element"],
                feats["ref_atom_name_chars"].reshape(B, N, 4 * 64),
            ]

            atom_feats = torch.cat(atom_feats, dim=-1)

            c = self.embed_atom_features(atom_feats)

        # note we are already creating the windows to make it more efficient
        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        B, N = c.shape[:2]
        K = N // W
        keys_indexing_matrix = get_indexing_matrix(K, W, H, c.device)
        to_keys = partial(
            single_to_keys, indexing_matrix=keys_indexing_matrix, W=W, H=H
        )

        atom_ref_pos_queries = atom_ref_pos.view(B, K, W, 1, 3)
        atom_ref_pos_keys = to_keys(atom_ref_pos).view(B, K, 1, H, 3)

        d = atom_ref_pos_keys - atom_ref_pos_queries  # Float['b k w h 3']
        d_norm = torch.sum(d * d, dim=-1, keepdim=True)  # Float['b k w h 1']
        d_norm = 1 / (1 + d_norm)  # AF3 feeds in the reciprocal of the distance norm

        atom_mask_queries = atom_mask.view(B, K, W, 1)
        atom_mask_keys = (
            to_keys(atom_mask.unsqueeze(-1).float()).view(B, K, 1, H).bool()
        )
        atom_uid_queries = atom_uid.view(B, K, W, 1)
        atom_uid_keys = to_keys(atom_uid.unsqueeze(-1).float()).view(B, K, 1, H).long()
        v = (
            (atom_mask_queries & atom_mask_keys & (atom_uid_queries == atom_uid_keys))
            .float()
            .unsqueeze(-1)
        )  # Bool['b k w h 1']

        p = self.embed_atompair_ref_pos(d) * v
        p = p + self.embed_atompair_ref_dist(d_norm) * v
        p = p + self.embed_atompair_mask(v) * v

        q = c

        if self.structure_prediction:
            # run only in structure model not in initial encoding
            atom_to_token = feats["atom_to_token"].float()  # Long['b m n'],

            s_to_c = self.s_to_c_trans(s_trunk.float())
            s_to_c = torch.bmm(atom_to_token, s_to_c)
            c = c + s_to_c.to(c)

            atom_to_token_queries = atom_to_token.view(B, K, W, atom_to_token.shape[-1])
            atom_to_token_keys = to_keys(atom_to_token)
            z_to_p = self.z_to_p_trans(z.float())
            z_to_p = torch.einsum(
                "bijd,bwki,bwlj->bwkld",
                z_to_p,
                atom_to_token_queries,
                atom_to_token_keys,
            )
            p = p + z_to_p.to(p)

        p = p + self.c_to_p_trans_q(c.view(B, K, W, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(to_keys(c).view(B, K, 1, H, c.shape[-1]))
        p = p + self.p_mlp(p)

        return q, c, p, to_keys


class AtomAttentionEncoder(Module):
    def __init__(
        self,
        atom_s,
        token_s,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        structure_prediction=True,
        activation_checkpointing=False,
        gaussian_random_3d_encoding_dim=0,
        transformer_post_layer_norm=False,
        tfmr_s=None,
        use_qk_norm=False,
    ):
        super().__init__()

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.gaussian_random_3d_encoding_dim = gaussian_random_3d_encoding_dim
            if gaussian_random_3d_encoding_dim > 0:
                self.encoding_3d = GaussianRandom3DEncodings(
                    gaussian_random_3d_encoding_dim
                )
            r_input_size = 3 + gaussian_random_3d_encoding_dim
            self.r_to_q_trans = LinearNoBias(r_input_size, atom_s)

        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            activation_checkpointing=activation_checkpointing,
            post_layer_norm=transformer_post_layer_norm,
            use_qk_norm=use_qk_norm,
        )

        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, tfmr_s if structure_prediction else token_s),
            nn.ReLU(),
        )
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

    def forward(
        self,
        feats,
        q,
        c,
        atom_enc_bias,
        to_keys,
        r=None,  # Float['bm m 3'],
        multiplicity=1,
    ):
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"].bool()  # Bool['b m'],

        if self.structure_prediction:
            # only here the multiplicity kicks in because we use the different positions r
            q = q.repeat_interleave(multiplicity, 0)

            r_input = r
            if self.gaussian_random_3d_encoding_dim > 0:
                r_input = torch.cat([r_input, self.encoding_3d(r)], dim=-1)

            r_to_q = self.r_to_q_trans(r_input)
            q = q + r_to_q

        c = c.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_enc_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        with torch.autocast("cuda", enabled=False):
            q_to_a = self.atom_to_token_trans(q).float()
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
            atom_to_token_mean = atom_to_token / (
                atom_to_token.sum(dim=1, keepdim=True) + 1e-6
            )
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        a = a.to(q)

        return a, q, c, to_keys


class AtomAttentionDecoder(Module):
    """Algorithm 6."""

    def __init__(
        self,
        atom_s,
        tfmr_s,
        attn_window_queries,
        attn_window_keys,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        activation_checkpointing=False,
        transformer_post_layer_norm=False,
        predict_res_type=False,
        use_qk_norm=False,
    ):
        super().__init__()
        self.predict_res_type = predict_res_type
        self.a_to_q_trans = LinearNoBias(tfmr_s, atom_s)

        self.atom_decoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            post_layer_norm=transformer_post_layer_norm,
            use_qk_norm=use_qk_norm,
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )

        if predict_res_type:
            self.res_type_predictor = Linear(atom_s, len(tokens))

    def forward(
        self,
        a,  # Float['bm n 2ts'],
        q,  # Float['bm m as'],
        c,  # Float['bm m as'],
        atom_dec_bias,  # Float['bm m m az'],
        feats,
        to_keys,
        multiplicity=1,
    ):
        with torch.autocast("cuda", enabled=False):
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

            a_to_q = self.a_to_q_trans(a.float())
            a_to_q = torch.bmm(atom_to_token, a_to_q)

        q = q + a_to_q.to(q)
        atom_mask = feats["atom_pad_mask"]  # Bool['b m'],
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_dec_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        res_type = None
        s_feat = None
        if self.predict_res_type:
            idx = torch.argmax(feats["atom_to_token"].int(), dim=-1)
            mask = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0)
            idx = idx.repeat_interleave(multiplicity, 0)
            src = q * mask[:, :, None]
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, q.size(-1))
            s_feat = torch.zeros(
                (q.shape[0], feats["res_type"].shape[1], q.shape[-1]),
                device=idx_expanded.device,
            )
            s_feat.scatter_add_(dim=1, index=idx_expanded, src=src)

        if self.predict_res_type and s_feat is not None:
            res_type = self.res_type_predictor(s_feat)


        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update, res_type


class GaussianSmearing(Module):
    """Gaussian smearing."""

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.num_gaussians = num_gaussians
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        shape = dist.shape
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2)).reshape(
            *shape, self.num_gaussians
        )


class AffinityModule(Module):
    """Algorithm 31"""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        transformer_args: dict,
        num_dist_bins=64,
        max_dist=22,
        use_cross_transformer: bool = False,
        groups: dict = {},
    ):
        super().__init__()
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)

        self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
        self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = LinearNoBias(token_z, token_z)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
        )

        self.pairformer_stack = PairformerNoSeqModule(token_z, **pairformer_args)
        self.affinity_heads = AffinityHeadsTransformer(
            token_z,
            transformer_args["token_s"],
            transformer_args["num_blocks"],
            transformer_args["num_heads"],
            transformer_args["activation_checkpointing"],
            False,
            groups=groups,
        )

    def forward(
        self,
        s_inputs,
        z,
        x_pred,
        feats,
        multiplicity=1,
        use_kernels: bool = False,
    ):
        z = self.z_linear(self.z_norm(z))
        z = z.repeat_interleave(multiplicity, 0)

        z = (
            z
            + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )

        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        else:
            BM, N, _ = x_pred.shape
            B = BM // multiplicity
            mult = multiplicity
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)

        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        pad_token_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        rec_mask = (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0)
        rec_mask = rec_mask * pad_token_mask
        lig_mask = (
            feats["affinity_token_mask"]
            .repeat_interleave(multiplicity, 0)
            .to(torch.bool)
        )
        lig_mask = lig_mask * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + lig_mask[:, :, None] * lig_mask[:, None, :]
        )
        z = self.pairformer_stack(
            z,
            pair_mask=cross_pair_mask,
            use_kernels=use_kernels,
        )

        out_dict = {}

        # affinity heads
        out_dict.update(
            self.affinity_heads(z=z, feats=feats, multiplicity=multiplicity)
        )

        return out_dict


class AffinityHeadsTransformer(Module):
    def __init__(
        self,
        token_z,
        input_token_s,
        num_blocks,
        num_heads,
        activation_checkpointing,
        use_cross_transformer,
        groups={},
    ):
        super().__init__()
        self.affinity_out_mlp = nn.Sequential(
            nn.Linear(token_z, token_z),
            nn.ReLU(),
            nn.Linear(token_z, input_token_s),
            nn.ReLU(),
        )

        self.to_affinity_pred_value = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )

        self.to_affinity_pred_score = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )
        self.to_affinity_logits_binary = nn.Linear(1, 1)

    def forward(
        self,
        z,
        feats,
        multiplicity=1,
    ):
        pad_token_mask = (
            feats["token_pad_mask"].repeat_interleave(multiplicity, 0).unsqueeze(-1)
        )
        rec_mask = (
            (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0).unsqueeze(-1)
        )
        rec_mask = rec_mask * pad_token_mask
        lig_mask = (
            feats["affinity_token_mask"]
            .repeat_interleave(multiplicity, 0)
            .to(torch.bool)
            .unsqueeze(-1)
        ) * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + (lig_mask[:, :, None] * lig_mask[:, None, :])
        ) * (
            1
            - torch.eye(lig_mask.shape[1], device=lig_mask.device)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        g = torch.sum(z * cross_pair_mask, dim=(1, 2)) / (
            torch.sum(cross_pair_mask, dim=(1, 2)) + 1e-7
        )

        g = self.affinity_out_mlp(g)

        affinity_pred_value = self.to_affinity_pred_value(g).reshape(-1, 1)
        affinity_pred_score = self.to_affinity_pred_score(g).reshape(-1, 1)
        affinity_logits_binary = self.to_affinity_logits_binary(
            affinity_pred_score
        ).reshape(-1, 1)
        out_dict = {
            "affinity_pred_value": affinity_pred_value,
            "affinity_logits_binary": affinity_logits_binary,
        }
        return out_dict


class MLPAttnGNN(Module):
    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        hidden_dim: int,
        dropout: float,
        softmax_dropout: float,
        transformation_scale_factor: float,
        num_heads: int = 4,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.transformation_scale_factor = transformation_scale_factor
        self.num_heads = num_heads

        self.attn_weight_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_heads),
        )

        self.attn_value_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
        )

        self.attn_output_linear = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_heads, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

        self.attn_FFN = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

        self.edge_FFN = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.pair_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.pair_dim),
        )

    def forward(self, s, z, edge_idx):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        z = z + self.edge_FFN(torch.cat([s[src_idx], s[dst_idx], z], dim=1))

        attn_weight = self.attn_weight_mlp(
            torch.cat([s[dst_idx], s[src_idx], z], dim=1)
        )
        attn_value = self.attn_value_mlp(torch.cat([s[src_idx], z], dim=1))

        if self.training & (self.softmax_dropout > 0):
            attn_weight = softmax_dropout(
                attn_weight, softmax_dropout=self.softmax_dropout, dst_idx=dst_idx
            )

        attn_weight = scatter_softmax(attn_weight, index=dst_idx, dim=0)
        attn_output = attn_weight.unsqueeze(-1) * attn_value.unsqueeze(1)
        attn_output = scatter_sum(attn_output, index=dst_idx, dim=0).flatten(
            start_dim=1
        )

        s = s + self.attn_output_linear(attn_output)
        s = s + self.attn_FFN(s)

        return s, z


class MLPAttnGNNDecoder(Module):
    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        hidden_dim: int,
        dropout: float,
        softmax_dropout: float,
        transformation_scale_factor: float,
        num_heads: int = 4,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.transformation_scale_factor = transformation_scale_factor
        self.num_heads = num_heads

        self.attn_weight_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_heads),
        )

        self.attn_value_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
        )

        self.attn_output_linear = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_heads, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

        self.attn_FFN = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

    def forward(self, s, z, edge_idx):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        attn_weight = self.attn_weight_mlp(torch.cat([s[dst_idx], z], dim=1))
        attn_value = self.attn_value_mlp(z)

        if self.training & (self.softmax_dropout > 0):
            attn_weight = softmax_dropout(
                attn_weight, softmax_dropout=self.softmax_dropout, dst_idx=dst_idx
            )

        attn_weight = scatter_softmax(attn_weight, index=dst_idx, dim=0)
        attn_output = attn_weight.unsqueeze(-1) * attn_value.unsqueeze(1)
        attn_output = scatter_sum(attn_output, index=dst_idx, dim=0).flatten(
            start_dim=1
        )

        s = s + self.attn_output_linear(attn_output)
        s = s + self.attn_FFN(s)

        return s

    def sample(self, s, z):
        dst_idx = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        attn_weight = self.attn_weight_mlp(torch.cat([s[dst_idx], z], dim=1))
        attn_value = self.attn_value_mlp(z)

        if self.training & (self.softmax_dropout > 0):
            attn_weight = softmax_dropout(
                attn_weight, softmax_dropout=self.softmax_dropout, dst_idx=dst_idx
            )

        attn_weight = scatter_softmax(attn_weight, index=dst_idx, dim=0)
        attn_output = attn_weight.unsqueeze(-1) * attn_value.unsqueeze(1)
        attn_output = scatter_sum(attn_output, index=dst_idx, dim=0).flatten(
            start_dim=1
        )

        s = s + self.attn_output_linear(attn_output)
        s = s + self.attn_FFN(s)

        return s


class InverseFoldingEncoder(Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        node_dim: int = 128,
        pair_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
        num_encoder_layers: int = 3,
        transformation_scale_factor: float = 1.0,
        inverse_fold_noise: float = 0.3,
        topk: int = 30,
        num_heads: int = 4,
        enable_input_embedder: bool = False,
        **kwargs, # old checkpoint compatibility
    ):
        """Initialize the Inverse Folding Encoder."""
        super().__init__()

        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.num_encoder_layers = num_encoder_layers
        self.transformation_scale_factor = transformation_scale_factor
        self.inverse_fold_noise = inverse_fold_noise
        self.topk = topk
        self.num_heads = num_heads
        self.r_max = 32
        self.enable_input_embedder = enable_input_embedder
        self.edge_input_dim = (
            256 + 2 * self.r_max + 2 + 1 + 1 + len(bond_types) + 1 + 1
        )

        self.linear_token_to_node = nn.Linear(
            135 if not self.enable_input_embedder else self.token_s, self.node_dim
        )
        self.linear_token_to_pair = nn.Linear(self.edge_input_dim, self.pair_dim)

        self.encoder_layers = ModuleList()
        for i in range(self.num_encoder_layers):
            layer = MLPAttnGNN(
                node_dim=node_dim,
                pair_dim=pair_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
                transformation_scale_factor=transformation_scale_factor,
                num_heads=num_heads,
            )
            self.encoder_layers.append(layer)

        self.distance_gaussian_smearing = GaussianSmearing(
            start=0.0,
            stop=20.0,
            num_gaussians=16,
        )

    @torch.no_grad()
    def init_knn_graph(self, feats: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        valid_mask = (
            feats["token_resolved_mask"].bool() & feats["token_pad_mask"].bool()
        )
        B, N = valid_mask.shape
        valid_mask_pair = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        token_index = (valid_mask.flatten().cumsum(dim=0) - 1).view(B, N)
        topk = min(self.topk, N)

        coords = feats["center_coords"]
        dists = torch.cdist(coords, coords)
        dists = dists.masked_fill(~valid_mask_pair, float("inf"))
        src_idx = torch.topk(dists, topk, largest=False).indices
        dst_idx = torch.arange(N, device=src_idx.device)[None, :, None].expand(
            B, -1, topk
        )

        assert feats["token_bonds"].shape[-1] == 1
        token_bonds = feats["token_bonds"][..., 0]
        type_bonds = feats["type_bonds"]
        token_bonds = torch.gather(token_bonds, 2, src_idx)
        type_bonds = torch.gather(type_bonds, 2, src_idx)

        src_idx, dst_idx = src_idx.flatten(start_dim=1), dst_idx.flatten(start_dim=1)
        token_bonds, type_bonds = token_bonds.flatten(), type_bonds.flatten()

        src_valid_mask = torch.gather(valid_mask, 1, src_idx).flatten()
        dst_valid_mask = torch.gather(valid_mask, 1, dst_idx).flatten()
        edge_valid_mask = src_valid_mask & dst_valid_mask

        src_idx = torch.gather(token_index, 1, src_idx).flatten()[edge_valid_mask]
        dst_idx = torch.gather(token_index, 1, dst_idx).flatten()[edge_valid_mask]
        edge_idx = torch.stack([src_idx, dst_idx], dim=0)

        token_bonds, type_bonds = (
            token_bonds[edge_valid_mask],
            type_bonds[edge_valid_mask],
        )
        type_bonds = one_hot(type_bonds, num_classes=len(bond_types) + 1)
        bond = torch.cat([token_bonds[..., None], type_bonds], dim=-1)

        return edge_idx, valid_mask, bond

    @torch.no_grad()
    def extract_geo_feat(
        self, feats: Dict[str, Tensor], edge_idx: Tensor, valid_mask: Tensor
    ) -> Tensor:
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        token_to_bb4_atoms = feats["token_to_bb4_atoms"]
        r = feats["coords"]
        if self.training and self.inverse_fold_noise > 0:
            noise = torch.randn_like(r) * self.inverse_fold_noise
            r = r + noise
        B, N = valid_mask.shape
        r_repr = torch.bmm(
            token_to_bb4_atoms.float().view(B, N * 4, -1), r.view(B, -1, 3)
        )
        r_repr = r_repr.reshape(B, N, 4, 3)
        r_repr = r_repr[valid_mask]

        dist = torch.norm(
            r_repr[src_idx, None] - r_repr[dst_idx, :, None], dim=-1
        ).flatten(start_dim=1)
        dist = self.distance_gaussian_smearing(dist)
        return dist

    @torch.no_grad()
    def extract_attr_feat(
        self, feats: Dict[str, Tensor], edge_idx: Tensor, valid_mask: Tensor
    ) -> Tensor:
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        feature_asym_id = feats["feature_asym_id"][valid_mask]
        b_same_chain = feature_asym_id[src_idx] == feature_asym_id[dst_idx]

        feature_residue_index = feats["feature_residue_index"][valid_mask]
        b_same_residue = (
            feature_residue_index[src_idx] == feature_residue_index[dst_idx]
        )
        d_residue = feature_residue_index[dst_idx] - feature_residue_index[src_idx]
        d_residue = torch.clip(
            d_residue + self.r_max,
            0,
            2 * self.r_max,
        )
        d_residue = torch.where(
            b_same_chain,
            d_residue,
            torch.zeros_like(d_residue) + 2 * self.r_max + 1,
        )
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        edge_attr = torch.cat(
            [
                a_rel_pos.float(),
                b_same_chain[..., None].float(),
                b_same_residue[..., None].float(),
            ],
            dim=-1,
        )

        b_standard = feats["is_standard"].bool()[valid_mask][..., None]
        mol_type = feats["mol_type"][valid_mask]
        mol_type_one_hot = F.one_hot(mol_type, num_classes=len(chain_type_ids))
        nonpolymer_mask = mol_type == chain_type_ids["NONPOLYMER"]
        modified = feats["modified"].unsqueeze(-1)[valid_mask]

        atom_feats = torch.cat(
            [
                feats["ref_charge"].unsqueeze(-1),
                feats["ref_element"],
            ],
            dim=-1,
        )

        with torch.autocast("cuda", enabled=False):
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token_mean = atom_to_token / (
                atom_to_token.sum(dim=1, keepdim=True) + 1e-6
            )
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_feats)

            atom_mask = feats["atom_pad_mask"]
            atom_padding_sum_to_token = torch.bmm(
                atom_to_token.transpose(1, 2), 1 - atom_mask.unsqueeze(-1)
            )
            assert atom_padding_sum_to_token[valid_mask].sum() == 0
        a = a[valid_mask]

        node_attr = torch.cat(
            [
                b_standard.float(),
                modified,
                mol_type_one_hot.float(),
                a,
            ],
            dim=-1,
        )

        return node_attr, edge_attr

    def forward(self, feats):
        edge_idx, valid_mask, bond = self.init_knn_graph(feats)
        geo_feat = self.extract_geo_feat(feats, edge_idx, valid_mask)
        node_attr, edge_attr = self.extract_attr_feat(feats, edge_idx, valid_mask)
        if self.enable_input_embedder:
            node_attr = feats["s_inputs"][valid_mask]

        N = valid_mask.sum()
        s = self.linear_token_to_node(node_attr)
        z = self.linear_token_to_pair(torch.cat([geo_feat, edge_attr, bond], dim=-1))

        for layer in self.encoder_layers:
            s, z = layer(s, z, edge_idx)

        return edge_idx, valid_mask, s, z



class InverseFoldingDecoder(Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        node_dim: int = 128,
        pair_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
        num_encoder_layers: int = 3,
        transformation_scale_factor: float = 1.0,
        inverse_fold_noise: float = 0.3,
        topk: int = 30,
        num_heads: int = 4,
        num_decoder_layers: int = 3,
        inverse_fold_restriction: List[str] = [],
        sampling_temperature: float = 0.1,
        **kwargs, # old checkpoint compatibility
    ):
        super().__init__()
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.num_encoder_layers = num_encoder_layers
        self.transformation_scale_factor = transformation_scale_factor
        self.inverse_fold_noise = inverse_fold_noise
        self.topk = topk
        self.num_heads = num_heads
        self.num_res_type = num_tokens
        self.num_decoder_layers = num_decoder_layers
        self.inverse_fold_restriction = inverse_fold_restriction
        self.sampling_temperature = sampling_temperature

        self.decoder_layers = ModuleList()
        self.inf = 10**6
        for i in range(self.num_decoder_layers):
            layer = MLPAttnGNNDecoder(
                node_dim=node_dim,
                pair_dim=pair_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
                transformation_scale_factor=transformation_scale_factor,
                num_heads=num_heads,
            )
            self.decoder_layers.append(layer)

        self.seq_to_s = nn.Linear(num_tokens, self.node_dim)
        self.predictor = nn.Linear(self.node_dim, num_tokens, bias=False)

        with torch.no_grad():
            # init the output of the predictor to be zero
            self.predictor.weight.zero_()

    def forward(self, s, z, edge_idx, valid_mask, feats):
        with torch.no_grad():
            src_idx, dst_idx = edge_idx[0], edge_idx[1]
            rand = torch.rand(valid_mask.sum(), device=valid_mask.device)
            res_type_edge_visibility = rand[src_idx] < rand[dst_idx]
            res_type_clone = feats["res_type_clone"].bool()
            res_type_clone = res_type_clone[valid_mask][src_idx]
            res_type_clone = res_type_clone * res_type_edge_visibility[:, None]
            res_type_clone = res_type_clone.to(z)
        res_rep = self.seq_to_s(res_type_clone)
        neighbors_rep = torch.concat([z, s[src_idx] + res_rep], dim=-1)
        for layer in self.decoder_layers:
            s = layer(s, neighbors_rep, edge_idx)

        logits = self.predictor(s)

        B, N = valid_mask.shape
        logist_dense = torch.zeros(B, N, self.num_res_type, device=logits.device)
        logist_dense[valid_mask] = logits

        out_dict = {
            "logits": logist_dense,
            "res_type": logist_dense,
            "valid_mask": valid_mask,
        }
        return out_dict

    @torch.no_grad()
    def sample(self, s, z, edge_idx, valid_mask, feats):
        """Sample the output from the decoder."""
        num_nodes = s.shape[0]

        if "inverse_fold_design_mask" in feats:
            design_mask = feats["inverse_fold_design_mask"].bool()[valid_mask]
        else:
            design_mask = feats["design_mask"].bool()[valid_mask]
        num_not_design = (~design_mask).sum().item()
        num_design = design_mask.sum().item()
        assert num_design == num_nodes - num_not_design, (
            f"num_design: {num_design}, num_not_design: {num_not_design}"
        )

        # Create restriction mask that sets the probability of excluded residues to 0
        if len(self.inverse_fold_restriction) > 0:
            restriction_mask = torch.zeros(len(canonical_tokens), device=s.device)
            for res_type in self.inverse_fold_restriction:
                restriction_mask[canonical_tokens.index(res_type)] = -self.inf
            restriction_mask = restriction_mask.unsqueeze(0)
        else:
            restriction_mask = torch.zeros(len(canonical_tokens), device=s.device)

        order = torch.randperm(num_nodes, device=s.device).cpu().numpy().tolist()
        # Non-design residues are not sampled and used as the condition. So the order should filter them out.
        if num_not_design > 0:
            id_not_design = torch.where(~design_mask)[0].cpu().numpy().tolist()
            for i in id_not_design:
                order.remove(i)
            decoded_seq = torch.zeros(num_nodes, num_tokens, device=s.device)
            logits = torch.zeros(num_nodes, num_tokens, device=s.device)
            decoded_seq[~design_mask] = logits[~design_mask] = feats["res_type_clone"][
                valid_mask
            ][~design_mask].float()
        else:
            decoded_seq = torch.zeros(num_nodes, num_tokens, device=s.device)
            logits = torch.zeros(num_nodes, num_tokens, device=s.device)
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        # decoding in order
        for i in order:
            s_i = s[i : i + 1]
            edge_mask_i = dst_idx == i
            z_i = z[edge_mask_i]
            src_idx_i = src_idx[edge_mask_i]
            res_type = decoded_seq[src_idx_i]
            res_rep = self.seq_to_s(res_type)
            neighbors_rep_i = torch.concat([z_i, s[src_idx_i] + res_rep], dim=-1)

            for layer in self.decoder_layers:
                s_i = layer.sample(s_i, neighbors_rep_i)

            logits_i = self.predictor(s_i)
            logits[i] = logits_i

            pred_canonical = (
                logits_i[
                    :,
                    canonicals_offset : len(canonical_tokens)
                    + canonicals_offset,
                ]
                + restriction_mask
            )
            ids_canonical = torch.argmax(pred_canonical, dim=-1)
            if self.sampling_temperature is None:
                ids_canonical = torch.argmax(pred_canonical, dim=-1)
            else:
                ids_canonical = torch.multinomial(
                    F.softmax(pred_canonical / self.sampling_temperature, dim=-1),
                    num_samples=1,
                ).squeeze(-1)

            ids = ids_canonical + canonicals_offset
            pred_one_hot = F.one_hot(ids, num_classes=num_tokens)
            decoded_seq[i] = pred_one_hot

        n_tokens = valid_mask.shape[1]
        res_type = torch.zeros(1, n_tokens, self.num_res_type, device=s.device)
        res_type[valid_mask] = decoded_seq
        unk_ids = torch.full(
            [(~valid_mask).sum().item()], tokens.index("UNK"), device=s.device
        )
        unk_value = F.one_hot(unk_ids, num_classes=num_tokens)
        res_type[~valid_mask] = unk_value.float()
        feats["res_type"] = res_type.long()

        logist_dense = torch.zeros(1, n_tokens, self.num_res_type, device=logits.device)
        logist_dense[valid_mask] = logits
        logist_dense[~valid_mask] = feats["res_type_clone"][~valid_mask].float()

        out_dict = {
            "logits": logist_dense,
            "res_type": res_type,
            "valid_mask": valid_mask,
        }
        out_dict["sample_atom_coords"] = feats["coords"][0]
        return out_dict


class BoltzMasker(Module):
    """Masking module for feats before passing to model forward."""

    def __init__(
        self,
        mask: bool = False,
        mask_backbone: bool = False,
        mask_disto: bool = False,
    ) -> None:
        """Initialize the masker.

        Parameters
        ----------
        mask : bool
            Whether or not to mask the input features.
        """
        super().__init__()
        self.mask = mask
        self.mask_backbone = mask_backbone
        self.mask_disto = mask_disto

    def forward(self, feats):
        if self.mask:
            new = {}
            new["id"] = feats["id"]
            new["structure_bonds"] = feats["structure_bonds"]
            skip_keys = [
                "id",
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
            ]

            clone = {
                k: v.clone()
                for k, v in feats.items()
                if (k not in skip_keys) and isinstance(v, torch.Tensor)
            }

            device = clone["token_index"].device
            token_pad_mask = clone["token_pad_mask"].bool()
            design_mask = clone["design_mask"].bool()
            token_mask = token_pad_mask & design_mask

            atom_pad_mask = clone["atom_pad_mask"].bool()
            atom_design_mask = (
                torch.bmm(
                    clone["atom_to_token"].float(), design_mask.float().unsqueeze(-1)
                )
                .squeeze()
                .bool()
            )
            atom_mask = atom_pad_mask & atom_design_mask
            if not self.mask_backbone:
                atom_mask = atom_mask & ~clone["backbone_mask"].bool()

            # token features that are copied
            new["contact_threshold"] = clone["contact_threshold"]
            new["contact_conditioning"] = clone["contact_conditioning"]
            new["design_mask"] = clone["design_mask"]
            if "inverse_fold_design_mask" in clone:
                new["inverse_fold_design_mask"] = clone["inverse_fold_design_mask"]
            if "chain_design_mask" in clone:
                new["chain_design_mask"] = clone["chain_design_mask"]
            new["token_index"] = clone["token_index"]
            new["residue_index"] = clone["residue_index"]
            new["is_standard"] = clone["is_standard"]
            new["token_resolved_mask"] = clone["token_resolved_mask"]
            new["asym_id"] = clone["asym_id"]
            new["entity_id"] = clone["entity_id"]
            new["sym_id"] = clone["sym_id"]
            new["mol_type"] = clone["mol_type"]
            new["token_pad_mask"] = clone["token_pad_mask"]
            new["token_disto_mask"] = clone["token_disto_mask"]
            new["msa_mask"] = clone["msa_mask"]
            if "disto_target" in clone:
                new["disto_target"] = clone["disto_target"]
            new["token_pair_mask"] = clone["token_pair_mask"]
            new["binding_type"] = clone["binding_type"]
            new["structure_group"] = clone["structure_group"]
            new["cyclic"] = clone["cyclic"]
            new["modified"] = clone["modified"]
            new["token_distance_mask"] = clone["token_distance_mask"]
            new["center_coords"] = clone["center_coords"]
            new["method_feature"] = clone["method_feature"]
            new["temp_feature"] = clone["temp_feature"]
            new["ph_feature"] = clone["ph_feature"]
            new["design_ss_mask"] = clone["design_ss_mask"]
            new["ss_type"] = clone["ss_type"]
            new["res_type_clone"] = clone["res_type_clone"]
            new["feature_residue_index"] = clone["feature_residue_index"]
            new["feature_asym_id"] = clone["feature_asym_id"]
            new["token_to_res"] = clone["token_to_res"]
            new["token_bonds"] = torch.zeros_like(
                clone["token_bonds"]
            )  # TODO: This is a mistake and needs to be fixed for the next training run. These should be copied not overridden (if we are not doing small molecule design)
            new["type_bonds"] = clone["type_bonds"]

            template_keys = [
                "visibility_ids",
                "query_to_template",
                "template_mask",
                "template_mask_frame",
                "template_mask_cb",
                "template_ca",
                "template_cb",
                "template_frame_t",
                "template_frame_rot",
                "template_restype",
            ]
            for k in template_keys:
                if k in clone.keys():
                    new[k] = clone[k]

            # atom features that are copied
            new["new_to_old_atomidx"] = clone["new_to_old_atomidx"]
            new["bfactor"] = clone["bfactor"]
            new["plddt"] = clone["plddt"]
            new["backbone_mask"] = clone["backbone_mask"]
            new["atom_resolved_mask"] = clone["atom_resolved_mask"]
            new["ref_space_uid"] = clone["ref_space_uid"]
            new["coords"] = clone["coords"]
            new["fake_atom_mask"] = clone["fake_atom_mask"]
            new["atom_pad_mask"] = clone["atom_pad_mask"]
            new["r_set_to_rep_atom"] = clone["r_set_to_rep_atom"]
            new["token_to_rep_atom"] = clone["token_to_rep_atom"]
            new["atom_to_token"] = clone["atom_to_token"]
            new["atom_pad_mask"] = clone["atom_pad_mask"]
            new["masked_ref_atom_name_chars"] = clone["masked_ref_atom_name_chars"]
            new["token_to_bb4_atoms"] = clone["token_to_bb4_atoms"]

            # apply token feature masking
            mask_val = one_hot(
                torch.ones(clone["res_type"].shape[:-1], device=device).long()
                * (token_ids["UNK"]),
                len(token_ids),
            )
            new["res_type"] = torch.where(
                token_mask[:, :, None], mask_val, clone["res_type"]
            )

            mask_val = torch.zeros_like(clone["ccd"])
            new["ccd"] = torch.where(token_mask[:, :, None], mask_val, clone["ccd"])

            mask_val = torch.ones_like(clone["msa"]) * token_ids["UNK"]
            new["msa"] = torch.where(token_mask[:, None, :], mask_val, clone["msa"])

            mask_val = torch.ones_like(clone["msa"]) * token_ids["-"]
            new["msa"][:, 1:] = torch.where(
                clone["target_msa_mask"][:, None, :].bool(),
                mask_val[:, 1:],
                new["msa"][:, 1:],
            )

            mask_val = torch.zeros_like(clone["msa_mask"][:, 1:])
            new["msa_mask"][:, 1:] = torch.where(
                clone["target_msa_mask"][:, None, :].bool(),
                mask_val,
                clone["msa_mask"][:, 1:],
            )

            mask_val = torch.zeros_like(clone["msa_paired"])
            new["msa_paired"] = torch.where(
                token_mask[:, None, :], mask_val, clone["msa_paired"]
            )
            mask_val = torch.zeros_like(clone["deletion_value"])
            new["deletion_value"] = torch.where(
                token_mask[:, None, :], mask_val, clone["deletion_value"]
            )

            # Mask disto loss for designed parts
            if self.mask_disto:
                mask_val = torch.zeros_like(clone["token_disto_mask"])
                new["token_disto_mask"] = torch.where(
                    token_mask,
                    mask_val,
                    clone["token_disto_mask"],
                )
            mask_val = torch.zeros_like(clone["has_deletion"])
            new["has_deletion"] = torch.where(
                token_mask[:, None, :], mask_val, clone["has_deletion"]
            )

            mask_val = torch.zeros_like(clone["deletion_mean"])
            new["deletion_mean"] = torch.where(
                token_mask, mask_val, clone["deletion_mean"]
            )

            mask_val = one_hot(
                torch.ones(clone["profile"].shape[:-1], device=device).long()
                * (token_ids["UNK"]),
                len(token_ids),
            ).to(clone["profile"].dtype)
            new["profile"] = torch.where(
                token_mask[:, :, None], mask_val, clone["profile"]
            )

            # apply atom feature designability mask
            mask_val = one_hot(
                torch.ones(clone["ref_element"].shape[:-1], device=device).long()
                * (mask_element_id),
                num_elements,
            )
            new["ref_element"] = torch.where(
                atom_mask[:, :, None],
                mask_val,
                clone["ref_element"],
            )

            mask_val = torch.zeros_like(clone["ref_charge"])
            new["ref_charge"] = torch.where(atom_mask, mask_val, clone["ref_charge"])

            mask_val = (
                torch.ones_like(clone["ref_chirality"]).long()
                * chirality_type_ids["CHI_UNSPECIFIED"]
            )
            new["ref_chirality"] = torch.where(
                atom_mask, mask_val, clone["ref_chirality"]
            )

            mask_val = clone["masked_ref_atom_name_chars"].clone()
            new["ref_atom_name_chars"] = torch.where(
                atom_mask[:, :, None, None],
                mask_val,
                clone["ref_atom_name_chars"],
            )

            # for the ref_pos, the backbone positions might be leaking information about the residue identity.
            # They might not always be the same across all residues. So we always mask them independent of mask_backbone.
            mask = atom_pad_mask & atom_design_mask
            mask_val = torch.zeros_like(clone["ref_pos"])
            new["ref_pos"] = torch.where(
                mask[:, :, None],
                mask_val,
                clone["ref_pos"],
            )
        else:
            new = feats
        return new


class ContactConditioning(Module):
    def __init__(self, token_z: int, cutoff_min: float, cutoff_max: float):
        super().__init__()

        self.fourier_embedding = FourierEmbedding(token_z)
        self.encoder = nn.Linear(
            token_z + len(contact_conditioning_info) - 1, token_z
        )
        self.encoding_unspecified = nn.Parameter(torch.zeros(token_z))
        self.encoding_unselected = nn.Parameter(torch.zeros(token_z))
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

    def forward(self, feats):
        assert contact_conditioning_info["UNSPECIFIED"] == 0
        assert contact_conditioning_info["UNSELECTED"] == 1
        contact_conditioning = feats["contact_conditioning"][:, :, :, 2:]
        contact_threshold = feats["contact_threshold"]
        contact_threshold_normalized = (contact_threshold - self.cutoff_min) / (
            self.cutoff_max - self.cutoff_min
        )
        contact_threshold_fourier = self.fourier_embedding(
            contact_threshold_normalized.flatten()
        ).reshape(contact_threshold_normalized.shape + (-1,))

        contact_conditioning = torch.cat(
            [
                contact_conditioning,
                contact_threshold_normalized.unsqueeze(-1),
                contact_threshold_fourier,
            ],
            dim=-1,
        )
        contact_conditioning = self.encoder(contact_conditioning)

        contact_conditioning = (
            contact_conditioning
            * (
                1
                - feats["contact_conditioning"][:, :, :, 0:2].sum(dim=-1, keepdim=True)
            )
            + self.encoding_unspecified * feats["contact_conditioning"][:, :, :, 0:1]
            + self.encoding_unselected * feats["contact_conditioning"][:, :, :, 1:2]
        )
        return contact_conditioning


class InputEmbedder(Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        activation_checkpointing: bool = False,
        add_method_conditioning: bool = False,
        add_modified_flag: bool = False,
        add_cyclic_flag: bool = False,
        add_mol_type_feat: bool = False,
        add_ph_flag: bool = False,
        add_temp_flag: bool = False,
        add_design_mask_flag: bool = False,
        add_binding_specification: bool = False,
        add_ss_specification: bool = False,
    ) -> None:
        """Initialize the input embedder.

        Parameters
        ----------
        atom_s : int
            The atom embedding size.
        atom_z : int
            The atom pairwise embedding size.
        token_s : int
            The token embedding size.

        """
        super().__init__()
        self.token_s = token_s
        self.add_method_conditioning = add_method_conditioning
        self.add_modified_flag = add_modified_flag
        self.add_cyclic_flag = add_cyclic_flag
        self.add_mol_type_feat = add_mol_type_feat
        self.add_ph_flag = add_ph_flag
        self.add_temp_flag = add_temp_flag
        self.add_design_mask_flag = add_design_mask_flag
        self.add_binding_specification = add_binding_specification
        self.add_ss_specification = add_ss_specification

        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=False,
        )

        self.atom_enc_proj_z = nn.Sequential(
            nn.LayerNorm(atom_z),
            nn.Linear(atom_z, atom_encoder_depth * atom_encoder_heads, bias=False),
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=False,
            activation_checkpointing=activation_checkpointing,
        )

        self.res_type_encoding = nn.Linear(num_tokens, token_s, bias=False)
        self.msa_profile_encoding = nn.Linear(num_tokens + 1, token_s, bias=False)
        if add_method_conditioning:
            self.method_conditioning_init = nn.Embedding(
                num_method_types, token_s
            )
        if add_modified_flag:
            self.modified_conditioning_init = nn.Embedding(2, token_s)
        if add_cyclic_flag:
            self.cyclic_conditioning_init = nn.Linear(1, token_s, bias=False)
        if add_mol_type_feat:
            self.mol_type_conditioning_init = nn.Embedding(
                len(chain_type_ids), token_s
            )
        if add_ph_flag:
            self.ph_conditioning_init = nn.Embedding(num_ph_bins, token_s)
        if add_temp_flag:
            self.temp_conditioning_init = nn.Embedding(num_temp_bins, token_s)
        if add_binding_specification:
            self.binding_specification_conditioning_init = nn.Embedding(
                len(binding_types), token_s
            )
        if add_design_mask_flag:
            self.design_mask_conditioning_init = nn.Embedding(2, token_s)
        if add_ss_specification:
            self.ss_specification_init = nn.Embedding(len(ss_types), token_s)

    def forward(self, feats: Dict[str, Tensor], affinity: bool = False) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        feats : Dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The embedded tokens.

        """
        # Load relevant features
        res_type = feats["res_type"].float()
        if affinity:
            profile = feats["profile_affinity"]
            deletion_mean = feats["deletion_mean_affinity"].unsqueeze(-1)
        else:
            profile = feats["profile"]
            deletion_mean = feats["deletion_mean"].unsqueeze(-1)

        # Compute input embedding
        q, c, p, to_keys = self.atom_encoder(feats)
        atom_enc_bias = self.atom_enc_proj_z(p)
        a, _, _, _ = self.atom_attention_encoder(
            feats=feats,
            q=q,
            c=c,
            atom_enc_bias=atom_enc_bias,
            to_keys=to_keys,
        )

        s = (
            a
            + self.res_type_encoding(res_type)
            + self.msa_profile_encoding(torch.cat([profile, deletion_mean], dim=-1))
        )

        if self.add_method_conditioning:
            s = s + self.method_conditioning_init(feats["method_feature"])
        if self.add_modified_flag:
            s = s + self.modified_conditioning_init(feats["modified"])
        if self.add_cyclic_flag:
            cyclic = feats["cyclic"].clamp(max=1.0).unsqueeze(-1)
            s = s + self.cyclic_conditioning_init(cyclic)
        if self.add_mol_type_feat:
            s = s + self.mol_type_conditioning_init(feats["mol_type"])
        if self.add_ph_flag:
            s = s + self.ph_conditioning_init(feats["ph_feature"])
        if self.add_temp_flag:
            s = s + self.temp_conditioning_init(feats["temp_feature"])
        if self.add_design_mask_flag:
            s = s + self.design_mask_conditioning_init(feats["design_mask"].int())
        if self.add_binding_specification:
            s = s + self.binding_specification_conditioning_init(feats["binding_type"])
        if self.add_ss_specification:
            s = s + self.ss_specification_init(feats["ss_type"])

        return s


class TemplateModule(Module):
    """Template module."""

    def __init__(
        self,
        token_z: int,
        template_dim: int,
        template_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        miniformer_blocks: bool = False,
    ) -> None:
        """Initialize the template module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.z_norm = nn.LayerNorm(token_z)
        self.v_norm = nn.LayerNorm(template_dim)
        self.z_proj = nn.Linear(token_z, template_dim, bias=False)
        self.a_proj = nn.Linear(
            num_tokens * 2 + num_bins + 5,
            template_dim,
            bias=False,
        )
        self.u_proj = nn.Linear(template_dim, token_z, bias=False)

        if miniformer_blocks:
            self.pairformer = MiniformerNoSeqModule(
                template_dim,
                num_blocks=template_blocks,
                dropout=dropout,
                post_layer_norm=post_layer_norm,
                activation_checkpointing=activation_checkpointing,
            )
        else:
            self.pairformer = PairformerNoSeqModule(
                template_dim,
                num_blocks=template_blocks,
                dropout=dropout,
                pairwise_head_width=pairwise_head_width,
                pairwise_num_heads=pairwise_num_heads,
                post_layer_norm=post_layer_norm,
                activation_checkpointing=activation_checkpointing,
            )

    def forward(
        self,
        z: Tensor,
        feats: Dict[str, Tensor],
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        feats : Dict[str, Tensor]
            Input features
        pair_mask : Tensor
            The pair mask

        Returns
        -------
        Tensor
            The updated pairwise embeddings.

        """
        # Load relevant features
        asym_id = feats["asym_id"]
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        template_mask = feats["template_mask"].any(dim=2).float()
        num_templates = template_mask.sum(dim=1)
        num_templates = num_templates.clamp(min=1)

        # Compute pairwise masks
        b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
        b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]

        b_cb_mask = b_cb_mask[..., None]
        b_frame_mask = b_frame_mask[..., None]

        # Compute asym mask, template features only attend within the same chain
        B, T = res_type.shape[:2]  # noqa: N806
        asym_mask = (asym_id[:, :, None] == asym_id[:, None, :]).float()
        asym_mask = asym_mask[:, None].expand(-1, T, -1, -1)

        # Compute template features
        with torch.autocast(device_type="cuda", enabled=False):
            # Compute distogram
            cb_dists = torch.cdist(cb_coords, cb_coords)
            boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
            boundaries = boundaries.to(cb_dists.device)
            distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = one_hot(distogram, num_classes=self.num_bins)

            # Compute unit vector in each frame
            frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)
            frame_t = frame_t.unsqueeze(2).unsqueeze(-1)
            ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)
            vector = torch.matmul(frame_rot, (ca_coords - frame_t))
            norm = torch.norm(vector, dim=-1, keepdim=True)
            unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
            unit_vector = unit_vector.squeeze(-1)

            # Concatenate input features
            a_tij = [distogram, b_cb_mask, unit_vector, b_frame_mask]
            a_tij = torch.cat(a_tij, dim=-1)
            a_tij = a_tij * asym_mask.unsqueeze(-1)
            res_type_i = res_type[:, :, :, None]
            res_type_j = res_type[:, :, None, :]
            res_type_i = res_type_i.expand(-1, -1, -1, res_type.size(2), -1)
            res_type_j = res_type_j.expand(-1, -1, res_type.size(2), -1, -1)
            a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
            a_tij = self.a_proj(a_tij)

        # Expand mask
        pair_mask = pair_mask[:, None].expand(-1, T, -1, -1)
        pair_mask = pair_mask.reshape(B * T, *pair_mask.shape[2:])

        # Compute input projections
        v = self.z_proj(self.z_norm(z[:, None])) + a_tij
        v = v.view(B * T, *v.shape[2:])
        v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)
        v = self.v_norm(v)
        v = v.view(B, T, *v.shape[1:])

        # Aggregate templates
        template_mask = template_mask[:, :, None, None, None]
        num_templates = num_templates[:, None, None, None]
        u = (v * template_mask).sum(dim=1) / num_templates.to(v)

        # Compute output projection
        u = self.u_proj(self.relu(u))
        return u


class TokenDistanceModule(Module):
    """Template module."""

    def __init__(
        self,
        token_z: int,
        token_distance_dim: int,
        token_distance_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        distance_gaussian_dim: int = 32,
        miniformer_blocks: bool = False,
        use_token_distance_feats: bool = True,
    ) -> None:
        """Initialize the template module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.use_token_distance_feats = use_token_distance_feats
        self.relu = nn.ReLU()
        self.z_norm = nn.LayerNorm(token_z)
        self.v_norm = nn.LayerNorm(token_distance_dim)
        self.z_proj = nn.Linear(token_z, token_distance_dim, bias=False)
        self.a_proj = nn.Linear(
            num_bins + (4 * token_z if use_token_distance_feats else 0),
            token_distance_dim,
            bias=False,
        )
        self.u_proj = nn.Linear(token_distance_dim, token_z, bias=False)

        if miniformer_blocks:
            self.pairformer = MiniformerNoSeqModule(
                token_distance_dim,
                num_blocks=token_distance_blocks,
                dropout=dropout,
                post_layer_norm=post_layer_norm,
                activation_checkpointing=activation_checkpointing,
            )
        else:
            self.pairformer = PairformerNoSeqModule(
                token_distance_dim,
                num_blocks=token_distance_blocks,
                dropout=dropout,
                pairwise_head_width=pairwise_head_width,
                pairwise_num_heads=pairwise_num_heads,
                post_layer_norm=post_layer_norm,
                activation_checkpointing=activation_checkpointing,
            )

        if use_token_distance_feats:
            self.token_distance_encoder = DistanceTokenEncoder(
                distance_gaussian_dim=distance_gaussian_dim,
                token_z=token_z,
                out_dim=token_z,
            )

    def forward(
        self,
        z: Tensor,
        feats: Dict[str, Tensor],
        pair_mask: Tensor,
        relative_position_encoding,
        use_kernels: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        feats : Dict[str, Tensor]
            Input features
        pair_mask : Tensor
            The pair mask

        Returns
        -------
        Tensor
            The updated pairwise embeddings.

        """
        # Load relevant features
        token_distance_mask = feats["token_distance_mask"]
        token_coords = feats["center_coords"]

        # Compute template features
        with torch.autocast(device_type="cuda", enabled=False):
            # Compute distogram
            dists = torch.cdist(token_coords, token_coords)
            boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
            boundaries = boundaries.to(dists.device)
            distogram = (dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = one_hot(distogram, num_classes=self.num_bins)

            # Distance features
            if self.use_token_distance_feats:
                dist_features = self.token_distance_encoder(
                    relative_position_encoding, feats
                )
                a_ij = [distogram, dist_features]
                a_ij = torch.cat(a_ij, dim=-1)
            else:
                a_ij = distogram

            a_ij = a_ij * token_distance_mask.unsqueeze(-1)
            a_ij = self.a_proj(a_ij)

        (B,) = a_ij.shape[:1]  # noqa: N806
        v = self.z_proj(self.z_norm(z)) + a_ij
        v = v.view(B, *v.shape[1:])
        v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)
        v = self.v_norm(v)
        v = v.view(B, *v.shape[1:])

        # Compute output projection
        u = self.u_proj(self.relu(v))
        return u


class MSAModule(Module):
    """MSA module."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        token_s: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        miniformer_blocks: bool = True,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        use_paired_feature: bool = False,
    ) -> None:
        """Initialize the MSA module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature
        self.activation_checkpointing = activation_checkpointing

        self.s_proj = nn.Linear(token_s, msa_s, bias=False)
        self.msa_proj = nn.Linear(
            num_tokens + 2 + int(use_paired_feature),
            msa_s,
            bias=False,
        )
        self.layers = ModuleList()
        for i in range(msa_blocks):
            self.layers.append(
                MSALayer(
                    msa_s,
                    token_z,
                    msa_dropout,
                    z_dropout,
                    miniformer_blocks,
                    pairwise_head_width,
                    pairwise_num_heads,
                )
            )

    def forward(
        self,
        z: Tensor,
        emb: Tensor,
        feats: Dict[str, Tensor],
        use_kernels: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        emb : Tensor
            The input embeddings
        feats : Dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The output pairwise embeddings.

        """
        # Set chunk sizes
        if not self.training:
            if z.shape[1] > chunk_size_threshold:
                chunk_heads_pwa = True
                chunk_size_transition_z = 64
                chunk_size_transition_msa = 32
                chunk_size_outer_product = 4
                chunk_size_tri_attn = 128
            else:
                chunk_heads_pwa = False
                chunk_size_transition_z = None
                chunk_size_transition_msa = None
                chunk_size_outer_product = None
                chunk_size_tri_attn = 512
        else:
            chunk_heads_pwa = False
            chunk_size_transition_z = None
            chunk_size_transition_msa = None
            chunk_size_outer_product = None
            chunk_size_tri_attn = None

        # Load relevant features
        msa = feats["msa"]
        msa = torch.nn.functional.one_hot(msa, num_classes=num_tokens)
        has_deletion = feats["has_deletion"].unsqueeze(-1)
        deletion_value = feats["deletion_value"].unsqueeze(-1)
        is_paired = feats["msa_paired"].unsqueeze(-1)
        msa_mask = feats["msa_mask"]
        token_mask = feats["token_pad_mask"].float()
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

        # Compute input projections
        m = self.msa_proj(m)
        m = m + self.s_proj(emb).unsqueeze(1)

        # Perform MSA blocks
        for i in range(self.msa_blocks):
            if self.activation_checkpointing:
                z, m = torch.utils.checkpoint.checkpoint(
                    self.layers[i],
                    z,
                    m,
                    token_mask,
                    msa_mask,
                    chunk_heads_pwa,
                    chunk_size_transition_z,
                    chunk_size_transition_msa,
                    chunk_size_outer_product,
                    chunk_size_tri_attn,
                    use_kernels=use_kernels,
                )
            else:
                z, m = self.layers[i](
                    z,
                    m,
                    token_mask,
                    msa_mask,
                    chunk_heads_pwa,
                    chunk_size_transition_z,
                    chunk_size_transition_msa,
                    chunk_size_outer_product,
                    chunk_size_tri_attn,
                    use_kernels=use_kernels,
                )
        return z


class MSALayer(Module):
    """MSA module."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        miniformer_blocks: bool = True,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ) -> None:
        """Initialize the MSA module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.msa_dropout = msa_dropout
        self.msa_transition = Transition(dim=msa_s, hidden=msa_s * 4)
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )

        if miniformer_blocks:
            self.pairformer_layer = MiniformerNoSeqLayer(
                token_z=token_z, dropout=z_dropout
            )
        else:
            self.pairformer_layer = PairformerNoSeqLayer(
                token_z=token_z,
                dropout=z_dropout,
                pairwise_head_width=pairwise_head_width,
                pairwise_num_heads=pairwise_num_heads,
            )

        self.outer_product_mean = OuterProductMean(
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
        )

    def forward(
        self,
        z: Tensor,
        m: Tensor,
        token_mask: Tensor,
        msa_mask: Tensor,
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z: int = None,
        chunk_size_transition_msa: int = None,
        chunk_size_outer_product: int = None,
        chunk_size_tri_attn: int = None,
        use_kernels: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        emb : Tensor
            The input embeddings
        feats : Dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The output pairwise embeddings.

        """
        # Communication to MSA stack
        msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)
        m = m + msa_dropout * self.pair_weighted_averaging(
            m, z, token_mask, chunk_heads_pwa
        )
        m = m + self.msa_transition(m, chunk_size_transition_msa)

        z = z + self.outer_product_mean(m, msa_mask, chunk_size_outer_product)

        # Compute pairwise stack
        z = self.pairformer_layer(
            z, token_mask, chunk_size_tri_attn, use_kernels=use_kernels
        )

        return z, m


class BFactorModule(Module):
    """BFactor Module."""

    def __init__(self, token_s: int, num_bins: int) -> None:
        """Initialize the bfactor module.

        Parameters
        ----------
        token_s : int
            The token embedding size.

        """
        super().__init__()
        self.bfactor = nn.Linear(token_s, num_bins)
        self.num_bins = num_bins

    def forward(self, s: Tensor) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        s : Tensor
            The sequence embeddings

        Returns
        -------
        Tensor
            The predicted bfactor histogram.

        """
        return self.bfactor(s)


class DistogramModule(Module):
    """Distogram Module."""

    def __init__(self, token_z: int, num_bins: int) -> None:
        """Initialize the distogram module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.distogram = nn.Linear(token_z, num_bins)
        self.num_bins = num_bins

    def forward(self, z: Tensor) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings

        Returns
        -------
        Tensor
            The predicted distogram.

        """
        z = z + z.transpose(1, 2)
        return self.distogram(z).reshape(
            z.shape[0], z.shape[1], z.shape[2], 1, self.num_bins
        )


class DiffusionModule(Module):
    """Algorithm 20."""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_layers: int = 1,
        token_transformer_depth: int = 6,
        token_transformer_heads: int = 8,
        use_miniformer: bool = False,
        diffusion_pairformer_args: Dict[str, Any] = None,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        gaussian_random_3d_encoding_dim: int = 0,
        transformer_post_ln: bool = False,
        tfmr_s: Optional[int] = None,
        predict_res_type: bool = False,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing
        if tfmr_s is None:
            tfmr_s = 2 * token_s
        self.tfmr_s = tfmr_s

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            tfmr_s=tfmr_s,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            gaussian_random_3d_encoding_dim=gaussian_random_3d_encoding_dim,
            transformer_post_layer_norm=transformer_post_ln,
            tfmr_s=tfmr_s,
            use_qk_norm=use_qk_norm,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(tfmr_s), LinearNoBias(tfmr_s, tfmr_s)
        )

        self.token_transformer_layers = ModuleList()
        self.token_pairformer_layers = ModuleList()

        self.token_transformer = DiffusionTransformer(
            dim=tfmr_s,
            dim_single_cond=tfmr_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            use_qk_norm=use_qk_norm,
        )

        self.a_norm = nn.LayerNorm(tfmr_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            tfmr_s=tfmr_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            predict_res_type=predict_res_type,
            use_qk_norm=use_qk_norm,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        if self.activation_checkpointing:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)

        # run token level transformations
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning["token_trans_bias"].float(),
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update, res_type = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return {
            "r_update": r_update,
            "token_a": a.detach(),
            "res_type": res_type,
        }


class OutTokenFeatUpdate(Module):
    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        mse_rotational_alignment: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
        second_order_correction: bool = False,
        pass_resolved_mask_diff_train: bool = False,
        sampling_schedule: str = "af3",
        noise_scale_function: str = "constant",
        step_scale_function: str = "constant",
        min_noise_scale: float = 1.0,
        max_noise_scale: float = 1.0,
        noise_scale_alpha: float = 1.0,
        noise_scale_beta: float = 1.0,
        min_step_scale: float = 1.0,
        max_step_scale: float = 1.0,
        step_scale_alpha: float = 1.0,
        step_scale_beta: float = 1.0,
        time_dilation: float = 1.0,
        time_dilation_start: float = 0.6,
        time_dilation_end: float = 0.8,
        pred_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std

        if pred_threshold is None:
            # disable nucleation mask
            self.pred_sigma_thresh = float("inf")
        else:
            q = norm.ppf(pred_threshold)
            self.pred_sigma_thresh = self.sigma_data * exp(self.P_mean + self.P_std * q)

        self.num_sampling_steps = num_sampling_steps
        self.sampling_schedule = sampling_schedule
        self.time_dilation = time_dilation
        self.time_dilation_start = time_dilation_start
        self.time_dilation_end = time_dilation_end
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.noise_scale_function = noise_scale_function
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self.noise_scale_alpha = noise_scale_alpha
        self.noise_scale_beta = noise_scale_beta
        self.step_scale = step_scale
        self.step_scale_function = step_scale_function
        self.min_step_scale = min_step_scale
        self.max_step_scale = max_step_scale
        self.step_scale_alpha = step_scale_alpha
        self.step_scale_beta = step_scale_beta
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.mse_rotational_alignment = mse_rotational_alignment
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.second_order_correction = second_order_correction
        self.pass_resolved_mask_diff_train = pass_resolved_mask_diff_train
        self.token_s = score_model_args["token_s"]

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return (
            log(sigma / self.sigma_data) * 0.25
        )  # note here the AF3 authors divide by sigma_data but not EDM

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        if training and self.pass_resolved_mask_diff_train:
            res_mask = (
                network_condition_kwargs["feats"]["atom_resolved_mask"]
                .unsqueeze(-1)
                .float()
            )
            noised_atom_coords = noised_atom_coords * res_mask.repeat_interleave(
                network_condition_kwargs["multiplicity"], 0
            )

        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )

        return denoised_coords, net_out

    def sample_schedule_af3(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data  # note: done by AF3 but not by EDM

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample_schedule_dilated(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        ts = steps / (num_sampling_steps - 1)

        # remap to dilate a particular interval
        def dilate(ts, start, end, dilation):
            x = end - start
            l = start
            u = 1 - end
            assert (dilation - 1) * x <= l + u, "dilation too large"

            inv_dilation = 1 / dilation
            ratio = (l + u + (1 - dilation) * x) / (l + u)
            inv_ratio = 1 / ratio
            lprime = l * ratio
            uprime = u * ratio
            xprime = x * dilation

            lower_third = ts * inv_ratio
            middle_third = (ts - lprime) * inv_dilation + l
            upper_third = (ts - (lprime + xprime)) * inv_ratio + l + x
            return (
                (ts < lprime) * lower_third
                + ((ts >= lprime) & (ts < lprime + xprime)) * middle_third
                + (ts >= lprime + xprime) * upper_third
            )

        dilated_ts = dilate(
            ts, self.time_dilation_start, self.time_dilation_end, self.time_dilation
        )
        sigmas = (
            self.sigma_max**inv_rho
            + dilated_ts * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data  # note: done by AF3 but not by EDM

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def beta_noise_scale_schedule(self, num_sampling_steps):
        t = np.linspace(0, 1, num_sampling_steps)
        beta_cdf_weights = torch.from_numpy(
            beta.cdf(1 - t, self.noise_scale_alpha, self.noise_scale_beta)
        )
        return (
            self.max_noise_scale
            + (self.min_noise_scale - self.max_noise_scale) * beta_cdf_weights
        )

    def beta_step_scale_schedule(self, num_sampling_steps=None):
        t = np.linspace(0, 1, num_sampling_steps)
        beta_cdf_weights = torch.from_numpy(
            beta.cdf(t, self.step_scale_alpha, self.step_scale_beta)
        )
        return (
            self.min_step_scale
            + (self.max_step_scale - self.min_step_scale) * beta_cdf_weights
        )

    def sample(
        self,
        atom_mask,  #: Bool['b m'] | None = None,
        num_sampling_steps=None,
        multiplicity=1,
        step_scale=None,
        noise_scale=None,
        inference_logging=False,
        **network_condition_kwargs,
    ):
        if self.training and self.step_scale_random is not None:
            step_scales = np.random.choice(self.step_scale_random) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        elif self.step_scale_function == "beta":
            step_scales = self.beta_step_scale_schedule(num_sampling_steps)
        else:
            step_scales = default(step_scale, self.step_scale) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        if self.noise_scale_function == "constant":
            noise_scales = default(noise_scale, self.noise_scale) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        elif self.noise_scale_function == "beta":
            noise_scales = self.beta_noise_scale_schedule(num_sampling_steps)
        else:
            raise ValueError(
                f"Invalid noise scale schedule: {self.noise_scale_function}"
            )
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        if self.sampling_schedule == "af3":
            sigmas = self.sample_schedule_af3(num_sampling_steps)
        elif self.sampling_schedule == "dilated":
            sigmas = self.sample_schedule_dilated(num_sampling_steps)

        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_gammas_ss_ns = list(
            zip(
                sigmas[:-1],
                sigmas[1:],
                gammas[1:],
                step_scales,
                noise_scales,
            )
        )

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        feats = network_condition_kwargs["feats"]

        # gradually denoise
        coords_traj = [atom_coords]
        x0_coords_traj = []
        for step_idx, (
            sigma_tm,
            sigma_t,
            gamma,
            step_scale,
            noise_scale,
        ) in optionally_tqdm(
            enumerate(sigmas_gammas_ss_ns),
            use_tqdm=inference_logging,
            desc="Denoising steps.",
        ):
            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
            # sigma_tm is sigma_t-1 and sigma_t is sigma_t
            t_hat = sigma_tm * (1 + gamma)
            noise_var = noise_scale**2 * (t_hat**2 - sigma_tm**2)

            atom_coords = center(atom_coords, atom_mask)

            if self.coordinate_augmentation_inference:
                random_R, random_tr = compute_random_augmentation(
                    multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
                )
                atom_coords = (
                    torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
                )

            eps = noise_scale * sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised, net_out = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        **network_condition_kwargs,
                    ),
                )

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            # note here I believe there is a mistake in the AF3 paper where they use atom_coords instead of atom_coords_noisy
            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            coords_traj.append(atom_coords_next)
            x0_coords_traj.append(atom_coords_denoised)
            atom_coords = atom_coords_next
        coords_traj.append(atom_coords)

        result = dict(
            sample_atom_coords=atom_coords,
            coords_traj=coords_traj,
            x0_coords_traj=x0_coords_traj,
        )

        return result

    # training
    def loss_weight(self, sigma):
        # note: in AF3 there is a + at denominator while in EDM a *, we think this is a mistake in the paper
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        # note: in AF3 the sample is scaled by sigma_data while in EDM it is not
        # in practice this just means scaling P_mean by the log

        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity
        atom_coords = feats["coords"]
        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)

        padded_sigmas = rearrange(sigmas, "b -> b 1 1")
        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise
        # alphas=1. in paper

        denoised_atom_coords, net_out = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            training=True,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        out_dict = {
            "noised_atom_coords": noised_atom_coords,
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }
        out_dict.update(net_out)

        return out_dict


class Boltz(Module):
    """Boltz Implementation."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: Dict[str, Any],
        validation_args: Dict[str, Any],
        embedder_args: Dict[str, Any],
        msa_args: Dict[str, Any],
        pairformer_args: Dict[str, Any],
        score_model_args: Dict[str, Any],
        diffusion_process_args: Dict[str, Any],
        diffusion_loss_args: Dict[str, Any],
        affinity_model_args: Dict[str, Any] = {},
        affinity_mw_correction: bool = True,
        affinity_ensemble: bool = False,
        affinity_model_args1: Dict[str, Any] = {},
        affinity_model_args2: Dict[str, Any] = {},
        confidence_model_args: Optional[Dict[str, Any]] = None,
        validators: Any = None,
        masker_args: dict[str, Any] = {},
        num_val_datasets: int = 1,
        atom_feature_dim: int = 128,
        template_args: Optional[Dict] = None,
        use_miniformer: bool = True,
        confidence_prediction: bool = False,
        affinity_prediction: bool = False,
        token_level_confidence: bool = True,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = True,
        validate_structure: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        exclude_ions_from_lddt: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        ignore_ckpt_shape_mismatch: bool = False,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[Dict[str, Any]] = None,
        checkpoints: Optional[Dict[str, Any]] = None,
        step_scale_schedule: Optional[List[Dict[str, float]]] = None,
        noise_scale_schedule: Optional[List[Dict[str, float]]] = None,
        aggregate_distogram: bool = True,
        bond_type_feature: bool = False,
        no_random_recycling_training: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        use_templates: bool = False,
        use_token_distances: bool = False,
        token_distance_args: Optional[Dict] = None,
        predict_bfactor: bool = False,
        log_loss_every_steps: int = 50,
        checkpoint_diffusion_conditioning: bool = False,
        freeze_template_weights: bool = False,
        refolding_validator=None,
        predict_res_type: bool = False,
        inverse_fold: bool = False,
        inverse_fold_args: Optional[Dict[str, Any]] = None,
        inference_logging: bool = False,
        use_kernels: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        """
        Module that does either:
        1. Design
        2. Folding with confidence prediction
        3. Inverse folding
        4. Affinity prediction
        """
        # Note: save_hyperparameters() is a LightningModule method
        # Since this is nn.Module, we skip it or manually save config
        self.inverse_fold = inverse_fold
        self.inference_logging = inference_logging

        self.use_kernels = use_kernels

        # No random recycling
        self.no_random_recycling_training = no_random_recycling_training

        # Arguments
        self.training_args = training_args
        self.validation_args = validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_args = predict_args
        self.refolding_validator = refolding_validator
        self.predict_res_type = predict_res_type

        # Checkpoints
        self.checkpoints = checkpoints
        self.inference_counter = 0

        if checkpoints:
            self.first_checkpoint_num_samples = checkpoints.get(
                "first_checkpoint_num_samples", 1.0
            )
            self.checkpoint_list = checkpoints.get("checkpoint_list", [])
            self.total_samples = None
            self.switch_points = []
            self.checkpoint_paths = []
            self.current_checkpoint_index = -1

        if "affinity_args" not in affinity_model_args:
            affinity_model_args["affinity_args"] = {}
        if "groups" not in affinity_model_args["affinity_args"]:
            affinity_model_args["affinity_args"]["groups"] = {0: 1}
        if "val_groups" not in affinity_model_args["affinity_args"]:
            affinity_model_args["affinity_args"]["val_groups"] = set([0])

        self.exclude_ions_from_lddt = exclude_ions_from_lddt

        # Distogram
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.aggregate_distogram = aggregate_distogram

        # Trunk
        self.use_miniformer = use_miniformer

        # Masker
        self.masker = BoltzMasker(**masker_args)

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            **embedder_args,
        }
        if not self.inverse_fold:
            self.input_embedder = InputEmbedder(**full_embedder_args)

            self.s_init = nn.Linear(token_s, token_s, bias=False)
            self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
            self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

            self.rel_pos = RelativePositionEncoder(token_z)

            self.token_bonds = nn.Linear(
                1,
                token_z,
                bias=False,
            )
            self.bond_type_feature = bond_type_feature
            if bond_type_feature:
                self.token_bonds_type = nn.Embedding(len(bond_types) + 1, token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=token_z,
                cutoff_min=conditioning_cutoff_min,
                cutoff_max=conditioning_cutoff_max,
            )

            # Normalization layers
            self.s_norm = nn.LayerNorm(token_s)
            self.z_norm = nn.LayerNorm(token_z)

            # Recycling projections
            self.s_recycle = nn.Linear(token_s, token_s, bias=False)
            self.z_recycle = nn.Linear(token_z, token_z, bias=False)

        # Pairwise stack
        self.use_token_distances = use_token_distances
        if self.use_token_distances:
            self.token_distance_module = TokenDistanceModule(
                token_z, **token_distance_args
            )

        self.freeze_template_weights = freeze_template_weights
        self.use_templates = use_templates

        if use_templates:
            self.template_module = TemplateModule(token_z, **template_args)

        if not self.inverse_fold:
            self.msa_module = MSAModule(
                token_z=token_z,
                token_s=token_s,
                **msa_args,
            )
            pairformer_class = MiniformerModule if use_miniformer else PairformerModule
            self.pairformer_module = pairformer_class(
                token_s, token_z, **pairformer_args
            )
            self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning

            self.diffusion_conditioning = DiffusionConditioning(
                token_s=token_s,
                token_z=token_z,
                atom_s=atom_s,
                atom_z=atom_z,
                atoms_per_window_queries=atoms_per_window_queries,
                atoms_per_window_keys=atoms_per_window_keys,
                atom_encoder_depth=score_model_args["atom_encoder_depth"],
                atom_encoder_heads=score_model_args["atom_encoder_heads"],
                token_transformer_depth=score_model_args["token_transformer_depth"],
                token_transformer_heads=score_model_args["token_transformer_heads"],
                atom_decoder_depth=score_model_args["atom_decoder_depth"],
                atom_decoder_heads=score_model_args["atom_decoder_heads"],
                atom_feature_dim=atom_feature_dim,
                conditioning_transition_layers=score_model_args[
                    "conditioning_transition_layers"
                ],
            )

            # Output modules
            self.structure_module = AtomDiffusion(
                score_model_args={
                    "token_s": token_s,
                    "atom_s": atom_s,
                    "atoms_per_window_queries": atoms_per_window_queries,
                    "atoms_per_window_keys": atoms_per_window_keys,
                    "predict_res_type": predict_res_type,
                    **score_model_args,
                },
                **diffusion_process_args,
            )
            self.distogram_module = DistogramModule(token_z, num_bins)
            self.predict_bfactor = predict_bfactor
            if predict_bfactor:
                self.bfactor_module = BFactorModule(token_s, num_bins)

        self.confidence_prediction = confidence_prediction
        self.token_level_confidence = token_level_confidence
        self.alpha_pae = alpha_pae

        self.structure_prediction_training = structure_prediction_training

        ### Affinity ###
        self.affinity_prediction = affinity_prediction
        self.affinity_ensemble = affinity_ensemble
        self.affinity_mw_correction = affinity_mw_correction
        self.validate_structure = validate_structure

        if self.affinity_prediction:
            if self.affinity_ensemble:
                self.affinity_module1 = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args1,
                )
                self.affinity_module2 = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args2,
                )
            else:
                self.affinity_module = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args,
                )

        if self.confidence_prediction:
            self.confidence_module = ConfidenceModule(
                token_s,
                token_z,
                token_level_confidence=token_level_confidence,
                bond_type_feature=bond_type_feature,
                conditioning_cutoff_min=conditioning_cutoff_min,
                conditioning_cutoff_max=conditioning_cutoff_max,
                **confidence_model_args,
            )

        if self.inverse_fold:
            self.enable_if_input_embedder = False
            if inverse_fold_args.get("enable_input_embedder", False):
                self.enable_if_input_embedder = True
                self.input_embedder = InputEmbedder(**full_embedder_args)
            self.predict_bfactor = False
            self.inverse_folding_encoder = InverseFoldingEncoder(**inverse_fold_args)
            self.structure_module = InverseFoldingDecoder(**inverse_fold_args)

    def load_checkpoint_weights(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Loaded weights from {checkpoint_path}")

    def forward(
        self,
        feats: Dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        run_confidence_sequentially: bool = False,
        return_z_feats: bool = False,
        step_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        dict_out = {}
        if self.inference_logging:
            print("\nRunning Trunk.\n")
        with torch.set_grad_enabled(
            (self.training and self.structure_prediction_training)
        ):
            if self.inverse_fold:
                if self.enable_if_input_embedder:
                    s_inputs = self.input_embedder(feats)
                    feats["s_inputs"] = s_inputs
                edge_idx, valid_mask, s, z = self.inverse_folding_encoder(feats)
                # Remove s_inputs from feats dictionary
                feats.pop("s_inputs", None)
            else:
                s_inputs = self.input_embedder(feats)

                # Initialize the sequence embeddings
                s_init = self.s_init(s_inputs)

                # Initialize pairwise embeddings
                z_init = (
                    self.z_init_1(s_inputs)[:, :, None]
                    + self.z_init_2(s_inputs)[:, None, :]
                )
                relative_position_encoding = self.rel_pos(feats)
                z_init = z_init + relative_position_encoding
                z_init = z_init + self.token_bonds(feats["token_bonds"].float())
                if self.bond_type_feature:
                    z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
                z_init = z_init + self.contact_conditioning(feats)

                # Perform rounds of the pairwise stack
                s = torch.zeros_like(s_init)
                z = torch.zeros_like(z_init)

                # Compute pairwise mask
                mask = feats["token_pad_mask"].float()
                pair_mask = mask[:, :, None] * mask[:, None, :]

            if not self.inverse_fold:
                for i in range(recycling_steps + 1):
                    with torch.set_grad_enabled(
                        (
                            self.training
                            and self.structure_prediction_training
                            and (i == recycling_steps)
                        )
                    ):
                        # Issue with unused parameters in autocast
                        if (
                            self.training
                            and (i == recycling_steps)
                            and torch.is_autocast_enabled()
                        ):
                            torch.clear_autocast_cache()

                        # Apply recycling
                        s = s_init + self.s_recycle(self.s_norm(s))
                        z = z_init + self.z_recycle(self.z_norm(z))

                        # Compute pairwise stack
                        if self.use_token_distances:
                            z = z + self.token_distance_module(
                                z, feats, pair_mask, relative_position_encoding
                            )

                        # Compute pairwise stack
                        if self.use_templates:
                            z = z + self.template_module(
                                z, feats, pair_mask, use_kernels=self.use_kernels
                            )

                        if not self.inverse_fold:
                            z = z + self.msa_module(
                                z, s_inputs, feats, use_kernels=self.use_kernels
                            )

                        s, z = self.pairformer_module(
                            s,
                            z,
                            mask=mask,
                            pair_mask=pair_mask,
                            use_kernels=self.use_kernels,
                        )

            if not self.inverse_fold:
                pdistogram = self.distogram_module(z)
                dict_out["pdistogram"] = pdistogram.float()

            if not self.inverse_fold:
                if self.checkpoint_diffusion_conditioning:
                    # TODO decide whether this should be with bf16 or not
                    (
                        q,
                        c,
                        to_keys,
                        atom_enc_bias,
                        atom_dec_bias,
                        token_trans_bias,
                    ) = torch.utils.checkpoint.checkpoint(
                        self.diffusion_conditioning,
                        s,
                        z,
                        relative_position_encoding,
                        feats,
                    )
                else:
                    (
                        q,
                        c,
                        to_keys,
                        atom_enc_bias,
                        atom_dec_bias,
                        token_trans_bias,
                    ) = self.diffusion_conditioning(
                        s_trunk=s,
                        z_trunk=z,
                        relative_position_encoding=relative_position_encoding,
                        feats=feats,
                    )
                diffusion_conditioning = {
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                }

                if self.predict_bfactor:
                    pbfactor = self.bfactor_module(s)
                    dict_out["pbfactor"] = pbfactor

            if (
                (not self.training)
                or self.confidence_prediction
                or self.affinity_prediction
            ):
                if self.inference_logging:
                    print("\nRunning Structure Module.\n")
                with torch.autocast("cuda", enabled=False):
                    if not self.inverse_fold:
                        struct_out = self.structure_module.sample(
                            s_trunk=s.float(),
                            s_inputs=s_inputs.float(),
                            feats=feats,
                            num_sampling_steps=num_sampling_steps,
                            atom_mask=feats["atom_pad_mask"].float(),
                            multiplicity=diffusion_samples,
                            diffusion_conditioning=diffusion_conditioning,
                            step_scale=step_scale,
                            noise_scale=noise_scale,
                            inference_logging=self.inference_logging,
                        )
                    else:
                        struct_out = self.structure_module.sample(
                            s=s,
                            z=z,
                            edge_idx=edge_idx,
                            valid_mask=valid_mask,
                            feats=feats,
                        )

                    dict_out.update(struct_out)

                if self.training and self.structure_prediction_training:
                    for idx in range(feats["token_index"].shape[0]):
                        minimum_lddt_symmetry_dist(
                            pred_distogram=pdistogram[idx],
                            feats=feats,
                            index_batch=idx,
                        )

            if self.training and (
                self.confidence_prediction or self.affinity_prediction
            ):
                assert len(feats["coords"].shape) == 4
                assert feats["coords"].shape[1] == 1, (
                    "Only one conformation is supported for confidence"
                )

            # Compute structure module
            if self.training and self.structure_prediction_training:
                atom_coords = feats["coords"]
                B, K, L = atom_coords.shape[0:3]
                assert K in (
                    multiplicity_diffusion_train,
                    1,
                )  # TODO make check somewhere else, expand to m % N == 0, m > N
                atom_coords = atom_coords.reshape(B * K, L, 3)
                atom_coords = atom_coords.repeat_interleave(
                    multiplicity_diffusion_train // K, 0
                )
                feats["coords"] = atom_coords  # (multiplicity, L, 3)
                assert len(feats["coords"].shape) == 3

                with torch.autocast("cuda", enabled=False):
                    if not self.inverse_fold:
                        struct_out = self.structure_module(
                            s_trunk=s.float(),
                            s_inputs=s_inputs.float(),
                            feats=feats,
                            multiplicity=multiplicity_diffusion_train,
                            diffusion_conditioning=diffusion_conditioning,
                        )
                    else:
                        struct_out = self.structure_module(
                            s=s,
                            z=z,
                            edge_idx=edge_idx,
                            valid_mask=valid_mask,
                            feats=feats,
                        )
                    dict_out.update(struct_out)

            elif self.training:
                feats["coords"] = feats["coords"].squeeze(1)
                assert len(feats["coords"].shape) == 3

        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    x_pred=(dict_out["sample_atom_coords"].detach()),
                    feats=feats,
                    pred_distogram_logits=(dict_out["pdistogram"][:, :, :, 0].detach()),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )

        if self.affinity_prediction:
            pad_token_mask = feats["token_pad_mask"][0]
            rec_mask = feats["mol_type"][0] == 0
            rec_mask = rec_mask * pad_token_mask
            lig_mask = feats["affinity_token_mask"][0].to(torch.bool)
            lig_mask = lig_mask * pad_token_mask
            cross_pair_mask = (
                lig_mask[:, None] * rec_mask[None, :]
                + rec_mask[:, None] * lig_mask[None, :]
                + lig_mask[:, None] * lig_mask[None, :]
            )
            z_affinity = z * cross_pair_mask[None, :, :, None]

            argsort = torch.argsort(dict_out["iptm"], descending=True)
            best_idx = argsort[0].item()
            coords_affinity = dict_out["sample_atom_coords"].detach()[best_idx][
                None, None
            ]
            s_inputs = self.input_embedder(feats, affinity=True)

            with torch.autocast("cuda", enabled=False):
                if self.affinity_ensemble:
                    dict_out_affinity1 = self.affinity_module1(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )

                    dict_out_affinity1["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity1["affinity_logits_binary"]
                        )
                    )
                    dict_out_affinity2 = self.affinity_module2(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out_affinity2["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity2["affinity_logits_binary"]
                        )
                    )

                    dict_out_affinity_ensemble = {
                        "affinity_pred_value": (
                            dict_out_affinity1["affinity_pred_value"]
                            + dict_out_affinity2["affinity_pred_value"]
                        )
                        / 2,
                        "affinity_probability_binary": (
                            dict_out_affinity1["affinity_probability_binary"]
                            + dict_out_affinity2["affinity_probability_binary"]
                        )
                        / 2,
                    }

                    dict_out_affinity1 = {
                        "affinity_pred_value1": dict_out_affinity1[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary1": dict_out_affinity1[
                            "affinity_probability_binary"
                        ],
                    }
                    dict_out_affinity2 = {
                        "affinity_pred_value2": dict_out_affinity2[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary2": dict_out_affinity2[
                            "affinity_probability_binary"
                        ],
                    }

                    if self.affinity_mw_correction:
                        model_coef = 1.03525938
                        mw_coef = -0.59992683
                        bias = 2.83288489
                        mw = feats["affinity_mw"][0] ** 0.3
                        dict_out_affinity_ensemble["affinity_pred_value"] = (
                            model_coef
                            * dict_out_affinity_ensemble["affinity_pred_value"]
                            + mw_coef * mw
                            + bias
                        )

                    dict_out.update(dict_out_affinity_ensemble)
                    dict_out.update(dict_out_affinity1)
                    dict_out.update(dict_out_affinity2)
                else:
                    dict_out_affinity = self.affinity_module(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out.update(
                        {
                            "affinity_pred_value": dict_out_affinity[
                                "affinity_pred_value"
                            ],
                            "affinity_probability_binary": torch.nn.functional.sigmoid(
                                dict_out_affinity["affinity_logits_binary"]
                            ),
                        }
                    )
        if return_z_feats:
            dict_out["z_feats"] = z

        # For stability checking as in, https://github.com/IntelliGen-AI/IntFold
        dict_out["s_trunk"] = s
        return dict_out

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict:
        # Skip invalid samples
        if "exception" in batch and any(batch["exception"]):
            print(f"WARNING: Skipping batch. Exception for {batch['id'][0]}")
            return {"exception": True}
        if "skip" in batch and any(batch["skip"]):
            print(f"WARNING: Skipping batch. Skip was set true for {batch['id'][0]}")
            return {"skip": True}

        # Checkpoint switching logic
        if self.checkpoints and self.next_switch_point is not None:
            if self.inference_counter == self.next_switch_point:
                self.current_checkpoint_index += 1
                if 0 <= self.current_checkpoint_index < len(self.checkpoint_paths):
                    checkpoint_path = self.checkpoint_paths[
                        self.current_checkpoint_index
                    ]
                    self.load_checkpoint_weights(checkpoint_path)
                    print(f"Switched checkpoint.")
                if self.current_checkpoint_index + 1 < len(self.switch_points):
                    self.next_switch_point = self.switch_points[
                        self.current_checkpoint_index + 1
                    ]
                else:
                    self.next_switch_point = None

        # Temperature switching logic
        if self.step_scale_schedule and self.next_step_scale_switch_point is not None:
            if self.inference_counter == self.next_step_scale_switch_point:
                self.current_step_scale_index += 1
                if self.current_step_scale_index < len(self.step_scale_values):
                    self.current_step_scale = self.step_scale_values[
                        self.current_step_scale_index
                    ]
                    if self.current_step_scale_index + 1 < len(
                        self.step_scale_switch_points
                    ):
                        self.next_step_scale_switch_point = (
                            self.step_scale_switch_points[
                                self.current_step_scale_index + 1
                            ]
                        )
                    else:
                        self.next_step_scale_switch_point = None
                print(f"Switched step_scale to {self.current_step_scale}")

        if self.noise_scale_schedule and self.next_noise_scale_switch_point is not None:
            if self.inference_counter == self.next_noise_scale_switch_point:
                self.current_noise_scale_index += 1
                if self.current_noise_scale_index < len(self.noise_scale_values):
                    self.current_noise_scale = self.noise_scale_values[
                        self.current_noise_scale_index
                    ]
                    if self.current_noise_scale_index + 1 < len(
                        self.noise_scale_switch_points
                    ):
                        self.next_noise_scale_switch_point = (
                            self.noise_scale_switch_points[
                                self.current_noise_scale_index + 1
                            ]
                        )
                    else:
                        self.next_noise_scale_switch_point = None
                print(f"Switched noise_scale to {self.current_noise_scale}")

        step_scale = getattr(self, "current_step_scale", None)
        noise_scale = getattr(self, "current_noise_scale", None)

        try:
            feat_masked = self.masker(batch)
            out = self(
                feat_masked,
                recycling_steps=self.predict_args["recycling_steps"],
                num_sampling_steps=self.predict_args["sampling_steps"],
                diffusion_samples=self.predict_args["diffusion_samples"],
                run_confidence_sequentially=True,
                step_scale=step_scale,
                noise_scale=noise_scale,
                return_z_feats=(
                    self.predict_args["return_z_feats"]
                    if "return_z_feats" in self.predict_args
                    else False
                ),
            )
            pred_dict = {"exception": False}
            pred_dict.update(feat_masked)

            if "keys_dict_batch" in self.predict_args:
                for key in self.predict_args["keys_dict_batch"]:
                    pred_dict[key] = batch[key]
            if (
                "return_z_feats" in self.predict_args
                and self.predict_args["return_z_feats"]
            ):
                pred_dict["z_feats"] = out["z_feats"]
            pred_dict["masks"] = batch["atom_pad_mask"]
            pred_dict["token_masks"] = batch["token_pad_mask"]
            if "keys_dict_out" in self.predict_args:
                for key in self.predict_args["keys_dict_out"]:
                    pred_dict[key] = out[key]

            # also save these keys for computing refolding metrics like scRMSD
            pred_dict["input_coords"] = batch["coords"]
            pred_dict["token_index"] = batch["token_index"]
            pred_dict["atom_resolved_mask"] = batch["atom_resolved_mask"]
            pred_dict["atom_to_token"] = batch["atom_to_token"]
            pred_dict["mol_type"] = batch["mol_type"]
            pred_dict["backbone_mask"] = batch["backbone_mask"]

            pred_dict["coords"] = out["sample_atom_coords"]
            if not self.inverse_fold:
                pred_dict["coords_traj"] = out["coords_traj"]
                pred_dict["x0_coords_traj"] = out["x0_coords_traj"]
            if self.confidence_prediction:
                # pred_dict["confidence"] = out.get("ablation_confidence", None)
                pred_dict["pde"] = out["pde"]
                pred_dict["plddt"] = out["plddt"]
                pred_dict["confidence_score"] = (
                    4 * out["complex_plddt"]
                    + (
                        out["iptm"]
                        if not torch.allclose(
                            out["iptm"], torch.zeros_like(out["iptm"])
                        )
                        else out["ptm"]
                    )
                ) / 5

                pred_dict["complex_plddt"] = out["complex_plddt"]
                pred_dict["complex_iplddt"] = out["complex_iplddt"]
                pred_dict["complex_pde"] = out["complex_pde"]
                pred_dict["complex_ipde"] = out["complex_ipde"]
                if self.alpha_pae > 0:
                    pred_dict["pae"] = out["pae"]
                    pred_dict["ptm"] = out["ptm"]
                    pred_dict["iptm"] = out["iptm"]
                    pred_dict["ligand_iptm"] = out["ligand_iptm"]
                    pred_dict["protein_iptm"] = out["protein_iptm"]
                    pred_dict["pair_chains_iptm"] = out["pair_chains_iptm"]

                if self.affinity_prediction:
                    pred_dict["affinity_pred_value"] = out["affinity_pred_value"]
                    pred_dict["affinity_probability_binary"] = out[
                        "affinity_probability_binary"
                    ]
                    if self.affinity_ensemble:
                        pred_dict["affinity_pred_value1"] = out["affinity_pred_value1"]
                        pred_dict["affinity_probability_binary1"] = out[
                            "affinity_probability_binary1"
                        ]
                        pred_dict["affinity_pred_value2"] = out["affinity_pred_value2"]
                        pred_dict["affinity_probability_binary2"] = out[
                            "affinity_probability_binary2"
                        ]
            self.inference_counter += 1
            return pred_dict

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                return {"exception": True}
            else:
                raise e


