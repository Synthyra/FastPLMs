# Copyright 2025 EvolutionaryScale
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical amino-acid geometry tables used by ESMFold2.

The literal chemistry measurements are kept visible for review. Derived masks,
indices, frames, and atom mappings are built locally below and are checked
exactly against the pinned Biohub implementation.
"""

from __future__ import annotations

import functools
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

ca_ca = 3.80209737096
chi_angles_atoms = {
    "ALA": [],
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
    "UNK": [],
}
chi_angles_mask = [
    [1.0] * len(groups) + [0.0] * (4 - len(groups)) for groups in chi_angles_atoms.values()
]
_PI_PERIODIC_CHI = {"ASP": {1}, "GLU": {2}, "PHE": {1}, "TYR": {1}}
chi_pi_periodic = [
    [1.0 if chi_index in _PI_PERIODIC_CHI.get(residue_name, ()) else 0.0 for chi_index in range(4)]
    for residue_name in chi_angles_atoms
]
rigid_group_atom_positions: dict[str, list[list[Any]]] = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.526, -0.0, -0.0)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.0)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.525, -0.0, -0.0)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.0)],
        ["CG", 4, (0.616, 1.39, -0.0)],
        ["CD", 5, (0.564, 1.414, 0.0)],
        ["NE", 6, (0.539, 1.357, -0.0)],
        ["NH1", 7, (0.206, 2.301, 0.0)],
        ["NH2", 7, (2.078, 0.978, -0.0)],
        ["CZ", 7, (0.758, 1.093, -0.0)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.526, -0.0, -0.0)],
        ["CB", 0, (-0.531, -0.787, -1.2)],
        ["O", 3, (0.625, 1.062, 0.0)],
        ["CG", 4, (0.584, 1.399, 0.0)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.0)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.527, 0.0, -0.0)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.0)],
        ["CG", 4, (0.593, 1.398, -0.0)],
        ["OD1", 5, (0.61, 1.091, 0.0)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.524, 0.0, 0.0)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.0)],
        ["SG", 4, (0.728, 1.653, 0.0)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.526, 0.0, 0.0)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.0)],
        ["CG", 4, (0.615, 1.393, 0.0)],
        ["CD", 5, (0.587, 1.399, -0.0)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.06, 0.0)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.526, -0.0, -0.0)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.0)],
        ["CG", 4, (0.615, 1.392, 0.0)],
        ["CD", 5, (0.6, 1.397, 0.0)],
        ["OE1", 6, (0.607, 1.095, -0.0)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.517, -0.0, -0.0)],
        ["O", 3, (0.626, 1.062, -0.0)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.36, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.525, 0.0, 0.0)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.0)],
        ["CG", 4, (0.6, 1.37, -0.0)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.16, -0.0)],
        ["CE1", 5, (2.03, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.527, -0.0, -0.0)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.0)],
        ["CG1", 4, (0.534, 1.437, -0.0)],
        ["CG2", 4, (0.54, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.0)],
    ],
    "LEU": [
        ["N", 0, (-0.52, 1.363, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.525, -0.0, -0.0)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.0)],
        ["CG", 4, (0.678, 1.371, 0.0)],
        ["CD1", 5, (0.53, 1.43, -0.0)],
        ["CD2", 5, (0.535, -0.774, 1.2)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.526, 0.0, 0.0)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.0)],
        ["CG", 4, (0.619, 1.39, 0.0)],
        ["CD", 5, (0.559, 1.417, 0.0)],
        ["CE", 6, (0.56, 1.416, 0.0)],
        ["NZ", 7, (0.554, 1.387, 0.0)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.525, 0.0, 0.0)],
        ["CB", 0, (-0.523, -0.776, -1.21)],
        ["O", 3, (0.625, 1.062, -0.0)],
        ["CG", 4, (0.613, 1.391, -0.0)],
        ["SD", 5, (0.703, 1.695, 0.0)],
        ["CE", 6, (0.32, 1.786, -0.0)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.524, 0.0, -0.0)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.0)],
        ["CG", 4, (0.607, 1.377, 0.0)],
        ["CD1", 5, (0.709, 1.195, -0.0)],
        ["CD2", 5, (0.706, -1.196, 0.0)],
        ["CE1", 5, (2.102, 1.198, -0.0)],
        ["CE2", 5, (2.098, -1.201, -0.0)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.527, -0.0, 0.0)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.0)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],
    ],
    "SER": [
        ["N", 0, (-0.529, 1.36, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.525, -0.0, -0.0)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.0)],
        ["OG", 4, (0.503, 1.325, 0.0)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.526, 0.0, -0.0)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.0)],
        ["CG2", 4, (0.55, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.0)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.525, -0.0, 0.0)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.0)],
        ["CG", 4, (0.609, 1.37, -0.0)],
        ["CD1", 5, (0.824, 1.091, 0.0)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.53, -0.007)],
        ["NE1", 5, (2.14, 0.69, -0.004)],
        ["CH2", 5, (3.028, -2.89, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.524, -0.0, -0.0)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.0)],
        ["CG", 4, (0.607, 1.382, -0.0)],
        ["CD1", 5, (0.716, 1.195, -0.0)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.2, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.527, -0.0, -0.0)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.0)],
        ["CG1", 4, (0.54, 1.429, -0.0)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
    "UNK": [
        ["N", 0, (-0.525, 1.363, 0.0)],
        ["CA", 0, (0.0, 0.0, 0.0)],
        ["C", 0, (1.526, -0.0, -0.0)],
    ],
}
residue_atoms = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": [
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "N",
        "NE1",
        "O",
    ],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
    "UNK": ["C", "CA", "N"],
}
residue_atom_renaming_swaps = {
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}
van_der_waals_radius = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}
Bond = namedtuple("Bond", ["atom1_name", "atom2_name", "length", "stddev"])
BondAngle = namedtuple(
    "BondAngle", ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"]
)

_STEREO_CHEMICAL_PROPS_PATH = Path("evolutionaryscale/structure/stereo_chemical_props.txt")


def _bond_key(atom_a: str, atom_b: str) -> tuple[str, str]:
    """Return an order-independent key for a covalent bond."""

    return (atom_a, atom_b) if atom_a <= atom_b else (atom_b, atom_a)


def _read_stereo_sections(text: str) -> tuple[list[str], list[str]]:
    """Split the two tabular sections while discarding their headers."""

    lines = iter(text.splitlines())
    next(lines)
    bond_rows = list(iter(lambda: next(lines).strip(), "-"))
    next(lines)
    next(lines)
    angle_rows = list(iter(lambda: next(lines).strip(), "-"))
    return bond_rows, angle_rows


@functools.cache
def load_stereo_chemical_props() -> tuple[
    dict[str, list[Any]], dict[str, list[Any]], dict[str, list[Any]]
]:
    """Load covalent geometry and derive virtual bonds for bond angles.

    The returned dictionaries are keyed by three-letter residue name. Virtual
    bond lengths and uncertainties use the same operation order as the
    checkpoint's reference feature pipeline so their floating-point values are
    bitwise reproducible.
    """

    bond_rows, angle_rows = _read_stereo_sections(_STEREO_CHEMICAL_PROPS_PATH.read_text())
    residue_bonds: dict[str, list[Any]] = {}
    for row in bond_rows:
        atom_pair, residue_name, length, stddev = row.split()
        atom_a, atom_b = atom_pair.split("-")
        residue_bonds.setdefault(residue_name, []).append(
            Bond(atom_a, atom_b, float(length), float(stddev))
        )
    residue_bonds["UNK"] = []

    residue_bond_angles: dict[str, list[Any]] = {}
    for row in angle_rows:
        atom_triple, residue_name, angle_degrees, stddev_degrees = row.split()
        atom_a, atom_b, atom_c = atom_triple.split("-")
        residue_bond_angles.setdefault(residue_name, []).append(
            BondAngle(
                atom_a,
                atom_b,
                atom_c,
                float(angle_degrees) / 180.0 * np.pi,
                float(stddev_degrees) / 180.0 * np.pi,
            )
        )
    residue_bond_angles["UNK"] = []

    residue_virtual_bonds: dict[str, list[Any]] = {}
    for residue_name, angles in residue_bond_angles.items():
        lookup = {
            _bond_key(bond.atom1_name, bond.atom2_name): bond
            for bond in residue_bonds[residue_name]
        }
        derived = residue_virtual_bonds.setdefault(residue_name, [])
        for angle in angles:
            left = lookup[_bond_key(angle.atom1_name, angle.atom2_name)]
            right = lookup[_bond_key(angle.atom2_name, angle.atom3name)]
            theta = angle.angle_rad
            length = np.sqrt(
                left.length**2 + right.length**2 - 2 * left.length * right.length * np.cos(theta)
            )
            dl_outer = 0.5 / length
            dl_dgamma = 2 * left.length * right.length * np.sin(theta) * dl_outer
            dl_db1 = (2 * left.length - 2 * right.length * np.cos(theta)) * dl_outer
            dl_db2 = (2 * right.length - 2 * left.length * np.cos(theta)) * dl_outer
            stddev = np.sqrt(
                (dl_dgamma * angle.stddev) ** 2
                + (dl_db1 * left.stddev) ** 2
                + (dl_db2 * right.stddev) ** 2
            )
            derived.append(Bond(angle.atom1_name, angle.atom3name, length, stddev))
    return residue_bonds, residue_virtual_bonds, residue_bond_angles


between_res_bond_length_c_n = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]
between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]
between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
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
atom_order = {atom_type: i for (i, atom_type) in enumerate(atom_types)}
atom_type_num = len(atom_types)
restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "",
        "",
        "",
    ],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "",
        "",
        "",
        "",
    ],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "",
        "",
        "",
    ],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
    ],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    "UNK": ["N", "CA", "C", "", "", "", "", "", "", "", "", "", "", ""],
}
restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for (i, restype) in enumerate(restypes)}
restype_num = len(restypes)
unk_restype_index = restype_num
restypes_with_x = [*restypes, "X"]
restype_order_with_x = {restype: i for (i, restype) in enumerate(restypes_with_x)}
bb_atoms = ["N", "CA", "C", "O"]
hydrophobicity = {
    "ALA": 0.116,
    "ARG": -0.5,
    "ASN": -0.264,
    "ASP": -0.472,
    "CYS": 0.18,
    "GLN": -0.249,
    "GLU": -0.457,
    "GLY": 0.001,
    "HIS": -0.335,
    "ILE": 0.443,
    "LEU": 0.443,
    "LYS": -0.217,
    "MET": 0.238,
    "PHE": 0.5,
    "PRO": 0.211,
    "SER": -0.141,
    "THR": -0.05,
    "TRP": 0.378,
    "TYR": 0.38,
    "VAL": 0.325,
}
side_chain_asa = {
    "ALA": 64.7809,
    "ARG": 210.02,
    "ASN": 113.187,
    "ASP": 110.209,
    "CYS": 95.2439,
    "GLN": 147.855,
    "GLU": 143.924,
    "GLY": 23.1338,
    "HIS": 146.449,
    "ILE": 151.242,
    "LEU": 139.524,
    "LYS": 177.366,
    "MET": 164.674,
    "PHE": 186.7,
    "PRO": 111.533,
    "SER": 81.2159,
    "THR": 111.597,
    "TRP": 229.619,
    "TYR": 200.306,
    "VAL": 124.237,
}
amino_acid_volumes = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
    "X": 88.6,
}


def sequence_to_onehot(
    sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool = False
) -> np.ndarray:
    """Encode a sequence as X with shape ``(l, n_alphabet)``.

    Unknown uppercase letters map to ``X`` only when ``map_unknown_to_x`` is
    enabled. Non-alphabetic or lowercase input remains invalid in that mode.
    """

    n_alphabet = max(mapping.values()) + 1
    observed_indices = sorted(set(mapping.values()))
    if observed_indices != list(range(n_alphabet)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_aas-1 without any gaps. "
            f"Got: {sorted(mapping.values())}"
        )

    encoded: np.ndarray = np.empty(len(sequence), dtype=np.intp)
    for position, symbol in enumerate(sequence):
        if map_unknown_to_x:
            if not symbol.isalpha() or not symbol.isupper():
                raise ValueError(f"Invalid character in the sequence: {symbol}")
            encoded[position] = mapping.get(symbol, mapping["X"])
        else:
            encoded[position] = mapping[symbol]

    one_hot: np.ndarray = np.zeros((len(sequence), n_alphabet), dtype=np.int32)
    one_hot[np.arange(len(sequence)), encoded] = 1
    return one_hot


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
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
}
restype_3to1 = {v: k for (k, v) in restype_1to3.items()}
unk_restype = "UNK"
resnames = [restype_1to3[r] for r in restypes] + [unk_restype]
resname_to_idx = {resname: i for (i, resname) in enumerate(resnames)}
hydrophobic_resnames = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}
HHBLITS_AA_TO_ID = {
    "A": 0,
    "B": 2,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "J": 20,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 20,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "U": 1,
    "V": 17,
    "W": 18,
    "X": 20,
    "Y": 19,
    "Z": 3,
    "-": 21,
}
ID_TO_HHBLITS_AA = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
    21: "-",
}
restypes_with_x_and_gap = [*restypes, "X", "-"]
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = tuple(
    restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i]) for i in range(len(restypes_with_x_and_gap))
)


def _make_standard_atom_mask() -> np.ndarray:
    """Return M with shape ``(n_residue_types, n_atom_types)``."""

    mask: np.ndarray = np.zeros((restype_num + 1, atom_type_num), dtype=np.int32)
    for residue_index, residue_code in enumerate(restypes):
        residue_name = restype_1to3[residue_code]
        atom_indices = [atom_order[name] for name in residue_atoms[residue_name]]
        mask[residue_index, atom_indices] = 1
    return mask


STANDARD_ATOM_MASK = _make_standard_atom_mask()


def chi_angle_atom(atom_index: int) -> np.ndarray:
    """Return chi-group selectors X with shape ``(21, 37, 4)``."""

    selectors: list[np.ndarray] = []
    identity = np.eye(atom_type_num)
    for residue_code in restypes:
        groups = chi_angles_atoms[restype_1to3[residue_code]]
        indices = [atom_order[group[atom_index]] for group in groups]
        indices += [-1] * (4 - len(indices))
        selectors.append(identity[indices])
    selectors.append(np.zeros((4, atom_type_num)))
    return cast(np.ndarray, np.stack(selectors).transpose(0, 2, 1))


chi_atom_1_one_hot = chi_angle_atom(1)
chi_atom_2_one_hot = chi_angle_atom(2)
chi_angles_atom_indices = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
chi_angles_atom_indices = np.array(
    [chi_atoms + [[0, 0, 0, 0]] * (4 - len(chi_atoms)) for chi_atoms in chi_angles_atom_indices]
)
_chi_groups_for_atom: defaultdict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
    for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
        for atom_i, atom in enumerate(chi_group):
            _chi_groups_for_atom[res_name, atom].append((chi_group_i, atom_i))
chi_groups_for_atom = dict(_chi_groups_for_atom)


def _make_rigid_transformation_4x4(
    ex: np.ndarray, ey: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Construct homogeneous transform M from two basis vectors and an origin."""

    unit_x = ex / np.linalg.norm(ex)
    orthogonal_y = ey - np.dot(ey, unit_x) * unit_x
    unit_y = orthogonal_y / np.linalg.norm(orthogonal_y)
    unit_z = np.cross(unit_x, unit_y)
    rotation_and_origin = np.stack((unit_x, unit_y, unit_z, translation), axis=0).T
    homogeneous_row = np.asarray(((0.0, 0.0, 0.0, 1.0),))
    return cast(np.ndarray, np.concatenate((rotation_and_origin, homogeneous_row), axis=0))


restype_atom37_to_rigid_group: np.ndarray = np.zeros((21, 37), dtype=int)
restype_atom37_mask: np.ndarray = np.zeros((21, 37), dtype=np.float32)
restype_atom37_rigid_group_positions: np.ndarray = np.zeros((21, 37, 3), dtype=np.float32)
restype_atom14_to_rigid_group: np.ndarray = np.zeros((21, 14), dtype=int)
restype_atom14_mask: np.ndarray = np.zeros((21, 14), dtype=np.float32)
restype_atom14_rigid_group_positions: np.ndarray = np.zeros((21, 14, 3), dtype=np.float32)
restype_rigid_group_default_frame: np.ndarray = np.zeros((21, 8, 4, 4), dtype=np.float32)


def _make_rigid_group_constants() -> None:
    """Populate atom-to-frame assignments and default rigid transforms."""

    for residue_index, residue_code in enumerate(restypes_with_x):
        residue_name = restype_1to3[residue_code]
        atom14_names = restype_name_to_atom14_names[residue_name]
        for atom_name, group_index, coordinates in rigid_group_atom_positions[residue_name]:
            atom37_index = atom_order[atom_name]
            atom14_index = atom14_names.index(atom_name)
            restype_atom37_to_rigid_group[residue_index, atom37_index] = group_index
            restype_atom37_mask[residue_index, atom37_index] = 1
            restype_atom37_rigid_group_positions[residue_index, atom37_index] = coordinates
            restype_atom14_to_rigid_group[residue_index, atom14_index] = group_index
            restype_atom14_mask[residue_index, atom14_index] = 1
            restype_atom14_rigid_group_positions[residue_index, atom14_index] = coordinates

        positions = {
            atom_name: np.asarray(coordinates)
            for atom_name, _group_index, coordinates in rigid_group_atom_positions[residue_name]
        }
        restype_rigid_group_default_frame[residue_index, :2] = np.eye(4)
        restype_rigid_group_default_frame[residue_index, 2] = _make_rigid_transformation_4x4(
            positions["N"] - positions["CA"],
            np.asarray((1.0, 0.0, 0.0)),
            positions["N"],
        )
        restype_rigid_group_default_frame[residue_index, 3] = _make_rigid_transformation_4x4(
            positions["C"] - positions["CA"],
            positions["CA"] - positions["N"],
            positions["C"],
        )

        groups = chi_angles_atoms[residue_name]
        if groups:
            first_group = [positions[name] for name in groups[0]]
            restype_rigid_group_default_frame[residue_index, 4] = _make_rigid_transformation_4x4(
                first_group[2] - first_group[1],
                first_group[0] - first_group[1],
                first_group[2],
            )
        for chi_index, group in enumerate(groups[1:], start=1):
            axis_end = positions[group[2]]
            restype_rigid_group_default_frame[residue_index, 4 + chi_index] = (
                _make_rigid_transformation_4x4(
                    axis_end,
                    np.asarray((-1.0, 0.0, 0.0)),
                    axis_end,
                )
            )


_make_rigid_group_constants()


def make_atom14_dists_bounds(
    overlap_tolerance: float = 1.5,
    bond_length_tolerance_factor: float = 15.0,
) -> dict[str, np.ndarray]:
    """Build lower, upper, and uncertainty tensors with shape ``(21, 14, 14)``."""

    lower_bounds: np.ndarray = np.zeros((21, 14, 14), np.float32)
    upper_bounds: np.ndarray = np.zeros((21, 14, 14), np.float32)
    stddevs: np.ndarray = np.zeros((21, 14, 14), np.float32)
    residue_bonds, residue_virtual_bonds, _angles = load_stereo_chemical_props()
    for residue_index, residue_code in enumerate(restypes):
        residue_name = restype_1to3[residue_code]
        atom_names = restype_name_to_atom14_names[residue_name]
        for atom_a_index, atom_a_name in enumerate(atom_names):
            if not atom_a_name:
                continue
            radius_a = van_der_waals_radius[atom_a_name[0]]
            for atom_b_index, atom_b_name in enumerate(atom_names):
                if not atom_b_name or atom_a_index == atom_b_index:
                    continue
                clash_floor = radius_a + van_der_waals_radius[atom_b_name[0]] - overlap_tolerance
                lower_bounds[residue_index, atom_a_index, atom_b_index] = clash_floor
                lower_bounds[residue_index, atom_b_index, atom_a_index] = clash_floor
                upper_bounds[residue_index, atom_a_index, atom_b_index] = 1e10
                upper_bounds[residue_index, atom_b_index, atom_a_index] = 1e10

        for bond in residue_bonds[residue_name] + residue_virtual_bonds[residue_name]:
            atom_a_index = atom_names.index(bond.atom1_name)
            atom_b_index = atom_names.index(bond.atom2_name)
            lower = bond.length - bond_length_tolerance_factor * bond.stddev
            upper = bond.length + bond_length_tolerance_factor * bond.stddev
            for row, column in (
                (atom_a_index, atom_b_index),
                (atom_b_index, atom_a_index),
            ):
                lower_bounds[residue_index, row, column] = lower
                upper_bounds[residue_index, row, column] = upper
                stddevs[residue_index, row, column] = bond.stddev
    return {
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "stddev": stddevs,
    }


restype_atom14_ambiguous_atoms: np.ndarray = np.zeros((21, 14), dtype=np.float32)
restype_atom14_ambiguous_atoms_swap_idx = np.tile(np.arange(14, dtype=int), (21, 1))


def _make_atom14_ambiguity_feats() -> None:
    """Mark symmetry-equivalent atom names and their exchange indices."""

    for residue_name, swaps in residue_atom_renaming_swaps.items():
        residue_index = restype_order[restype_3to1[residue_name]]
        atom_names = restype_name_to_atom14_names[residue_name]
        for atom_a, atom_b in swaps.items():
            atom_a_index = atom_names.index(atom_a)
            atom_b_index = atom_names.index(atom_b)
            restype_atom14_ambiguous_atoms[residue_index, (atom_a_index, atom_b_index)] = 1
            restype_atom14_ambiguous_atoms_swap_idx[residue_index, (atom_a_index, atom_b_index)] = (
                atom_b_index,
                atom_a_index,
            )


_make_atom14_ambiguity_feats()


def aatype_to_str_sequence(aatype: np.ndarray) -> str:
    """Decode residue-type indices without changing their order."""

    return "".join(restypes_with_x[index] for index in aatype)


CA_TO_N_NORM = 1.4591
CA_TO_C_NORM = 1.5252


def _make_restype_atom37_to_atom14() -> np.ndarray:
    """Return atom37-to-atom14 lookup M with shape ``(21, 37)``."""

    rows: list[list[int]] = []
    for residue_code in restypes:
        names = restype_name_to_atom14_names[restype_1to3[residue_code]]
        atom14_index = {name: index for index, name in enumerate(names)}
        rows.append([atom14_index.get(name, 0) for name in atom_types])
    rows.append([0] * atom_type_num)
    return np.asarray(rows, dtype=np.int32)


def _make_restype_atom14_to_atom37() -> np.ndarray:
    """Return atom14-to-atom37 lookup M with shape ``(21, 14)``."""

    rows = [
        [atom_order.get(name, 0) for name in restype_name_to_atom14_names[residue_name]]
        for residue_name in resnames[:-1]
    ]
    rows.append([0] * 14)
    return np.asarray(rows, dtype=np.int32)


RESTYPE_ATOM14_TO_ATOM37 = _make_restype_atom14_to_atom37()
RESTYPE_ATOM37_TO_ATOM14 = _make_restype_atom37_to_atom14()
CHAIN_BREAK_TOKEN = "|"
