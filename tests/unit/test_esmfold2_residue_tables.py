"""The ESMFold2 residue tables: the stereochemistry file reader, atom14 distance bounds, and sequence encodings.

The official stereochemistry file is not part of this tree, so the reader runs on a small file in its format.
"""

import math
import numpy as np
import pytest

from fastplms.models.esmfold2 import esmfold2_residue_constants as residue_constants


N_CA_LENGTH, N_CA_STDDEV = 1.459, 0.020
CA_C_LENGTH, CA_C_STDDEV = 1.525, 0.026
N_CA_C_ANGLE, N_CA_C_ANGLE_STDDEV = 111.0, 2.7  # degrees
BOND_TOLERANCE_FACTOR = 15.0  # the default of make_atom14_dists_bounds


def stereo_table() -> str:
    """The two tables of the official file, with the N-CA and CA-C bonds and the N-CA-C angle of every residue."""
    names = [residue_constants.restype_1to3[code] for code in residue_constants.restypes]
    bonds = [f"N-CA\t{name}\t{N_CA_LENGTH}\t{N_CA_STDDEV}\nCA-C\t{name}\t{CA_C_LENGTH}\t{CA_C_STDDEV}" for name in names]
    angles = [f"N-CA-C\t{name}\t{N_CA_C_ANGLE}\t{N_CA_C_ANGLE_STDDEV}" for name in names]
    return "Bond\tResidue\tMean\tStdDev\n" + "\n".join(bonds) + "\n-\n\nAngle\tResidue\tMean\tStdDev\n" + "\n".join(angles) + "\n-\n"


@pytest.fixture
def stereo_file(tmp_path, monkeypatch):
    path = tmp_path / "stereo_chemical_props.txt"
    path.write_text(stereo_table(), encoding="utf-8")
    monkeypatch.setattr(residue_constants, "_STEREO_CHEMICAL_PROPS_PATH", path)
    residue_constants.load_stereo_chemical_props.cache_clear()
    yield path
    residue_constants.load_stereo_chemical_props.cache_clear()


def test_a_bond_key_does_not_depend_on_the_order_of_its_atoms():
    assert residue_constants._bond_key("N", "CA") == ("CA", "N")
    assert residue_constants._bond_key("CA", "N") == ("CA", "N")
    assert residue_constants._bond_key("CA", "CA") == ("CA", "CA")


def test_the_two_tables_of_the_stereochemistry_file_are_split_without_their_headers():
    bond_rows, angle_rows = residue_constants._read_stereo_sections(stereo_table())

    assert len(bond_rows) == 40 and len(angle_rows) == 20
    assert bond_rows[0] == f"N-CA\tALA\t{N_CA_LENGTH}\t{N_CA_STDDEV}"
    assert angle_rows[0].startswith("N-CA-C\tALA")


def test_the_stereochemistry_file_gives_bonds_angles_and_the_virtual_bond_across_each_angle(stereo_file):
    bonds, virtual_bonds, angles = residue_constants.load_stereo_chemical_props()

    expected_length = math.sqrt(
        N_CA_LENGTH**2 + CA_C_LENGTH**2 - 2 * N_CA_LENGTH * CA_C_LENGTH * math.cos(math.radians(N_CA_C_ANGLE))
    )
    assert bonds["ALA"][0] == residue_constants.Bond("N", "CA", N_CA_LENGTH, N_CA_STDDEV)
    assert angles["ALA"][0].angle_rad == pytest.approx(math.radians(N_CA_C_ANGLE))
    assert angles["ALA"][0].stddev == pytest.approx(math.radians(N_CA_C_ANGLE_STDDEV))
    assert virtual_bonds["ALA"][0].atom1_name == "N" and virtual_bonds["ALA"][0].atom2_name == "C"
    assert virtual_bonds["ALA"][0].length == pytest.approx(expected_length)
    assert virtual_bonds["ALA"][0].stddev > 0
    assert bonds["UNK"] == [] and angles["UNK"] == [] and virtual_bonds["UNK"] == []
    assert residue_constants.load_stereo_chemical_props() is residue_constants.load_stereo_chemical_props()


def test_atom14_distance_bounds_hold_bonds_tight_and_clashes_loose(stereo_file):
    bounds = residue_constants.make_atom14_dists_bounds()
    alanine = residue_constants.restype_order["A"]
    names = residue_constants.restype_name_to_atom14_names["ALA"]
    n, ca, o = names.index("N"), names.index("CA"), names.index("O")

    assert {key: value.shape for key, value in bounds.items()} == {
        "lower_bound": (21, 14, 14),
        "upper_bound": (21, 14, 14),
        "stddev": (21, 14, 14),
    }
    # A bonded pair is held within a band around its mean length, in either order.
    for first, second in ((n, ca), (ca, n)):
        assert bounds["lower_bound"][alanine, first, second] == pytest.approx(N_CA_LENGTH - BOND_TOLERANCE_FACTOR * N_CA_STDDEV)
        assert bounds["upper_bound"][alanine, first, second] == pytest.approx(N_CA_LENGTH + BOND_TOLERANCE_FACTOR * N_CA_STDDEV)
        assert bounds["stddev"][alanine, first, second] == pytest.approx(N_CA_STDDEV)
    # A nonbonded pair of a nitrogen and an oxygen may come no closer than the van der Waals radii less the tolerance.
    assert bounds["lower_bound"][alanine, n, o] == pytest.approx(1.55 + 1.52 - 1.5)
    assert bounds["upper_bound"][alanine, n, o] == pytest.approx(1e10)
    assert bounds["lower_bound"][alanine, n, n] == 0 and bounds["lower_bound"][alanine, 13, 13] == 0


def test_a_sequence_encodes_as_one_hot_rows_and_unknown_letters_can_map_to_x():
    mapping = {"A": 0, "B": 1, "X": 2}

    plain = residue_constants.sequence_to_onehot("ABX", mapping)  # (l, n_alphabet)
    unknown_to_x = residue_constants.sequence_to_onehot("AZ", mapping, map_unknown_to_x=True)  # (l, n_alphabet)

    assert plain.tolist() == np.eye(3, dtype=np.int32).tolist()
    assert unknown_to_x.tolist() == [[1, 0, 0], [0, 0, 1]]
    with pytest.raises(KeyError):
        residue_constants.sequence_to_onehot("Z", mapping)
    with pytest.raises(ValueError, match="Invalid character"):
        residue_constants.sequence_to_onehot("a", mapping, map_unknown_to_x=True)
    with pytest.raises(ValueError, match="without any gaps"):
        residue_constants.sequence_to_onehot("A", {"A": 0, "B": 2})


def test_residue_type_indices_decode_to_their_letters_in_order():
    codes = np.array([0, 1, 20, 19])

    assert residue_constants.aatype_to_str_sequence(codes) == "ARXV"
    assert residue_constants.aatype_to_str_sequence(np.array([], dtype=int)) == ""
