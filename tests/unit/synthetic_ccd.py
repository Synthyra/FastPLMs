"""A miniature chemical component dictionary of RDKit molecules for the ESMFold2 ligand and modified-residue tests.

The real dictionary is a hash-pinned asset that the offline tests do not have. The conformer store reads any mapping of
component identifiers to RDKit molecules whose atoms carry a `name` property and whose conformers carry a `name`
property, so these few components exercise the same code. Heavy atoms are named as in the dictionary; hydrogens are named
`H<index>`, and the store removes them.

Shapes: `n` heavy atoms in a component.
"""

from rdkit import Chem
from rdkit.Chem import AllChem

from fastplms.models.esmfold2 import esmfold2_conformers as conformers


CONFORMER_SEED = 7  # makes the embedded conformers repeatable


def component(
    smiles: str,
    names: list[str],
    leaving: tuple[str, ...] = (),
    conformer_names: tuple[str, ...] = ("Ideal",),
) -> Chem.Mol:
    """A molecule of `smiles` whose heavy atoms are named in SMILES order, with one embedded conformer per label."""
    molecule = Chem.MolFromSmiles(smiles)
    for atom, name in zip(molecule.GetAtoms(), names, strict=True):
        atom.SetProp("name", name)
        if name in leaving:
            atom.SetProp("leaving_atom", "1")
    molecule = Chem.AddHs(molecule)
    for index, atom in enumerate(molecule.GetAtoms()):
        if not atom.HasProp("name"):
            atom.SetProp("name", f"H{index}")
    for number in range(len(conformer_names)):
        AllChem.EmbedMolecule(molecule, randomSeed=CONFORMER_SEED + number, clearConfs=False)
    for conformer, label in zip(molecule.GetConformers(), conformer_names, strict=True):
        conformer.SetProp("name", label)
    return molecule


def mini_ccd() -> dict[str, Chem.Mol]:
    """Alanine, glycine, phosphoserine (with one leaving atom and two conformers), ethanol, and a zinc ion."""
    return {
        "ALA": component("N[C@@H](C)C(=O)O", ["N", "CA", "CB", "C", "O", "OXT"], leaving=("OXT",)),
        "GLY": component("NCC(=O)O", ["N", "CA", "C", "O", "OXT"], leaving=("OXT",)),
        "SEP": component(
            "N[C@@H](COP(=O)(O)O)C(=O)O",
            ["N", "CA", "CB", "OG", "P", "O1P", "O2P", "O3P", "C", "O", "OXT"],
            leaving=("OXT",),
            conformer_names=("Ideal", "Computed"),
        ),
        "EOH": component("CCO", ["C1", "C2", "O"]),
        "ZN": component("[Zn+2]", ["ZN"]),
    }


def mini_store() -> conformers._ChemicalComponentStore:
    """A conformer store that already holds the miniature dictionary, so it never looks for the real asset."""
    store = conformers._ChemicalComponentStore()
    store.molecules = mini_ccd()
    return store


def install_mini_ccd(monkeypatch) -> conformers._ChemicalComponentStore:
    """Make the miniature dictionary the module's conformer store for the length of one test."""
    store = mini_store()
    monkeypatch.setattr(conformers, "_STORE", store)
    return store
