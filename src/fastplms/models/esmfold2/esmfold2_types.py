"""Stable namespace for ESMFold2 input schema types."""

from __future__ import annotations

from . import esmfold2_input_builder as _input_schema
from .esmfold2_msa import MSA
from .esmfold2_parsing import FastaEntry

Modification = _input_schema.Modification
ProteinInput = _input_schema.ProteinInput
RNAInput = _input_schema.RNAInput
DNAInput = _input_schema.DNAInput
LigandInput = _input_schema.LigandInput
DistogramConditioning = _input_schema.DistogramConditioning
PocketConditioning = _input_schema.PocketConditioning
CovalentBond = _input_schema.CovalentBond
StructurePredictionInput = _input_schema.StructurePredictionInput

__all__ = [
    "MSA",
    "CovalentBond",
    "DNAInput",
    "DistogramConditioning",
    "FastaEntry",
    "LigandInput",
    "Modification",
    "PocketConditioning",
    "ProteinInput",
    "RNAInput",
    "StructurePredictionInput",
]
