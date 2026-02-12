import dataclasses
from typing import List, Optional


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    repo_id: str
    reference_repo_id: Optional[str]


REPRESENTATIVE_MODELS: List[ModelSpec] = [
    ModelSpec(key="e1_150m", family="e1", repo_id="Synthyra/Profluent-E1-150M", reference_repo_id=None),
    ModelSpec(key="esm2_8m", family="esm2", repo_id="Synthyra/ESM2-8M", reference_repo_id="facebook/esm2_t6_8M_UR50D"),
    ModelSpec(key="esmplusplus_small", family="esmplusplus", repo_id="Synthyra/ESMplusplus_small", reference_repo_id=None),
]


FULL_MODELS: List[ModelSpec] = [
    ModelSpec(key="e1_150m", family="e1", repo_id="Synthyra/Profluent-E1-150M", reference_repo_id=None),
    ModelSpec(key="e1_300m", family="e1", repo_id="Synthyra/Profluent-E1-300M", reference_repo_id=None),
    ModelSpec(key="e1_600m", family="e1", repo_id="Synthyra/Profluent-E1-600M", reference_repo_id=None),
    ModelSpec(key="esm2_8m", family="esm2", repo_id="Synthyra/ESM2-8M", reference_repo_id="facebook/esm2_t6_8M_UR50D"),
    ModelSpec(key="esm2_35m", family="esm2", repo_id="Synthyra/ESM2-35M", reference_repo_id="facebook/esm2_t12_35M_UR50D"),
    ModelSpec(key="esm2_150m", family="esm2", repo_id="Synthyra/ESM2-150M", reference_repo_id="facebook/esm2_t30_150M_UR50D"),
    ModelSpec(key="esm2_650m", family="esm2", repo_id="Synthyra/ESM2-650M", reference_repo_id="facebook/esm2_t33_650M_UR50D"),
    ModelSpec(key="esm2_3b", family="esm2", repo_id="Synthyra/ESM2-3B", reference_repo_id="facebook/esm2_t36_3B_UR50D"),
    ModelSpec(key="esmplusplus_small", family="esmplusplus", repo_id="Synthyra/ESMplusplus_small", reference_repo_id=None),
    ModelSpec(key="esmplusplus_large", family="esmplusplus", repo_id="Synthyra/ESMplusplus_large", reference_repo_id=None),
]


def get_model_specs(full_models: bool, families: Optional[List[str]]) -> List[ModelSpec]:
    models = FULL_MODELS if full_models else REPRESENTATIVE_MODELS
    if families is None:
        return list(models)

    normalized = [family.strip().lower() for family in families]
    selected: List[ModelSpec] = []
    for spec in models:
        if spec.family in normalized:
            selected.append(spec)
    assert len(selected) > 0, "No models selected for requested families."
    return selected

