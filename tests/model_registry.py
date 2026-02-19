import dataclasses
from typing import List, Optional


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    repo_id: str
    reference_repo_id: Optional[str]


REPRESENTATIVE_MODELS: List[ModelSpec] = [
    ModelSpec(key="e1_150m", family="e1", repo_id="Synthyra/Profluent-E1-150M", reference_repo_id="Profluent-Bio/E1-150m"),
    ModelSpec(key="esm2_8m", family="esm2", repo_id="Synthyra/ESM2-8M", reference_repo_id="facebook/esm2_t6_8M_UR50D"),
    ModelSpec(key="esmplusplus_small", family="esmplusplus", repo_id="Synthyra/ESMplusplus_small", reference_repo_id="EvolutionaryScale/esmc-300m-2024-12"),
    ModelSpec(key="dplm_150m", family="dplm", repo_id="Synthyra/DPLM-150M", reference_repo_id="airkingbd/dplm_150m"),
    ModelSpec(key="dplm2_150m", family="dplm2", repo_id="Synthyra/DPLM2-150M", reference_repo_id="airkingbd/dplm2_150m"),
]


FULL_MODELS: List[ModelSpec] = [
    ModelSpec(key="e1_150m", family="e1", repo_id="Synthyra/Profluent-E1-150M", reference_repo_id="Profluent-Bio/E1-150m"),
    ModelSpec(key="e1_300m", family="e1", repo_id="Synthyra/Profluent-E1-300M", reference_repo_id="Profluent-Bio/E1-300m"),
    ModelSpec(key="e1_600m", family="e1", repo_id="Synthyra/Profluent-E1-600M", reference_repo_id="Profluent-Bio/E1-600m"),
    ModelSpec(key="esm2_8m", family="esm2", repo_id="Synthyra/ESM2-8M", reference_repo_id="facebook/esm2_t6_8M_UR50D"),
    ModelSpec(key="esm2_35m", family="esm2", repo_id="Synthyra/ESM2-35M", reference_repo_id="facebook/esm2_t12_35M_UR50D"),
    ModelSpec(key="esm2_150m", family="esm2", repo_id="Synthyra/ESM2-150M", reference_repo_id="facebook/esm2_t30_150M_UR50D"),
    ModelSpec(key="esm2_650m", family="esm2", repo_id="Synthyra/ESM2-650M", reference_repo_id="facebook/esm2_t33_650M_UR50D"),
    ModelSpec(key="esm2_3b", family="esm2", repo_id="Synthyra/ESM2-3B", reference_repo_id="facebook/esm2_t36_3B_UR50D"),
    ModelSpec(key="esmplusplus_small", family="esmplusplus", repo_id="Synthyra/ESMplusplus_small", reference_repo_id="EvolutionaryScale/esmc-300m-2024-12"),
    ModelSpec(key="esmplusplus_large", family="esmplusplus", repo_id="Synthyra/ESMplusplus_large", reference_repo_id="EvolutionaryScale/esmc-600m-2024-12"),
    ModelSpec(key="dplm_150m", family="dplm", repo_id="Synthyra/DPLM-150M", reference_repo_id="airkingbd/dplm_150m"),
    ModelSpec(key="dplm_650m", family="dplm", repo_id="Synthyra/DPLM-650M", reference_repo_id="airkingbd/dplm_650m"),
    ModelSpec(key="dplm_3b", family="dplm", repo_id="Synthyra/DPLM-3B", reference_repo_id="airkingbd/dplm_3b"),
    ModelSpec(key="dplm2_150m", family="dplm2", repo_id="Synthyra/DPLM2-150M", reference_repo_id="airkingbd/dplm2_150m"),
    ModelSpec(key="dplm2_650m", family="dplm2", repo_id="Synthyra/DPLM2-650M", reference_repo_id="airkingbd/dplm2_650m"),
    ModelSpec(key="dplm2_3b", family="dplm2", repo_id="Synthyra/DPLM2-3B", reference_repo_id="airkingbd/dplm2_3b"),
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

