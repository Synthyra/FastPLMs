from transformers import PretrainedConfig
from typing import Dict, Any, Optional, List
import json


class BoltzGenConfig(PretrainedConfig):
    """Configuration class for BoltzGen model."""
    model_type = "boltzgen"
    
    def _make_json_serializable(self, obj, visited=None):
        """Recursively convert non-JSON-serializable objects to serializable ones."""
        if visited is None:
            visited = set()
        
        # Handle basic JSON-serializable types first (fast path)
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        # Prevent infinite recursion with circular references
        obj_id = id(obj)
        if obj_id in visited:
            return "<circular_reference>"
        visited.add(obj_id)
        
        try:
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: self._make_json_serializable(v, visited) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._make_json_serializable(item, visited) for item in obj]
            else:
                # Check if it's JSON serializable
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    # For non-serializable types, convert to string
                    return str(obj)
        finally:
            visited.discard(obj_id)
    
    def __init__(
        self,
        atom_s: int = 128,
        atom_z: int = 16,
        token_s: int = 384,
        token_z: int = 128,
        num_bins: int = 64,
        training_args: Dict[str, Any] = {
            'recycling_steps': 3,
            'sampling_steps': 50,
            'sampling_steps_random': [20, 50, 200],
            'diffusion_multiplicity': 32,
            'diffusion_samples': 1,
            'affinity_loss_weight': 0.003,
            'confidence_loss_weight': 0.3,
            'diffusion_loss_weight': 4.0,
            'distogram_loss_weight': 0.03,
            'bfactor_loss_weight': 0.001,
            'adam_beta_1': 0.9,
            'adam_beta_2': 0.95,
            'adam_eps': 1e-08,
            'lr_scheduler': 'af3',
            'base_lr': 0.0,
            'max_lr': 0.001,
            'lr_warmup_no_steps': 1000,
            'lr_start_decay_after_n_steps': 50000,
            'lr_decay_every_n_steps': 50000,
            'lr_decay_factor': 0.95,
            'weight_decay': 0.003,
            'weight_decay_exclude': True,
            'symmetry_correction': True,
        },
        validation_args: Dict[str, Any] = {
            'recycling_steps': 3,
            'sampling_steps': 200,
            'diffusion_samples': 5,
            'run_confidence_sequentially': True,
            'symmetry_correction': True,
        },
        embedder_args: Dict[str, Any] = {
            'atom_encoder_depth': 3,
            'atom_encoder_heads': 4,
            'add_mol_type_feat': True,
            'add_method_conditioning': True,
            'add_modified_flag': True,
            'add_cyclic_flag': True,
        },
        msa_args: Dict[str, Any] = {
            'msa_s': 64,
            'msa_blocks': 4,
            'msa_dropout': 0.15,
            'z_dropout': 0.25,
            'miniformer_blocks': False,
            'pairwise_head_width': 32,
            'pairwise_num_heads': 4,
            'use_paired_feature': True,
            'activation_checkpointing': False,
        },
        pairformer_args: Dict[str, Any] = {
            'num_blocks': 64,
            'num_heads': 16,
            'dropout': 0.25,
            'post_layer_norm': False,
            'activation_checkpointing': False,
        },
        score_model_args: Dict[str, Any] = {
            'sigma_data': 16,
            'dim_fourier': 256,
            'atom_encoder_depth': 3,
            'atom_encoder_heads': 4,
            'token_transformer_depth': 24,
            'token_transformer_heads': 16,
            'atom_decoder_depth': 3,
            'atom_decoder_heads': 4,
            'conditioning_transition_layers': 2,
            'transformer_post_ln': False,
            'activation_checkpointing': False,
        },
        diffusion_process_args: Dict[str, Any] = {
            'sigma_min': 0.0004,
            'sigma_max': 160.0,
            'sigma_data': 16.0,
            'rho': 7,
            'P_mean': -1.2,
            'P_std': 1.5,
            'gamma_0': 0.8,
            'gamma_min': 1.0,
            'noise_scale': 1.0,
            'step_scale': 1.0,
            'step_scale_random': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            'mse_rotational_alignment': True,
            'coordinate_augmentation': True,
            'alignment_reverse_diff': True,
            'synchronize_sigmas': False,
        },
        diffusion_loss_args: Dict[str, Any] = {
            'add_smooth_lddt_loss': True,
            'add_bond_loss': False,
            'nucleotide_loss_weight': 5.0,
            'ligand_loss_weight': 10.0,
            'filter_by_plddt': 0.0,
        },
        affinity_model_args: Dict[str, Any] = {},
        affinity_mw_correction: bool = True,
        affinity_ensemble: bool = False,
        affinity_model_args1: Dict[str, Any] = {},
        affinity_model_args2: Dict[str, Any] = {},
        confidence_model_args: Optional[Dict[str, Any]] = {
            'use_gaussian': False,
            'num_dist_bins': 64,
            'max_dist': 22,
            'use_miniformer': False,
            'no_trunk_feats': False,
            'add_s_to_z_prod': True,
            'add_s_input_to_s': True,
            'use_s_diffusion': False,
            'add_z_input_to_z': True,
            'pairformer_args': {
                'num_blocks': 8,
                'num_heads': 16,
                'dropout': 0.25,
            },
            'confidence_args': {
                'num_plddt_bins': 50,
                'num_pde_bins': 64,
                'num_pae_bins': 64,
                'relative_confidence': 'none',
                'use_separate_heads': True,
            },
        },
        validators: Any = None,
        masker_args: dict[str, Any] = {},
        num_val_datasets: int = 4,
        chain_sampling_args: Optional[Any] = None,
        atom_feature_dim: int = 388,
        template_args: Optional[Dict] = {
            'template_dim': 64,
            'template_blocks': 2,
            'activation_checkpointing': False,
        },
        use_miniformer: bool = False,
        use_miniformer_plus: bool = False,
        recycling_detach: bool = True,
        confidence_prediction: bool = True,
        affinity_prediction: bool = False,
        token_level_confidence: bool = True,
        confidence_imitate_trunk: bool = False,
        confidence_regression: bool = False,
        affinity_transformer: bool = False,
        affinity_transformer_atom: bool = False,
        affinity_pair_transformer: bool = False,
        affinity_baseline: bool = False,
        affinity_confidence_different: bool = False,
        affinity_confidence_different_multiple: bool = False,
        alpha_pae: float = 1.0,
        relative_confidence_supervision_weight: float = 0.0,
        structure_prediction_training: bool = False,
        skip_run_structure: bool = False,
        tau_affinity_score: float = -1.0,
        alpha_affinity_absolute: float = 0.0,
        alpha_affinity_difference: float = 0.0,
        alpha_affinity_binary: float = 0.0,
        alpha_affinity_score_binder_decoy: float = 0.0,
        alpha_affinity_score_binder_binder: float = 0.0,
        alpha_affinity_focal: float = 0.0,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        compile_msa: bool = False,
        representative_lddt: bool = False,
        exclude_ions_from_lddt: bool = True,
        ema: bool = True,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        maximum_bond_distance: int = 0,
        predict_args: Optional[Dict[str, Any]] = None,
        dynamic_args: Dict[str, Any] = {},
        compute_tistogram: bool = False,
        num_tistogram_axis: int = 1,
        symmetry_correction_trunk: bool = False,
        fix_sym_check: bool = True,
        cyclic_pos_enc: bool = True,
        trunk_resolved_loss: bool = False,
        ignore_ckpt_shape_mismatch: bool = False,
        num_distograms: int = 1,
        checkpoints: Optional[Dict[str, Any]] = None,
        step_scale_schedule: Optional[List[Dict[str, float]]] = None,
        noise_scale_schedule: Optional[List[Dict[str, float]]] = None,
        aggregate_distogram: bool = True,
        bond_type_feature: bool = True,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        no_random_recycling_training: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        guidance_args: Optional[Any] = None,
        use_templates: bool = True,
        compile_templates: bool = False,
        use_token_distances: bool = False,
        token_distance_args: Optional[Dict] = None,
        predict_bfactor: bool = True,
        log_loss_every_steps: int = 50,
        checkpoint_diffusion_conditioning: bool = True,
        use_templates_v2: bool = True,
        freeze_template_weights: bool = False,
        refolding_validator: Optional[Any] = None,
        predict_res_type: bool = False,
        inverse_fold: bool = False,
        inverse_fold_args: Optional[Dict[str, Any]] = None,
        inference_logging: bool = False,
        use_kernels: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.num_bins = num_bins
        self.training_args = training_args
        self.validation_args = validation_args
        self.embedder_args = embedder_args
        self.msa_args = msa_args
        self.pairformer_args = pairformer_args
        self.score_model_args = score_model_args
        self.diffusion_process_args = diffusion_process_args
        self.diffusion_loss_args = diffusion_loss_args
        self.affinity_model_args = affinity_model_args
        self.affinity_mw_correction = affinity_mw_correction
        self.affinity_ensemble = affinity_ensemble
        self.affinity_model_args1 = affinity_model_args1
        self.affinity_model_args2 = affinity_model_args2
        self.confidence_model_args = confidence_model_args
        self.validators = validators
        self.masker_args = masker_args
        self.num_val_datasets = num_val_datasets
        self.chain_sampling_args = chain_sampling_args
        self.atom_feature_dim = atom_feature_dim
        self.template_args = template_args
        self.use_miniformer = use_miniformer
        self.use_miniformer_plus = use_miniformer_plus
        self.recycling_detach = recycling_detach
        self.confidence_prediction = confidence_prediction
        self.affinity_prediction = affinity_prediction
        self.token_level_confidence = token_level_confidence
        self.confidence_imitate_trunk = confidence_imitate_trunk
        self.confidence_regression = confidence_regression
        self.affinity_transformer = affinity_transformer
        self.affinity_transformer_atom = affinity_transformer_atom
        self.affinity_pair_transformer = affinity_pair_transformer
        self.affinity_baseline = affinity_baseline
        self.affinity_confidence_different = affinity_confidence_different
        self.affinity_confidence_different_multiple = affinity_confidence_different_multiple
        self.alpha_pae = alpha_pae
        self.relative_confidence_supervision_weight = relative_confidence_supervision_weight
        self.structure_prediction_training = structure_prediction_training
        self.skip_run_structure = skip_run_structure
        self.tau_affinity_score = tau_affinity_score
        self.alpha_affinity_absolute = alpha_affinity_absolute
        self.alpha_affinity_difference = alpha_affinity_difference
        self.alpha_affinity_binary = alpha_affinity_binary
        self.alpha_affinity_score_binder_decoy = alpha_affinity_score_binder_decoy
        self.alpha_affinity_score_binder_binder = alpha_affinity_score_binder_binder
        self.alpha_affinity_focal = alpha_affinity_focal
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.compile_pairformer = compile_pairformer
        self.compile_structure = compile_structure
        self.compile_confidence = compile_confidence
        self.compile_msa = compile_msa
        self.representative_lddt = representative_lddt
        self.exclude_ions_from_lddt = exclude_ions_from_lddt
        self.ema = ema
        self.ema_decay = ema_decay
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.maximum_bond_distance = maximum_bond_distance
        self.dynamic_args = dynamic_args
        self.compute_tistogram = compute_tistogram
        self.num_tistogram_axis = num_tistogram_axis
        self.symmetry_correction_trunk = symmetry_correction_trunk
        self.fix_sym_check = fix_sym_check
        self.cyclic_pos_enc = cyclic_pos_enc
        self.trunk_resolved_loss = trunk_resolved_loss
        self.ignore_ckpt_shape_mismatch = ignore_ckpt_shape_mismatch
        self.num_distograms = num_distograms
        self.bond_type_feature = bond_type_feature
        self.use_no_atom_char = use_no_atom_char
        self.use_atom_backbone_feat = use_atom_backbone_feat
        self.use_residue_feats_atoms = use_residue_feats_atoms
        self.guidance_args = guidance_args
        self.use_templates = use_templates
        self.compile_templates = compile_templates
        self.use_token_distances = use_token_distances
        self.token_distance_args = token_distance_args
        self.predict_bfactor = predict_bfactor
        self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning
        self.use_templates_v2 = use_templates_v2
        self.freeze_template_weights = freeze_template_weights
        self.refolding_validator = refolding_validator
        self.predict_res_type = predict_res_type
        self.inverse_fold = inverse_fold
        self.inverse_fold_args = inverse_fold_args
        self.inference_logging = inference_logging
        self.use_kernels = use_kernels
    
    def to_dict(self):
        """Override to_dict to ensure all values are JSON serializable."""
        output = super().to_dict()
        return self._make_json_serializable(output)