from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric

from boltzgen_flat import data_const as const
from boltzgen_flat.data_rmsd_computation import get_true_coordinates
from boltzgen_flat.model_loss_distogram import distogram_loss

from boltzgen_flat.model_loss_validation import (
    factored_lddt_loss,
    factored_token_lddt_dist_loss,
)


class Validator(nn.Module):
    """Compute validation step and aggregation."""

    def __init__(
        self,
        val_names: List[str],
        confidence_prediction: bool = False,
        override_val_method: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.val_names = val_names

        self.override_val_method = override_val_method
        if override_val_method is not None:
            override_val_method = override_val_method.lower()
            assert override_val_method in const.method_types_ids, "Invalid method type."
            self.override_val_method = const.method_types_ids[override_val_method]

        self.num_val_datasets = num_val_datasets = len(val_names)

        msg = "Only one dataset supported for now per validator. Define multiple validators for multiple datasets."
        assert num_val_datasets == 1, msg

        ### Initialize metrics ##

        # Folding metrics
        folding_metric_labels = [
            "lddt",
            "disto_lddt",
            "complex_lddt",
            "disto_loss",
        ]
        self.fold_metrics = nn.ModuleDict(
            {
                k: nn.ModuleList([nn.ModuleDict() for _ in range(num_val_datasets)])
                for k in folding_metric_labels
            }
        )

        # Confidence metrics
        confidence_metric_prefixes = [
            "top1",
            "iplddt_top1",
            "ipde_top1",
            "pde_top1",
            "ptm_top1",
            "iptm_top1",
            "ligand_iptm_top1",
            "protein_iptm_top1",
            "avg",
        ]
        mae_metric_labels = ["plddt_mae", "pde_mae", "pae_mae"]
        lddt_confidence_metric_labels = [
            prefix + "_lddt" for prefix in confidence_metric_prefixes
        ]

        if confidence_prediction:
            self.confidence_metrics = nn.ModuleDict(
                {
                    k: nn.ModuleList([nn.ModuleDict() for _ in range(num_val_datasets)])
                    for k in lddt_confidence_metric_labels + mae_metric_labels
                }
            )

        # Initialize metrics for datasets
        for val_idx in range(num_val_datasets):
            for m_ in [
                *const.out_types,
            ]:
                self.fold_metrics["disto_lddt"][val_idx][m_] = MeanMetric()
                for suffix in [":recall", ":precision", ":diversity"]:
                    m = m_ + suffix
                    self.fold_metrics["lddt"][val_idx][m] = MeanMetric()
                    self.fold_metrics["complex_lddt"][val_idx][m] = MeanMetric()
                    if confidence_prediction:
                        for k in lddt_confidence_metric_labels + mae_metric_labels:
                            self.confidence_metrics[k][val_idx][m_] = MeanMetric()

            for m in const.out_single_types:
                if confidence_prediction:
                    self.confidence_metrics["plddt_mae"][val_idx][m] = MeanMetric()

            for m in ["disto_loss"]:
                self.fold_metrics["disto_loss"][val_idx][m] = MeanMetric()

    def run_model(
        self, model: LightningModule, batch: Dict[str, torch.Tensor], idx_dataset: int
    ) -> Dict[str, torch.Tensor]:
        """Compute the forward pass."""
        if self.override_val_method is not None:
            new_feature = batch["method_feature"] * 0 + self.override_val_method
            batch["method_feature"] = new_feature

        out = model(
            batch,
            recycling_steps=model.validation_args.recycling_steps,
            num_sampling_steps=model.validation_args.sampling_steps,
            diffusion_samples=model.validation_args.diffusion_samples,
            run_confidence_sequentially=model.validation_args.get(
                "run_confidence_sequentially", False
            ),
        )

        return out

    # @abstractmethod
    def process(
        self,
        model: LightningModule,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx_dataset: int,
        dataloader_idx: int,
        n_samples: int,
    ) -> None:
        """Run a validation step.

        Parameters
        ----------
        model : LightningModule
            The LightningModule model.
        batch : Dict[str, torch.Tensor]
            The batch input.
        out : Dict[str, torch.Tensor]
            The output of the model.

        """
        raise NotImplementedError

    def get_local_val_index(self, model: LightningModule, idx_dataset: int) -> int:
        """Get the local validation index.

        Parameters
        ----------
        idx_dataset : int
            The dataset index.

        Returns
        -------
        int
            The local validation index.
        """
        val_name = model.val_group_mapper[idx_dataset]["label"]
        return self.val_names.index(val_name)

    def compute_disto_loss(
        self,
        model: LightningModule,
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        idx_dataset: int,
    ) -> None:
        """Compute distogram loss."""
        # Compute validation disto loss
        val_disto_loss, _ = distogram_loss(
            out, batch, aggregate_distogram=model.aggregate_distogram
        )

        return val_disto_loss

    def compute_disto_lddt(self, model, batch, out, idx_dataset) -> Tuple[Dict, Dict]:
        """Compute distogram lddt."""
        boundaries = torch.linspace(model.min_dist, model.max_dist, model.num_bins - 1)
        lower = torch.tensor([1.0])
        upper = torch.tensor([model.max_dist + 5.0])
        exp_boundaries = torch.cat((lower, boundaries, upper))
        mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
            out["pdistogram"]
        )

        # Compute true distogram
        K = batch["coords"].shape[1]
        true_center = batch["disto_coords"].reshape(K, -1, 3)  # (K, L, 3)

        batch["token_disto_mask"] = batch["token_disto_mask"]

        # Compute distogram lddt by looping over predicted distograms
        disto_lddt_dict = defaultdict(lambda: torch.zeros(K, 1).to(model.device))
        disto_total_dict = defaultdict(lambda: torch.zeros(K, 1).to(model.device))

        # Compute predicted dists
        preds = out["pdistogram"][:, :, :, 0]
        pred_softmax = torch.softmax(preds, dim=-1)
        pred_softmax = pred_softmax.argmax(dim=-1)
        pred_softmax = torch.nn.functional.one_hot(
            pred_softmax, num_classes=preds.shape[-1]
        )
        pred_dist_i = (pred_softmax * mid_points).sum(dim=-1)  # (B, L, L)
        del pred_softmax

        # Compute true distances for each conformer
        # Implemented in a loop to avoid memory issues with large number of
        # conformers. Batched version over K factored_token_lddt_dist_loss_ensemble
        # more efficient for small K.
        for k in range(K):
            true_dists_k = torch.cdist(true_center[k], true_center[k])[
                None
            ]  # (1, L * L)

            # Compute lddt
            disto_lddt_dict_, disto_total_dict_ = factored_token_lddt_dist_loss(
                feats=batch,
                true_d=true_dists_k,
                pred_d=pred_dist_i,
            )

        for key in disto_lddt_dict_:
            disto_lddt_dict[key][k, 0] = disto_lddt_dict_[key].item()
            disto_total_dict[key][k, 0] = disto_total_dict_[key].item()

        for key in disto_lddt_dict:
            # Take min over distograms and average over conformers. Add batch dimension.
            disto_lddt_dict[key] = (
                disto_lddt_dict[key].min(dim=1).values.mean(dim=0)[None]
            )
            disto_total_dict[key] = (
                disto_total_dict[key].min(dim=1).values.mean(dim=0)[None]
            )
        del true_center
        del preds

        return disto_lddt_dict, disto_total_dict

    def get_lddt_metrics(
        self,
        model,
        batch,
        out,
        idx_dataset,
        n_samples,
        true_coords_resolved_mask,
        true_coords,
        expand_to_diffusion_samples,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        K = batch["coords"].shape[1]

        if not expand_to_diffusion_samples:
            true_coords_resolved_mask = true_coords_resolved_mask.unsqueeze(0).repeat(
                (n_samples, 1)
            )

        ### Compute lddt ###
        # Implemented in a loop to avoid memory issues with large number
        # of conformers
        all_lddt_dict = defaultdict(list)
        all_total_dict = defaultdict(list)
        for ensemble_idx in range(K):
            # This OOM for large n_samples. Need to chunk or loop over samples.

            if expand_to_diffusion_samples:
                true_coords_k = true_coords[:, ensemble_idx]
            else:
                true_coords_k = (
                    true_coords[ensemble_idx].unsqueeze(0).repeat((n_samples, 1, 1))
                )

            all_lddt_dict_s, all_total_dict_s = factored_lddt_loss(
                feats=batch,
                atom_mask=true_coords_resolved_mask,
                true_atom_coords=true_coords_k,  # (multiplicity, L, 3)
                pred_atom_coords=out["sample_atom_coords"],
                multiplicity=n_samples,
                exclude_ions=model.exclude_ions_from_lddt,
            )
            for key in all_lddt_dict_s:
                all_lddt_dict[key].append(all_lddt_dict_s[key])
                all_total_dict[key].append(all_total_dict_s[key])

        for key in all_lddt_dict:
            all_lddt_dict[key] = torch.stack(
                all_lddt_dict[key], dim=1
            )  # (multiplicity, K)
            all_total_dict[key] = torch.stack(all_total_dict[key], dim=1)
        return all_lddt_dict, all_total_dict

    def compute_best_lddt_metrics(
        self,
        model,
        batch,
        all_lddt_dict,
        all_total_dict,
        n_samples,
    ):
        K = batch["coords"].shape[1]

        # if the multiplicity used is > 1 then we take the best lddt of the different samples
        # AF3 combines this with the confidence based filtering
        best_lddt_dict, best_total_dict = {}, {}
        best_complex_lddt_dict, best_complex_total_dict = {}, {}
        # B = true_coords.shape[0] // n_samples
        if n_samples > 1:
            # NOTE: we can change the way we aggregate the lddt
            complex_total = 0
            complex_lddt = 0
            for key in all_lddt_dict:
                complex_lddt += all_lddt_dict[key] * all_total_dict[key]
                complex_total += all_total_dict[key]
            complex_lddt /= complex_total + 1e-7

            # Take best over samples, average over conformers: recall groundtruth
            # conformers
            suffix = ":recall"
            best_complex_idx = complex_lddt.argmax(dim=0)
            for key in all_lddt_dict:
                # take best across diffusion samples
                best_idx = all_lddt_dict[key].argmax(dim=0)
                best_lddt_dict[key + suffix] = all_lddt_dict[key][
                    best_idx, torch.arange(K)
                ].mean(dim=0)[None]  # take average across conformers in ensemble
                best_total_dict[key + suffix] = all_total_dict[key][
                    best_idx, torch.arange(K)
                ].mean(dim=0)[None]
                # mean(dim=0) since samples was argmaxed, add back batch dim
                best_complex_lddt_dict[key + suffix] = all_lddt_dict[key][
                    best_complex_idx, torch.arange(K)
                ].mean(dim=0)[None]
                best_complex_total_dict[key + suffix] = all_total_dict[key][
                    best_complex_idx, torch.arange(K)
                ].mean(dim=0)[None]  # sum here ?

            # Take best over conformers, average over samples: precision
            suffix = ":precision"
            best_complex_idx = complex_lddt.argmax(dim=1)
            for key in all_lddt_dict:
                # take best across diffusion conformers
                best_idx = all_lddt_dict[key].argmax(dim=1)
                best_lddt_dict[key + suffix] = all_lddt_dict[key][
                    torch.arange(n_samples), best_idx
                ].mean(dim=0)[None]  # take average across samples in ensemble
                best_total_dict[key + suffix] = all_total_dict[key][
                    torch.arange(n_samples), best_idx
                ].mean(dim=0)[None]
                # dim 0 since samples was argmaxed, add back batch dim [None]
                best_complex_lddt_dict[key + suffix] = all_lddt_dict[key][
                    torch.arange(n_samples), best_complex_idx
                ].mean(dim=0)[None]
                best_complex_total_dict[key + suffix] = all_total_dict[key][
                    torch.arange(n_samples), best_complex_idx
                ].mean(dim=0)[None]

        else:
            # Take average across conformers in ensemble.
            for key in all_lddt_dict:
                for suffix in [":recall", ":precision"]:
                    best_lddt_dict[key + suffix] = (
                        all_lddt_dict[key].max(dim=1).values[None]
                    )  # (sample, K) -> (B=1, samples=1)
                    best_total_dict[key + suffix] = (
                        all_total_dict[key].max(dim=1).values[None]
                    )
                    best_complex_lddt_dict[key + suffix] = (
                        all_lddt_dict[key].max(dim=1).values[None]
                    )
                    best_complex_total_dict[key + suffix] = (
                        all_total_dict[key].max(dim=1).values[None]
                    )
                suffix = ":diversity"
                best_lddt_dict[key + suffix] = torch.tensor([0.0]).to(model.device)
                best_total_dict[key + suffix] = torch.tensor([1.0]).to(model.device)
                best_complex_lddt_dict[key + suffix] = torch.tensor([0.0]).to(
                    model.device
                )
                best_complex_total_dict[key + suffix] = torch.tensor([1.0]).to(
                    model.device
                )

        return (
            best_lddt_dict,
            best_total_dict,
            best_complex_lddt_dict,
            best_complex_total_dict,
        )

    def update_lddt_rmsd_metrics(
        self,
        batch,
        disto_lddt_dict,
        disto_total_dict,
        best_lddt_dict,
        best_total_dict,
        best_complex_lddt_dict,
        best_complex_total_dict,
        idx_dataset,
        return_dict,
    ):
        # Folding metrics
        for m_ in const.out_types:
            if m_ == "ligand_protein":
                self.fold_metrics["disto_lddt"][idx_dataset]["ligand_protein"].update(
                    disto_lddt_dict[m_], disto_total_dict[m_]
                )
                for suffix in [":recall", ":precision", ":diversity"]:
                    m = m_ + suffix
                    self.fold_metrics["lddt"][idx_dataset][
                        "ligand_protein" + suffix
                    ].update(best_lddt_dict[m], best_total_dict[m])
                    self.fold_metrics["complex_lddt"][idx_dataset][
                        "ligand_protein" + suffix
                    ].update(best_complex_lddt_dict[m], best_complex_total_dict[m])

            elif m_ == "protein_protein":
                self.fold_metrics["disto_lddt"][idx_dataset]["protein_protein"].update(
                    disto_lddt_dict[m_], disto_total_dict[m_]
                )
                for suffix in [":recall", ":precision", ":diversity"]:
                    m = m_ + suffix
                    self.fold_metrics["lddt"][idx_dataset][
                        "protein_protein" + suffix
                    ].update(best_lddt_dict[m], best_total_dict[m])
                    self.fold_metrics["complex_lddt"][idx_dataset][
                        "protein_protein" + suffix
                    ].update(best_complex_lddt_dict[m], best_complex_total_dict[m])

            else:
                self.fold_metrics["disto_lddt"][idx_dataset][m_].update(
                    disto_lddt_dict[m_], disto_total_dict[m_]
                )
                for suffix in [":recall", ":precision", ":diversity"]:
                    m = m_ + suffix
                    self.fold_metrics["lddt"][idx_dataset][m].update(
                        best_lddt_dict[m], best_total_dict[m]
                    )
                    self.fold_metrics["complex_lddt"][idx_dataset][m].update(
                        best_complex_lddt_dict[m], best_complex_total_dict[m]
                    )

    def common_val_step(
        self,
        model: LightningModule,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx_dataset: int,
        expand_to_diffusion_samples: bool = True,
        symmetry_correction: bool = None,
    ) -> None:
        """Run a common validation step.

        Parameters
        ----------
        model : LightningModule
            The LightningModule model.
        batch : Dict[str, torch.Tensor]
            The batch input.
        out : Dict[str, torch.Tensor]
            The output of the model.
        """
        if symmetry_correction is None:
            symmetry_correction = model.val_group_mapper[idx_dataset][
                "symmetry_correction"
            ]  # global val index

        # Get the local validation index from the global index
        idx_dataset = self.get_local_val_index(model, idx_dataset)

        n_samples = model.validation_args.diffusion_samples

        # Compute distogram loss and update metrics

        val_disto_loss = self.compute_disto_loss(model, out, batch, idx_dataset)

        # Compute distogram lddt and update metrics

        disto_lddt_dict, disto_total_dict = self.compute_disto_lddt(
            model, batch, out, idx_dataset
        )

        # Get true coords

        return_dict = get_true_coordinates(
            batch,
            out,
            n_samples,
            symmetry_correction,
            expand_to_diffusion_samples=expand_to_diffusion_samples,
        )

        # Move this and do better as to when to interleave
        true_coords = return_dict[
            "true_coords"
        ]  # (multiplicity, K, L, 3) if expand_to_diffusion_samples else (K, L, 3)
        true_coords_resolved_mask = return_dict[
            "true_coords_resolved_mask"
        ]  # (multiplicity, L) if expand_to_diffusion_samples else (L)

        # Get lddt metrics

        all_lddt_dict, all_total_dict = self.get_lddt_metrics(
            model,
            batch,
            out,
            idx_dataset,
            n_samples,
            true_coords_resolved_mask,
            true_coords,
            expand_to_diffusion_samples,
        )

        # Compute best lddt metrics based on oracle lddt and
        # average across conformers
        (
            best_lddt_dict,
            best_total_dict,
            best_complex_lddt_dict,
            best_complex_total_dict,
        ) = self.compute_best_lddt_metrics(
            model,
            batch,
            all_lddt_dict,
            all_total_dict,
            n_samples,
        )

        ### Update the metrics ###

        # Update distogram loss
        self.fold_metrics["disto_loss"][idx_dataset]["disto_loss"].update(
            val_disto_loss
        )

        # Update folding metrics
        self.update_lddt_rmsd_metrics(
            batch,
            disto_lddt_dict,
            disto_total_dict,
            best_lddt_dict,
            best_total_dict,
            best_complex_lddt_dict,
            best_complex_total_dict,
            # rmsds,
            idx_dataset,
            return_dict,
        )

    def on_epoch_end(self, model: LightningModule):
        raise NotImplementedError

    def common_on_epoch_end(self, model: LightningModule, logname: str = "val"):
        avg_lddt = [{} for _ in range(self.num_val_datasets)]
        avg_disto_lddt = [{} for _ in range(self.num_val_datasets)]

        for idx_dataset in range(self.num_val_datasets):
            dataset_name_ori = self.val_names[idx_dataset]

            # this is harcodeded for now to compare with Boltz-1 metrics
            dataset_name = (
                ""
                if dataset_name_ori == "RCSB" or dataset_name_ori == "monomer"
                else f"__{dataset_name_ori}"
            )

            for m_ in [
                *const.out_types,
            ]:
                avg_disto_lddt[idx_dataset][m_] = self.fold_metrics["disto_lddt"][
                    idx_dataset
                ][m_].compute()

                avg_disto_lddt[idx_dataset][m_] = (
                    0.0
                    if torch.isnan(avg_disto_lddt[idx_dataset][m_])
                    else avg_disto_lddt[idx_dataset][m_].item()
                )
                self.fold_metrics["disto_lddt"][idx_dataset][m_].reset()
                model.log(
                    f"{logname}/disto_lddt_{m_}{dataset_name}",
                    avg_disto_lddt[idx_dataset][m_],
                )

                m = m_ + ":recall"
                avg_lddt[idx_dataset][m] = self.fold_metrics["lddt"][idx_dataset][
                    m
                ].compute()
                avg_lddt[idx_dataset][m] = (
                    0.0
                    if torch.isnan(avg_lddt[idx_dataset][m])
                    else avg_lddt[idx_dataset][m].item()
                )
                self.fold_metrics["lddt"][idx_dataset][m].reset()
                model.log(
                    f"{logname}/lddt_{m_}{dataset_name}",
                    avg_lddt[idx_dataset][m],
                )

            overall_disto_lddt = sum(
                avg_disto_lddt[idx_dataset][m] * w
                for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())
            model.log(
                f"{logname}/disto_lddt{dataset_name}",
                overall_disto_lddt,
            )

            overall_lddt = sum(
                avg_lddt[idx_dataset][m + ":recall"] * w
                for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())

            model.log(
                f"{logname}/lddt",
                overall_lddt,
            )

            # Distogram loss
            r = self.fold_metrics["disto_loss"][idx_dataset]["disto_loss"].compute()
            model.log(f"{logname}/disto_loss{dataset_name}", r)
            self.fold_metrics["disto_loss"][idx_dataset]["disto_loss"].reset()
