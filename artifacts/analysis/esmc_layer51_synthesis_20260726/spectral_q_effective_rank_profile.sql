SELECT block, produced_state, family, effective_rank, depth_residual, local_robust_z, bh_q FROM read_csv_auto('report_tables/spectral_q_effective_rank_profile.csv') ORDER BY block;
