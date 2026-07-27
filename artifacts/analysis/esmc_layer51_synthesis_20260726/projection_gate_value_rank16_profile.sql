SELECT block, produced_state, family, mean_overlap, minimum_overlap, maximum_overlap FROM read_csv_auto('report_tables/projection_gate_value_rank16_profile.csv') ORDER BY family, block;
