SELECT checkpoint, checkpoint_label, state_index, producing_block, mixing_weight_pct FROM read_csv_auto('report_tables/mixer_profile.csv') ORDER BY checkpoint, state_index;
