SELECT block, produced_state, threshold, head_count FROM read_csv_auto('report_tables/attention_head_threshold_profile.csv') ORDER BY threshold, block;
