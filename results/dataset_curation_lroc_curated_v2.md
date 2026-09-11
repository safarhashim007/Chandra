# Scientific LROC curation — lroc_curated_v2

- run_id: lroc_curated_v2
- products_discovered: 44
- scientific_products: 44
- browse_only_products: 0
- regions: 11
- train_regions: 7
- train_candidate_pairs: 64
- dense_gt_valid_pairs: 25
- rejected_pairs: 39
- selected_pairs: 25
- selected_images: 20
- selected_regions: 5
- validation_used_for_training: 0
- test_used_for_training: 0
- t0_used_for_training: 0
- manifest: runs/lroc_curated_v2/selected_training_data.json
- parquet_manifest: runs/lroc_curated_v2/selected_training_data.parquet
- checksum_path: runs/lroc_curated_v2/selected_training_data.sha256
- sha256: 80641bb801a51e628dbce5d5601e6755fc899f4fcea611e9271694ca7d884a99
- region_pair_yield: {'E011N1499': {'candidate_positive_pairs': 1, 'dense_gt_valid_pairs': 0, 'selected_pairs': 0, 'pair_yield': 0.0}, 'E018N3346': {'candidate_positive_pairs': 15, 'dense_gt_valid_pairs': 12, 'selected_pairs': 12, 'pair_yield': 0.8}, 'E041N1229': {'candidate_positive_pairs': 1, 'dense_gt_valid_pairs': 1, 'selected_pairs': 1, 'pair_yield': 1.0}, 'E090S0155': {'candidate_positive_pairs': 15, 'dense_gt_valid_pairs': 6, 'selected_pairs': 6, 'pair_yield': 0.4}, 'E186N1213': {'candidate_positive_pairs': 2, 'dense_gt_valid_pairs': 2, 'selected_pairs': 2, 'pair_yield': 1.0}, 'E199N0308': {'candidate_positive_pairs': 15, 'dense_gt_valid_pairs': 4, 'selected_pairs': 4, 'pair_yield': 0.26666666666666666}, 'E207N3357': {'candidate_positive_pairs': 15, 'dense_gt_valid_pairs': 0, 'selected_pairs': 0, 'pair_yield': 0.0}}
- rejection_reasons: {'no_valid_dense_gt': 35, 'accepted': 25, 'cycle_or_coverage_failure': 4}
- selected_fraction_by_region: {'E018N3346': 0.48, 'E041N1229': 0.04, 'E090S0155': 0.24, 'E186N1213': 0.08, 'E199N0308': 0.16}
- max_selected_region_fraction: 0.48
- data_curation_gate: PASS
- checkpoint_provenance: None

## Leakage

VALIDATION USED FOR TRAINING: 0
TEST USED FOR TRAINING: 0
T0 USED FOR TRAINING: 0

## Region pair yield

- E011N1499: 0/1 valid, 0 selected, yield 0.0%
- E018N3346: 12/15 valid, 12 selected, yield 80.0%
- E041N1229: 1/1 valid, 1 selected, yield 100.0%
- E090S0155: 6/15 valid, 6 selected, yield 40.0%
- E186N1213: 2/2 valid, 2 selected, yield 100.0%
- E199N0308: 4/15 valid, 4 selected, yield 26.7%
- E207N3357: 0/15 valid, 0 selected, yield 0.0%
