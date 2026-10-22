# E0 unified evaluation review

Validation images: 10000; checkpoint best epoch: 10.
Matching attribute confusion rows: 26/26.
Maximum legacy group AttrAcc absolute difference: 0.0000000000.
Independent NumPy recomputation of exported prediction counts: passed, all 26 attributes.

| Metric | Value (%) |
| --- | ---: |
| Legacy mean group AttrAcc | 92.3576 |
| Macro mA | 76.8736 |
| Macro precision | 71.3556 |
| Macro recall | 59.6454 |
| Macro f1 | 63.1667 |
| Instance precision | 85.9305 |
| Instance recall | 82.3885 |
| Instance f1 | 83.6997 |

| Attribute | Positive support | mA (%) | P (%) | R (%) | F1 (%) | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| boots | 45 | 53.2932 | 27.2727 | 6.6667 | 10.7143 | 3 | 8 | 42 | 9947 |
| holding | 86 | 51.7291 | 50.0000 | 3.4884 | 6.5217 | 3 | 3 | 83 | 9911 |
| lower_stripe | 21 | 59.4937 | 40.0000 | 19.0476 | 25.8065 | 4 | 6 | 17 | 9973 |
| age_lt18 | 1443 | 80.4091 | 88.6364 | 62.1622 | 73.0754 | 897 | 115 | 546 | 8442 |

## Verification boundary

The archived E0 used CUDA; local review environment is recorded in verification.md (and in JSON for current evaluator versions).
Matching counts verify predictions and metric interpretation; they do not establish multi-seed robustness.
E0 has no saved training-data hash; current annotation audit documents the locally supplied files.
The historical checkpoint selection remains legacy group AttrAcc, not macro mA.
