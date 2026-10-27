# E0 / E1a / E1b unified validation results

All percentages; best checkpoints selected by legacy mean group AttrAcc.

| Experiment | Best epoch | Legacy group AttrAcc | Macro mA | Macro P | Macro R | Macro F1 | Instance P | Instance R | Instance F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 10 | 92.36 | 76.87 | 71.36 | 59.65 | 63.17 | 85.93 | 82.39 | 83.70 |
| E1a | 14 | 90.76 | 78.38 | 61.07 | 65.45 | 61.00 | 79.34 | 83.29 | 80.78 |
| E1b | 6 | 92.17 | 75.85 | 70.40 | 56.94 | 61.47 | 86.74 | 80.70 | 83.13 |

## Rare attributes and age control

| Attribute | Experiment | Positives | AttrAcc | mA | P | R | F1 | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| boots | E0 | 45 | 99.50 | 53.29 | 27.27 | 6.67 | 10.71 | 3 | 8 | 42 | 9947 |
| boots | E1a | 45 | 98.96 | 57.45 | 9.59 | 15.56 | 11.86 | 7 | 66 | 38 | 9889 |
| boots | E1b | 45 | 99.21 | 62.00 | 19.64 | 24.44 | 21.78 | 11 | 45 | 34 | 9910 |
| holding | E0 | 86 | 99.14 | 51.73 | 50.00 | 3.49 | 6.52 | 3 | 3 | 83 | 9911 |
| holding | E1a | 86 | 99.14 | 57.49 | 50.00 | 15.12 | 23.21 | 13 | 13 | 73 | 9901 |
| holding | E1b | 86 | 99.11 | 51.71 | 33.33 | 3.49 | 6.32 | 3 | 6 | 83 | 9908 |
| lower_stripe | E0 | 21 | 99.77 | 59.49 | 40.00 | 19.05 | 25.81 | 4 | 6 | 17 | 9973 |
| lower_stripe | E1a | 21 | 99.35 | 61.66 | 9.26 | 23.81 | 13.33 | 5 | 49 | 16 | 9930 |
| lower_stripe | E1b | 21 | 99.79 | 50.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 21 | 9979 |
| age_lt18 | E0 | 1443 | 93.39 | 80.41 | 88.64 | 62.16 | 73.08 | 897 | 115 | 546 | 8442 |
| age_lt18 | E1a | 1443 | 91.58 | 72.35 | 92.50 | 45.32 | 60.84 | 654 | 53 | 789 | 8504 |
| age_lt18 | E1b | 1443 | 95.20 | 86.85 | 89.96 | 75.12 | 81.87 | 1084 | 121 | 359 | 8436 |

## Provenance

- E0: checkpoint SHA-256 `991891f5b367a2a51cefaffd1d647e5e9861b225884ac78122f89d1da778c806`; labels `c99e7cc03fab148b932d4c5e66292c5e3008b648a714a451c98ecf4d4595a88c`.
- E1a: checkpoint SHA-256 `19affea68f877b29a5ebccf810e145012548a6dfccca98020ab0dea91e67565b`; labels `c99e7cc03fab148b932d4c5e66292c5e3008b648a714a451c98ecf4d4595a88c`.
- E1b: checkpoint SHA-256 `d2fa32ee72fd83164422bdeece6a0096de12b4dfd951405cab87f152fc7ec7e5`; labels `c99e7cc03fab148b932d4c5e66292c5e3008b648a714a451c98ecf4d4595a88c`.

A single seed is descriptive evidence, not statistical significance. Examine both recall and false positives.
age_lt18 uses unchanged CE; differences can arise indirectly through the shared backbone.
