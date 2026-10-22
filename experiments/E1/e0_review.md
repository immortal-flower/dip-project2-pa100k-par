# E0 unified evaluation review

Validation images: 10000; checkpoint best epoch: 10.
Matching attribute confusion rows: 3/26.
Maximum legacy group AttrAcc absolute difference: 0.0012999177.
Independent NumPy recomputation of exported prediction counts: passed, all 26 attributes.

| Metric | Value (%) |
| --- | ---: |
| Legacy mean group AttrAcc | 92.3583 |
| Macro mA | 76.9077 |
| Macro precision | 71.2962 |
| Macro recall | 59.6659 |
| Macro f1 | 63.1968 |
| Instance precision | 85.9498 |
| Instance recall | 82.3774 |
| Instance f1 | 83.7033 |

| Attribute | Positive support | mA (%) | P (%) | R (%) | F1 (%) | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| boots | 45 | 53.2932 | 27.2727 | 6.6667 | 10.7143 | 3 | 8 | 42 | 9947 |
| holding | 86 | 51.7291 | 50.0000 | 3.4884 | 6.5217 | 3 | 3 | 83 | 9911 |
| lower_stripe | 21 | 59.4937 | 40.0000 | 19.0476 | 25.8065 | 4 | 6 | 17 | 9973 |
| age_lt18 | 1443 | 81.0441 | 88.5024 | 63.4789 | 73.9306 | 916 | 119 | 527 | 8438 |

## Verification boundary

The archived E0 used CUDA; local review environment is recorded in verification.md (and in JSON for current evaluator versions).
Matching counts verify predictions and metric interpretation; they do not establish multi-seed robustness.
E0 has no saved training-data hash; current annotation audit documents the locally supplied files.
The historical checkpoint selection remains legacy group AttrAcc, not macro mA.

## Differences requiring review

This CPU review is provisional. Do not mix these numbers with GPU E1 runs as the final comparison.
Re-evaluate E0/E1a/E1b on the same server/software/preprocessing stack; E0 need not be retrained.
The current evidence does not identify the cause of prediction differences.

```json
{
  "male": {
    "archive": [
      4417,
      762,
      635,
      4186
    ],
    "unified": [
      4413,
      760,
      639,
      4188
    ]
  },
  "age_lt18": {
    "archive": [
      897,
      115,
      546,
      8442
    ],
    "unified": [
      916,
      119,
      527,
      8438
    ]
  },
  "age_18_60": {
    "archive": [
      8345,
      622,
      128,
      905
    ],
    "unified": [
      8339,
      603,
      134,
      924
    ]
  },
  "age_gt60": {
    "archive": [
      5,
      16,
      79,
      9900
    ],
    "unified": [
      5,
      18,
      79,
      9898
    ]
  },
  "front": {
    "archive": [
      2598,
      461,
      424,
      6517
    ],
    "unified": [
      2597,
      471,
      425,
      6507
    ]
  },
  "back": {
    "archive": [
      2772,
      335,
      396,
      6497
    ],
    "unified": [
      2773,
      335,
      395,
      6497
    ]
  },
  "side": {
    "archive": [
      3256,
      578,
      554,
      5612
    ],
    "unified": [
      3246,
      578,
      564,
      5612
    ]
  },
  "hat": {
    "archive": [
      118,
      26,
      285,
      9571
    ],
    "unified": [
      119,
      27,
      284,
      9570
    ]
  },
  "glasses": {
    "archive": [
      1188,
      455,
      679,
      7678
    ],
    "unified": [
      1187,
      455,
      680,
      7678
    ]
  },
  "short_sleeve": {
    "archive": [
      5072,
      423,
      556,
      3949
    ],
    "unified": [
      5066,
      422,
      562,
      3950
    ]
  },
  "long_sleeve": {
    "archive": [
      3954,
      556,
      418,
      5072
    ],
    "unified": [
      3949,
      564,
      423,
      5064
    ]
  },
  "upper_stripe": {
    "archive": [
      239,
      124,
      137,
      9500
    ],
    "unified": [
      239,
      121,
      137,
      9503
    ]
  },
  "upper_logo": {
    "archive": [
      823,
      338,
      448,
      8391
    ],
    "unified": [
      822,
      332,
      449,
      8397
    ]
  },
  "upper_plaid": {
    "archive": [
      505,
      407,
      503,
      8585
    ],
    "unified": [
      510,
      406,
      498,
      8586
    ]
  },
  "upper_splice": {
    "archive": [
      157,
      70,
      95,
      9678
    ],
    "unified": [
      157,
      68,
      95,
      9680
    ]
  },
  "lower_pattern": {
    "archive": [
      36,
      24,
      63,
      9877
    ],
    "unified": [
      36,
      25,
      63,
      9876
    ]
  },
  "long_coat": {
    "archive": [
      293,
      107,
      368,
      9232
    ],
    "unified": [
      287,
      103,
      374,
      9236
    ]
  },
  "trousers": {
    "archive": [
      6917,
      341,
      418,
      2324
    ],
    "unified": [
      6921,
      337,
      414,
      2328
    ]
  },
  "shorts": {
    "archive": [
      1389,
      298,
      311,
      8002
    ],
    "unified": [
      1389,
      296,
      311,
      8004
    ]
  },
  "skirt_dress": {
    "archive": [
      656,
      345,
      309,
      8690
    ],
    "unified": [
      656,
      339,
      309,
      8696
    ]
  },
  "backpack": {
    "archive": [
      1876,
      279,
      726,
      7119
    ],
    "unified": [
      1881,
      292,
      721,
      7106
    ]
  },
  "shoulder_bag": {
    "archive": [
      963,
      508,
      1093,
      7436
    ],
    "unified": [
      962,
      509,
      1094,
      7435
    ]
  },
  "handbag": {
    "archive": [
      1001,
      571,
      969,
      7459
    ],
    "unified": [
      1000,
      565,
      970,
      7465
    ]
  }
}
```
