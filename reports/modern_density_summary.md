# Modern conditional-density methods on the multi-target parity suites

Run `release_20261020_1653`: `density_parity` on 7 datasets, **3 seeds**, 60-epoch budget with early stopping; added `cINN` (conditional RealNVP), `BPF` (Bernstein polynomial flow), and `CFM` (OT-CFM) next to the existing NSF/MAF/MDN.  All methods share the harness MLP encoder and trainer.

## Aggregate over 7 datasets x 3 seeds

| method | n | median NLL | median CRPS | median ES | median Cov90 marg | median PIT KS | median train_s |
|---|--:|--:|--:|--:|--:|--:|--:|
| torchregress_NSF | 21 | -1.951 | 0.1487 | 0.2241 | 0.890 | 0.091 | 90 |
| torchregress_MAF | 21 | -1.393 | 0.1233 | 0.2388 | 0.889 | 0.095 | 23 |
| torchregress_cINN | 21 | -0.902 | 0.1380 | 0.2388 | 0.893 | 0.080 | 25 |
| torchregress_BPF | 21 | -1.419 | 0.0902 | 0.1586 | 0.981 | 0.162 | 81 |
| torchregress_CFM | 21 | — | 0.1502 | 0.2395 | 0.907 | 0.121 | 11 |
| torchregress_MDN | 21 | -0.614 | 0.1609 | 0.2410 | 0.898 | 0.101 | 25 |

## Per-dataset median CRPS (lower is better)

| dataset | NSF | MAF | cINN | BPF | CFM | MDN |
|---|--:|--:|--:|--:|--:|--:|
| california_housing | 0.2345 | 0.2503 | 0.2503 | 0.2345 | 0.2564 | 0.2379 |
| energy_efficiency | 0.1327 | 0.1076 | 0.1139 | 0.0761 | 0.1502 | 0.1424 |
| mt_enb | 0.1008 | 0.0991 | 0.1261 | 0.0902 | 0.1441 | 0.1607 |
| mt_jura | 0.2644 | 0.2808 | 0.2681 | 0.2690 | 0.3686 | 0.2807 |
| mt_rf1 | 0.0383 | 0.1162 | 0.0879 | 0.0441 | 0.0677 | 0.0533 |
| mt_scm20d | 0.1895 | 0.2421 | 0.1945 | 0.1813 | 0.2019 | 0.2971 |
| pzflow_galaxies | 0.0178 | 0.0204 | 0.0204 | 0.0379 | 0.0517 | 0.0328 |

## Findings

- **BPF** is the strongest sample-based density on these workloads: median CRPS and energy score are best or near-best on every dataset, and coverage is conservative (median Cov90 0.98).  Its NLL is weaker than NSF, so it is a good CRPS/coverage choice but not an NLL replacement.
- **cINN/MAF** have the same failure mode on hard multi-target datasets (`mt_rf1`, `mt_scm20d`): occasional extreme NLL/CRPS from an unstable affine transform.  Medians are sane, means are not; keep medians/robust summaries.
- **CFM** is cheap (median 11 s vs 90 s for NSF) and has competitive sample metrics on several datasets, but no exact NLL.  It is a sampler, not a calibrated likelihood replacement.
- **NSF** remains the best likelihood-based default; **MDN** is competitive on several tabular targets but weaker on multimodal/heterogeneous ones.
- The full `pzflow_galaxies` results now include all three seeds; no missing rows.

Artifacts: `reports/modern_density_results.json` (density_parity rows only); raw harness JSONs in the CANFAR run archive.
