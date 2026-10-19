# PZ training-removal (paper-style paired test)

Same held-out test galaxies; training set reduced by magnitude, colour or sky-quadrant removal.

| dataset | method | scenario | removed | RMSE | sigma_NMAD | coverage90 | coverage68 |
|---|---|---|--:|--:|--:|--:|--:|
| pz_challenge_taskset4_cardinal_10yr | MSE_MLP | full | 0.00 | 0.1782 | 0.0648 | — | — |
| pz_challenge_taskset4_cardinal_10yr | GaussianNLL_MLP | full | 0.00 | 0.2289 | 0.0527 | 0.9087675892995206 | 0.7146281119529921 |
| pz_challenge_taskset4_cardinal_10yr | MDN_K3 | full | 0.00 | 0.2171 | 0.0490 | 0.8948507808875831 | 0.6806865625483223 |
| pz_challenge_taskset4_cardinal_10yr | TabM_MSE | full | 0.00 | 0.1608 | 0.0584 | — | — |
| pz_challenge_taskset4_cardinal_10yr | SplitCP_MLP | full | 0.00 | 0.1808 | 0.0702 | 0.8923766816143498 | — |
| pz_challenge_taskset4_cardinal_10yr | MSE_MLP | omit_faint | 0.33 | 0.6782 | 0.0769 | — | — |
| pz_challenge_taskset4_cardinal_10yr | GaussianNLL_MLP | omit_faint | 0.33 | 0.7970 | 0.0763 | 0.7862223596721818 | 0.5896860986547086 |
| pz_challenge_taskset4_cardinal_10yr | MDN_K3 | omit_faint | 0.33 | 0.7728 | 0.0654 | 0.8292098345446111 | 0.6193752899335085 |
| pz_challenge_taskset4_cardinal_10yr | TabM_MSE | omit_faint | 0.33 | 0.6231 | 0.0699 | — | — |
| pz_challenge_taskset4_cardinal_10yr | SplitCP_MLP | omit_faint | 0.33 | 0.6588 | 0.0787 | 0.7027215092005566 | — |
| pz_challenge_taskset4_cardinal_10yr | MSE_MLP | omit_red | 0.20 | 0.1930 | 0.0820 | — | — |
| pz_challenge_taskset4_cardinal_10yr | GaussianNLL_MLP | omit_red | 0.20 | 0.2344 | 0.0608 | 0.8677903200865935 | 0.6761249420132983 |
| pz_challenge_taskset4_cardinal_10yr | MDN_K3 | omit_red | 0.20 | 0.2120 | 0.0539 | 0.8755218803154476 | 0.6583423534869337 |
| pz_challenge_taskset4_cardinal_10yr | TabM_MSE | omit_red | 0.20 | 0.1760 | 0.0705 | — | — |
| pz_challenge_taskset4_cardinal_10yr | SplitCP_MLP | omit_red | 0.20 | 0.1939 | 0.0789 | 0.8878150610793258 | — |
| pz_challenge_taskset4_cardinal_10yr | MSE_MLP | omit_field | 0.28 | 0.1898 | 0.0673 | — | — |
| pz_challenge_taskset4_cardinal_10yr | GaussianNLL_MLP | omit_field | 0.28 | 0.2406 | 0.0558 | 0.9011906602752435 | 0.6978506262563785 |
| pz_challenge_taskset4_cardinal_10yr | MDN_K3 | omit_field | 0.28 | 0.2309 | 0.0513 | 0.887119220658729 | 0.6734188959331994 |
| pz_challenge_taskset4_cardinal_10yr | TabM_MSE | omit_field | 0.28 | 0.1735 | 0.0605 | — | — |
| pz_challenge_taskset4_cardinal_10yr | SplitCP_MLP | omit_field | 0.28 | 0.1964 | 0.0720 | 0.8870419050564404 | — |
| pz_challenge_cardinal_10yr | MSE_MLP | full | 0.00 | 0.0996 | 0.0757 | — | — |
| pz_challenge_cardinal_10yr | GaussianNLL_MLP | full | 0.00 | 0.1438 | 0.0694 | 0.8997 | 0.69295 |
| pz_challenge_cardinal_10yr | MDN_K3 | full | 0.00 | 0.1213 | 0.0652 | 0.89195 | 0.67775 |
| pz_challenge_cardinal_10yr | TabM_MSE | full | 0.00 | 0.0922 | 0.0675 | — | — |
| pz_challenge_cardinal_10yr | SplitCP_MLP | full | 0.00 | 0.1087 | 0.0792 | 0.89925 | — |
| pz_challenge_cardinal_10yr | MSE_MLP | omit_faint | 0.33 | 0.1589 | 0.0912 | — | — |
| pz_challenge_cardinal_10yr | GaussianNLL_MLP | omit_faint | 0.33 | 0.2194 | 0.0812 | 0.85095 | 0.64015 |
| pz_challenge_cardinal_10yr | MDN_K3 | omit_faint | 0.33 | 0.1579 | 0.0715 | 0.85065 | 0.6283 |
| pz_challenge_cardinal_10yr | TabM_MSE | omit_faint | 0.33 | 0.1418 | 0.0783 | — | — |
| pz_challenge_cardinal_10yr | SplitCP_MLP | omit_faint | 0.33 | 0.1669 | 0.0832 | 0.82125 | — |
| pz_challenge_cardinal_10yr | MSE_MLP | omit_red | 0.20 | 0.1641 | 0.0906 | — | — |
| pz_challenge_cardinal_10yr | GaussianNLL_MLP | omit_red | 0.20 | 0.1893 | 0.0819 | 0.8548 | 0.64925 |
| pz_challenge_cardinal_10yr | MDN_K3 | omit_red | 0.20 | 0.1460 | 0.0763 | 0.858 | 0.64915 |
| pz_challenge_cardinal_10yr | TabM_MSE | omit_red | 0.20 | 0.1433 | 0.0857 | — | — |
| pz_challenge_cardinal_10yr | SplitCP_MLP | omit_red | 0.20 | 0.1623 | 0.0972 | 0.84235 | — |
| pz_challenge_cardinal_10yr | MSE_MLP | omit_field | 0.26 | 0.1162 | 0.0775 | — | — |
| pz_challenge_cardinal_10yr | GaussianNLL_MLP | omit_field | 0.26 | 0.1450 | 0.0685 | 0.89705 | 0.688 |
| pz_challenge_cardinal_10yr | MDN_K3 | omit_field | 0.26 | 0.1259 | 0.0664 | 0.8832 | 0.6639 |
| pz_challenge_cardinal_10yr | TabM_MSE | omit_field | 0.26 | 0.1055 | 0.0685 | — | — |
| pz_challenge_cardinal_10yr | SplitCP_MLP | omit_field | 0.26 | 0.1167 | 0.0810 | 0.8995 | — |
