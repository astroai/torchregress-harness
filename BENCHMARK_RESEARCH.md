# Comparative benchmark research (2026-10-17)

Scope: established papers/packages to compare against torchregress, and the
real/synthetic datasets their authors actually used. Decisions for the CANFAR
GPU campaign are recorded at the end.

## 1. What the original papers evaluate

| Method | Paper | Data reported in the paper |
|---|---|---|
| Neural ratio estimation (NRE-A/B) | Cranmer, Pavez & Louppe 2015, [arXiv:1506.02169](https://arxiv.org/abs/1506.02169) | Synthetic only: 1-D three-component Gaussian mixture (signal fraction) and a 5-D synthetic generator; LHC/Higgs discussed, not run. |
| Contrastive likelihood-free inference / NRE-B | Durkan, Murray & Papamakarios 2020, [arXiv:2002.03712](https://arxiv.org/abs/2002.03712), [code](https://github.com/conormdurkan/lfi) | Three simulators: SLCP (nonlinear Gaussian), Lotka-Volterra, M/G/1 queue. |
| CNRE / NRE-C | Miller, Weniger & Forré 2022, [arXiv:2210.06170](https://arxiv.org/abs/2210.06170), [code](https://github.com/bkmi/cnre) | Three tractable-likelihood simulators plus the sbibm benchmark (10 tasks). |
| NPE / SNPE | Papamakarios & Murray 2016, [arXiv:1605.06376](https://arxiv.org/abs/1605.06376); Greenberg et al. 2019, [arXiv:1905.07488](https://arxiv.org/abs/1905.07488) | Toy Gaussian, Bayesian linear regression, Lotka-Volterra, M/G/1; Two Moons, SLCP, SLCP+distractors. |
| sbibm benchmark | Lueckmann et al. 2021, [arXiv:2101.04653](https://arxiv.org/abs/2101.04653), [code](https://github.com/sbi-benchmark/sbibm) | Gaussian Linear, Gaussian Linear Uniform, SLCP, SLCP Distractors, Bernoulli GLM, Bernoulli GLM Raw, Gaussian Mixture, Two Moons, SIR, Lotka-Volterra. Common subset: SLCP, Two Moons, Bernoulli GLM, Gaussian Mixture, SIR. |
| Contrastive normalizing flows (CNF) | Elsharkawy & Kahn 2025, [arXiv:2505.08709](https://arxiv.org/abs/2505.08709) | High-energy physics: two-Gaussian toy, **HiggsML / FAIR Universe Uncertainty Challenge** ([Zenodo 15131565](https://zenodo.org/records/15131565), 15.1 GB, CC-BY-4.0), MNIST appendix. The paper is HEP, but torchregress is domain-agnostic: the CNF method is reproduced below on the public FAIR-Universe sample. |
| PRIME | Lim et al., ICML 2025, [poster](https://icml.cc/virtual/2025/poster/44388) | Real DIR re-splits: AgeDB-DIR, IMDB-WIKI-DIR, NYUD2-DIR, STS-B-DIR. Metrics MAE/GM/RMSE/δ1/MSE/Pearson on All/Many/Median/Few subsets. |
| ConR | Keramati, Meng & Evans, ICLR 2024, [arXiv:2309.06651](https://arxiv.org/abs/2309.06651), [code](https://github.com/BorealisAI/ConR) | AgeDB-DIR, IMDB-WIKI-DIR, NYUD2-DIR, MPIIGaze-DIR (code licence CC BY-NC-SA 4.0). |
| Continuous group DRO | Inspired by Sagawa et al. 2020, [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) | Original Group DRO is classification over discrete groups: Waterbirds, CelebA, MultiNLI. The torchregress continuous variant is a new regression formulation. |
| Split-measurement / corrected-score EIV | Hausman et al. 1991; Stefanski & Nakamura 1994 | Theory/simulation papers with no standard public benchmark; typical setup is synthetic linear/polynomial data with known input-noise variance. |
| cINN / FrEIA | Ardizzone et al. 2019, [arXiv:1907.02392](https://arxiv.org/abs/1907.02392) | cINN papers use MNIST and ImageNet colourisation; FrEIA is a coupling-layer framework, not a dataset/method benchmark. |

## 2. Candidate comparator packages

| Package / code | What it gives us | Install | Licence | GPU / headless | Decision |
|---|---|---|---|---|---|
| **sbi** | NPE, NLE, NRE, SBC/TARP diagnostics | `pip install sbi` (CANFAR pins a git SHA) | Apache-2.0 | PyTorch, GPU yes | **Run** (`sbi_parity`) |
| **sbibm** | 10 benchmark tasks + reference posteriors | `pip install --no-deps sbibm==1.1.0` | MIT | CPU-heavy; works headless | **Run** (freeze in CANFAR session) |
| **pzflow** | tabular normalizing flows | `pip install pzflow` | MIT | JAX; CPU/GPU | **Run** (`sbi_parity`) |
| **FrEIA** | invertible coupling layers for a cINN | `pip install FrEIA` | MIT | PyTorch, GPU yes | **Run** (`freia_parity`) |
| vislearn/conditional_INNs | reference cINN examples | git clone | **no licence** | — | Do not vendor; use FrEIA directly |
| ~~imbalanced-learn~~ | classification samplers (SMOTE, ROS, …) | — | MIT | CPU | **Deliberately not used**: classification-only, and resampling/SMOTE is excluded from the torchregress roadmap (see imbalance decisions below). |
| ~~smogn~~ | SMOGN regression resampling | — | GPL-3.0 | CPU | **Deliberately not used**: resampling/SMOTE family excluded; also incompatible with the current numpy 2/pandas 3 stack. |
| ~~ImbalancedLearningRegression~~ | regression resampling (RO/SMOTE/ADASYN) | — | GPL-3.0 | CPU | **Deliberately not used**: same decision; package also incompatible with the current stack. |
| YyzHarry/imbalanced-regression | LDS/FDS/DBS research code | git clone | MIT | PyTorch GPU | Possible follow-up; not dependency-installable |
| swyft / lampe / bayesflow / pyabc | other SBI frameworks | pip | MIT/BSD | varies | Not in this campaign (sbibm/sbi covers the standard benchmark) |
| scipy.odr | classical orthogonal distance regression | SciPy | BSD-3 | CPU | Already compared in the existing `eiv` suite |
| linearmodels / semopy | IV/GMM, SEM | pip | NCSA/MIT | CPU | Not corrected-score equivalents; no added value |

Resampling (SMOTE/SMOGN/SMOTER and variants) is **intentionally excluded** from
the torchregress roadmap: synthetic-neighbour interpolation is hard to defend
for continuous targets and the literature documents leakage/optimism traps
(arXiv:2001.06296). The `imbalanced_representation` suite therefore stays
loss/representation based (PRIME/ConR/continuous GroupDRO vs MSE) on synthetic
and real tabular long-tailed data.

## 3. Real-data options

- **SBI papers are synthetic-only.** The only real-data benchmark among them
  is the HiggsML/FAIR Universe challenge (15.1 GB); it targets the CNF
  method, not NRE/NPE parity. For NRE/NPE the faithful benchmark is sbibm.
  torchregress is not astronomy-only: the HEP CNF reproduction is in scope
  (`cnf_parity`), and the public sample is enough to exercise the method
  end-to-end.
- **DIR papers use real data**, but the full DIR setup needs AgeDB/IMDB-WIKI/
  NYUD2/STS-B downloads and research-use licences (NYUD2 raw ~428 GB,
  IMDB-WIKI ~270 GB raw). They are not staged in this campaign; real tabular
  coverage instead uses the harness datasets `california_housing` (sklearn),
  `diabetes` (sklearn), `photoz_eiv`/`galaxiesml_eiv` (HSC photo-z), and real
  image suites (`bbbc_cellcount`, etc.).
- **cINN/FrEIA** has no canonical tabular benchmark; `freia_parity` therefore
  compares cINN, torchregress NSF and Gaussian NLL on the same real tabular
  regression splits.

## 4. Imbalance follow-up decisions (2026-10-18)

- **Adopted: VIR** (Variational Imbalanced Regression, arXiv:2306.06599,
  NeurIPS 2023). Regression-specific, UQ-aware, not superseded by PRIME/ConR;
  reference code is preview/no-licence, so implement from the paper.
- **Rejected: AutoBalance** (arXiv:2201.01212). Bi-level classification/fairness
  loss search; no regression path worth adding.
- **Existing coverage:** class-balanced effective-number weighting already sits
  inside PRIME; LDS/FDS, Rank-N-Contrast, PRIME/ConR, Density/Propensity
  weighting and continuous GroupDRO are already implemented.
- **Corrected:** the historical `BalancedMSELoss` is inverse bin-frequency
  weighting, not Ren et al.'s noise-adaptive Balanced MSE (BMC); docs and
  docstrings no longer misattribute it. True BMC and RankSim are the remaining
  regression-specific candidates.
- **Excluded:** SMOTE/SMOGN/SMOTER/resampling, imbalanced SSL (DASO/RDA/CoSSL/
  debiased pseudo-labels), LDAM/logit-adjustment/BBN/decoupling classification
  methods, aLRP, CASED, class-rectification mining, flooding, loss-calibrated
  BNNs, learned data-manipulation meta-learning, and tutorials.
- **Follow-ups that matter:** Balanced MSE (Ren et al., CVPR 2022,
  arXiv:2203.16427), RankSim (Gong et al., ICML 2022), and the existing
  PRIME/ConR/FDS line; for classification history, LDAM+DRW/decoupling/logit
  adjustment are the better references but out of scope.

## 5. Conformal-prediction ecosystem (awesome-conformal-prediction)

Audit of [valeman/awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction)
against the harness. The harness already compares **MAPIE**, **crepes**, and
**torchcp** (split, CQR, CV+, jackknife-style and the torchregress predictor
zoo) in the conformal suite, plus MAPIE/crepes in image conformal and
multivariate-interval suites. Candidates not yet covered:

| Candidate | Kind | Why it matters for the harness |
|---|---|---|
| [puncc](https://github.com/deel-ai/puncc) | PyTorch conformal library | Independent PyTorch implementation of split/CQR/risk-control regression wrappers; strongest missing software comparator. |
| [conformal-tights](https://github.com/radix-ai/conformal-tights) | sklearn meta-estimator | Coherent quantiles/intervals across several targets; useful for multi-output interval parity. |
| [cqr](https://github.com/yromano/cqr) | reference code | Original conformalized quantile regression implementation (Romano et al. 2019); good CQR parity reference. |
| [TorchUQ](https://github.com/TorchUQ/torchuq) | PyTorch UQ/conformal toolkit | Benchmark-style UQ + conformal baselines; overlaps with torch-uncertainty but adds an independent API. |
| [crepes-weighted](https://github.com/predict-idlab/crepes-weighted) | weighted conformal | Covariate-shift weighted CP; the harness has shift suites but no external weighted-CP comparator yet. |
| [unquad](https://github.com/OliverHennhoefer/unquad) | conformal anomaly detection | Conformal OOD/anomaly detection; the harness has OOD metrics but no conformal-anomaly suite. |
| [copula-conformal MTR](https://github.com/M-Soundouss/CopulaConformalMTR) | multi-target conformal | Multivariate interval alternative to the harness's own multivariate-intersection rows. |
| [nonconformist](https://github.com/donlnz/nonconformist) | classic library | Widely cited but unmaintained; not recommended as a new dependency. |
| Venn-ABERS (`venn-abers`, `VennABERS`) | probabilistic calibration | Relevant to calibration/risk but not to regression intervals directly. |

Method-level gaps worth adding to the conformal suite: Mondrian/group-conditional
coverage, weighted conformal under covariate shift, conformal risk control, and
a proper jackknife+/CV+ external parity row. Priority for the next harness
iteration: **puncc** (software parity), then **conformal-tights** and the
official **cqr** reference; `nonconformist` should stay out because of
maintenance status.

## 6. Contrastive normalizing flow reproduction

The preceding CNF paper is HEP but the method is domain-agnostic. The new
`cnf_parity` suite reproduces the core recipe on:

* the paper's two-Gaussian toy (2-D, rotation nuisance tests), and
* the public FAIR-Universe HiggsML starting-kit sample (1000 simulated events,
  28 features; the full challenge set is 15.1 GB on Zenodo 15131565).

Implemented: contrastive loss with loss clamping at `k=-50`, `c=2` for the
signal flow and `c=0.5` for the background flow, discrimination functions
`Phi = p/(1+p)`, and a DNN classifier on raw features + `Phi` values. The
paper's 30-layer gated-affine/quadratic coupling architecture is approximated
with zuko NSF couplings (documented deviation). On the CANFAR GPU run the toy
reproduction did **not** recover the paper's rotation-robustness advantage
over a plain DNN, while on the real sample it matched the DNN's AUC and
slightly improved accuracy/log-loss. Those are reported as measured negative/
neutral findings; the full 15.1 GB challenge data and Neyman confidence-belt
pipeline remain a follow-up.

## 7. CANFAR GPU campaign

A single multi-group headless run (`scripts/canfar_launch_release.sh`,
`ALL_GPU=1`, one replica per group, `SEEDS=0`) with a job filter:

- `sbi`: all `sbi_parity` sbibm tasks — torchregress NSF/MAF/MDN vs **sbi**
  NPE/MAF/MDN/FMPE/NPSE and **pzflow** NSF, plus sbi's own SBC/TARP.
- `tabular`: `likelihood_ratio` (NRE/contrastive/NPE vs sklearn NRE),
  `eiv_estimators` (forward inversion/split/corrected score vs sklearn OLS),
  `imbalanced_representation` (PRIME/ConR/GroupDRO vs MSE; synthetic + real
  `california_housing`), `freia_parity` (cINN vs NSF/GaussianNLL; real tabular),
  `calibration` (VarianceRestoring vs sklearn isotonic), `cnf_parity` (CNF
  reproduction on the FAIR-Universe sample + toy), and `eiv photoz_eiv`
  (real HSC photo-z EIV).
- `image`: `cell_counting bbbc_cellcount[-tiny]` — real fluorescence images on
  GPU, exercising the image wrappers/backbones.
