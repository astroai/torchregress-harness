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
| Contrastive normalizing flows (CNF) | Elsharkawy & Kahn 2025, [arXiv:2505.08709](https://arxiv.org/abs/2505.08709) | High-energy physics: two-Gaussian toy, **HiggsML / FAIR Universe Uncertainty Challenge** ([Zenodo 15131565](https://zenodo.org/records/15131565), 15.1 GB, CC-BY-4.0), MNIST appendix. **Not astronomy/strong lensing.** |
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
| imbalanced-learn | classification samplers (SMOTE, ROS, …) | `pip install imbalanced-learn` | MIT | CPU | **No regression API** — not used as a regression comparator |
| smogn | SMOGN regression resampling | `pip install smogn` | GPL-3.0 | CPU | Installed and tested; **incompatible with numpy 2 / pandas 3** (`np.quantile(interpolation=)`, scalar `choice`); not added |
| ImbalancedLearningRegression | regression resampling (RO/SMOTE/ADASYN) | `pip install ImbalancedLearningRegression` | GPL-3.0 | CPU | Same numpy-2 incompatibility; random over-sampling works only after a shim; not added |
| YyzHarry/imbalanced-regression | LDS/FDS/DBS research code | git clone | MIT | PyTorch GPU | Possible follow-up; not dependency-installable |
| swyft / lampe / bayesflow / pyabc | other SBI frameworks | pip | MIT/BSD | varies | Not in this campaign (sbibm/sbi covers the standard benchmark) |
| scipy.odr | classical orthogonal distance regression | SciPy | BSD-3 | CPU | Already compared in the existing `eiv` suite |
| linearmodels / semopy | IV/GMM, SEM | pip | NCSA/MIT | CPU | Not corrected-score equivalents; no added value |

Notes from testing in the harness env: `smogn` and
`ImbalancedLearningRegression` both call APIs removed in numpy 2/pandas 3
(`np.quantile(..., interpolation=...)`; `int(np.random.choice(...))`), so they
cannot be used as clean comparators without pinning an old scientific stack.
The `imbalanced_representation` suite therefore keeps torchregress PRIME/ConR/
continuous-GroupDRO vs MSE on synthetic and real tabular long-tailed data, and
documents the absence of a usable regression-resampling package as a research
finding rather than shipping a broken baseline.

## 3. Real-data options

- **SBI papers are synthetic-only.** The only real-data SBI benchmark among them
  is the HiggsML/FAIR Universe challenge (15.1 GB); it targets HEP CNF, not
  NRE/NPE parity. For NRE/NPE the faithful benchmark is sbibm.
- **DIR papers use real data**, but the full DIR setup needs AgeDB/IMDB-WIKI/
  NYUD2/STS-B downloads and research-use licences (NYUD2 raw ~428 GB,
  IMDB-WIKI ~270 GB raw). They are not staged in this campaign; real tabular
  coverage instead uses the harness datasets `california_housing` (sklearn),
  `diabetes` (sklearn), `photoz_eiv`/`galaxiesml_eiv` (HSC photo-z), and real
  image suites (`bbbc_cellcount`, etc.).
- **cINN/FrEIA** has no canonical tabular benchmark; `freia_parity` therefore
  compares cINN, torchregress NSF and Gaussian NLL on the same real tabular
  regression splits.

## 4. CANFAR GPU campaign

A single multi-group headless run (`scripts/canfar_launch_release.sh`,
`ALL_GPU=1`, one replica per group, `SEEDS=0`) with a job filter:

- `sbi`: all `sbi_parity` sbibm tasks — torchregress NSF/MAF/MDN vs **sbi**
  NPE/MAF/MDN/FMPE/NPSE and **pzflow** NSF, plus sbi's own SBC/TARP.
- `tabular`: `likelihood_ratio` (NRE/contrastive/NPE vs sklearn NRE),
  `eiv_estimators` (forward inversion/split/corrected score vs sklearn OLS),
  `imbalanced_representation` (PRIME/ConR/GroupDRO vs MSE; synthetic + real
  `california_housing`), `freia_parity` (cINN vs NSF/GaussianNLL; real tabular),
  `calibration` (VarianceRestoring vs sklearn isotonic), and `eiv photoz_eiv`
  (real HSC photo-z EIV).
- `image`: `cell_counting bbbc_cellcount[-tiny]` — real fluorescence images on
  GPU, exercising the image wrappers/backbones.
