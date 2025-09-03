SCORF: Selective Cost-Aware Oblique Random Forests

A compact, reproducible reference implementation of SCORF—a decision-focused oblique tree ensemble with abstention and cost-aware predictions under unreliable data (covariate shift, label noise, and MCAR missingness). Includes baselines, ablations, and paper-style tables printed to stdout.

⸻

Highlights
	•	Oblique geometry from gradients. Learns a low-rank spectral sensitivity transform H from finite-difference class-probability gradients.
	•	Targeted robustness. Trains on sensitivity-direction augmentations (x \pm \rho\,\hat{u}(x)).
	•	Selective, cost-aware decisions. Isotonic calibration + Wilson bound to enforce kept-error ≤ α with an abstain option; minimizes expected policy cost.
	•	Stress testing. Turn-key experiments for Clean, Covariate Shift, Asymmetric Label Noise, and MCAR Missingness.
	•	Baselines & ablations. CRC-style confidence thresholding, HGBT/XGBoost, RotationForest, RerF; hyper-parameter sensitivity, surrogate-for-gradient study, runtime scaling.

⸻

Repository structure (suggested)

.
├── README.md
├── scorf.py                 # this file (all code shown below)
└── data/
    ├── UCI_Credit_Card.csv  # UCI Credit Card Default
    ├── heloc_dataset_v1.csv # HELOC
    └── cs-training.csv      # Give Me Some Credit (Kaggle)

If your entrypoint file is named differently, replace scorf.py in the commands below with your filename.

⸻

Installation

Tested with Python ≥ 3.9.

# 1) Create and activate a fresh environment (recommended)
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 2) Core dependencies
pip install -U numpy scipy scikit-learn pandas

# 3) Optional baselines (install what you need)
pip install xgboost lightgbm rerf rotation-forest

The code try/except-imports optional packages; if a package isn’t installed, the corresponding baseline is skipped gracefully.

⸻

Data

Place the following files into data/ with exact filenames expected by the loaders:
	•	UCI_Credit_Card.csv (expects column default.payment.next.month)
	•	heloc_dataset_v1.csv   (expects column RiskPerformance with values like Bad/Good)
	•	cs-training.csv        (expects column SeriousDlqin2yrs)

Tip: The script also uses sklearn.datasets.fetch_covtype() for a quick sanity check on the Covertype dataset.

⸻

Quickstart

Run everything (core experiments + baselines + ablations):

python scorf.py

You’ll see paper-style summaries printed to stdout, e.g.:

[Clean]          cost=...  abstain=...  kept-error=...
[CovariateShift] cost=...  abstain=...  kept-error=...
[LabelNoise]     cost=...  abstain=...  kept-error=...
[MCAR Missing]   cost=...  abstain=...  kept-error=...

[Table 3] CRC vs SCORF on UCI Credit (α=0.10)
...

[Table 4] UCI Credit: Rotation/RerF vs SCORF (α=0.10)
...

[Table 5] Per-dataset policy cost (Clean / Shift / Noise / Missing)
...

[Table 6] GMSC temporal split (natural shift):
...

[Table 7] Runtime (one run)
...

[Table 8] Sensitivity on UCI Credit (Shift) – report cost
...

[Table 9] Surrogate for G on UCI Credit (Shift)
...

Run just the “paper-like” credit-risk summary:

# from a Python shell
from scorf import run_paper_like
run_paper_like(dataset_key="uci", seeds=10)   # keys: "uci", "heloc", "gmsc"


⸻

What the code does (at a glance)
	1.	Imputation. Median imputer for numerical features.
	2.	Gradient surrogate. Fit a small RF on a subsample then compute finite-difference gradients of P(y{=}1\mid x).
	3.	Spectral sensitivity. Form S=\frac{1}{m}G^\top G, shrink eigenvalues via \lambda/(\lambda+\gamma) to get transform H.
	4.	Targeted augmentation. For a random subset, move along normalized directions \hat{u}(x)=\frac{H\,g(x)}{\|H\,g(x)\|} by \pm \rho.
	5.	Final model. Train a standard RandomForest on transformed features XH plus the augmented samples.
	6.	Calibration & selection.
	•	Fit IsotonicRegression on calibration data.
	•	Build a score combining decision margin, abstention cost, optional confidence/severity z-scores.
	•	Use Wilson’s upper bound to pick a threshold \tau so that kept-error ≤ α with high probability.
	•	At test time, abstain if expected error cost exceeds abstention cost or score < \tau.

Stressors implemented
	•	Covariate shift: add fixed offsets on top-variance features.
	•	Label noise: asymmetric flips (1→0 with prob 0.15, 0→1 with prob 0.05).
	•	MCAR missingness: random missing entries at chosen rates.

Metrics printed
	•	policy cost (mean cost using cost_fp, cost_fn, c_abs)
	•	abstain rate (fraction abstained)
	•	kept-error (error rate among non-abstained)

⸻

Experiments included
	•	Core sanity check: run_clean, run_shift, run_label_noise, run_missing on Covertype.
	•	Credit-risk suite: run_paper_like("uci" | "heloc" | "gmsc").
	•	CRC vs SCORF: run_crc_vs_scorf_on_uci_shift_and_missing.
	•	Oblique baselines: RotationForest & RerF vs SCORF (run_oblique_baselines_uci).
	•	Per-dataset comparison: HGBT vs SCORF (run_per_dataset_results).
	•	Temporal split: GMSC natural drift (run_temporal_gmsc).
	•	Runtime & scaling: run_runtime_scalability.
	•	Hyper-parameter sensitivity: run_hyperparam_sensitivity_uci_shift.
	•	Surrogate for gradients: run_surrogate_ablation_uci_shift.

Each function prints a reproducible table; seeds are controllable via arguments and SCORFParams.random_state.

⸻

Configuration

Tune behavior via SCORFParams:

Param	Default	Meaning
n_trees_grad	50	Trees in the small RF used only to compute probability gradients.
grad_sample	2000	Subsample size for gradient estimation.
step	1e-2	Finite-difference step size.
gamma	1e-3	Eigenvalue shrinkage in spectral transform H.
rho	0.10	Step size for sensitivity-direction augmentation.
aug_frac	0.5	Fraction of training set to augment.
n_trees_final	200	Trees in the final RF trained on XH (+ augmentation).
alpha	0.10	Target kept-error upper bound on calibration.
z	1.96	z-value for Wilson bound (~95%).
beta	1.0	Weight for abstention-vs-error tradeoff in the score.
w_conf / w_sev	0.5 / 0.5	Optional confidence/severity weights (z-scored).
cost_fp / cost_fn / c_abs	1 / 25 / 2	Policy costs for FP, FN, and abstention.
random_state	42	RNG seed.

Edit the defaults in the file or construct SCORF(SCORFParams(...)) manually.

⸻

Reproducibility notes
	•	Fixed random_state seeds are used where applicable; repeated runs with the same seeds produce identical summaries (allowing for ML library nondeterminism).
	•	Optional baselines are skipped if the corresponding packages aren’t installed.
	•	The shift/missingness/noise generators are intentionally simple to keep experiments portable.

⸻

Limitations (practical)
	•	The covariate shift simulator uses fixed offsets on high-variance features (not dataset-specific drift modeling).
	•	Missingness is MCAR; other mechanisms (MAR/MNAR) are not modeled here.
	•	The CRC-style baseline uses a confidence-margin heuristic for thresholding; it serves as a lightweight reference, not a full CRC implementation.

⸻

How to cite

If this code helps your work, please cite this repository:

@software{scorf_repo,
  title  = {SCORF: Selective Cost-Aware Oblique Random Forests},
  author = {Rauniyar, Sarwesh and contributors},
  year   = {2025},
  url    = {https://github.com/<your-org>/<your-repo>}
}

(Replace the URL once your repository is public.)

⸻

License

Choose a license that matches your goals (e.g., MIT, Apache-2.0, BSD-3-Clause) and add it as LICENSE. Until then, this code is “all rights reserved” by default.

⸻

Contributing

Issues and PRs are welcome—especially for:
	•	new datasets/loaders
	•	additional shift/missingness/noise models
	•	faster/cleaner gradient surrogates and oblique baselines

⸻

Acknowledgements

This repo uses open-source libraries including NumPy, SciPy, scikit-learn, pandas, and (optionally) XGBoost, LightGBM, RerF, and RotationForest.
