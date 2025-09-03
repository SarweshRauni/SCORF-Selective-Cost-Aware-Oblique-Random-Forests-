import numpy as np
from numpy.linalg import eigh
from scipy.spatial.distance import mahalanobis
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_random_state
from dataclasses import dataclass


def mcar(X, p, seed=None):
    rng = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)
    Xc  = X.copy()
    mask = rng.rand(*Xc.shape) < p
    Xc[mask] = np.nan
    return Xc, mask.astype(float)  # return mask for severity

def flip_labels_asymmetric(y, p10=0.15, p01=0.05, seed=0):
    rng = np.random.RandomState(seed)
    y = y.copy()
    flip_10 = (y==1) & (rng.rand(len(y)) < p10)  # 1->0
    flip_01 = (y==0) & (rng.rand(len(y)) < p01)  # 0->1
    y[flip_10] = 0
    y[flip_01] = 1
    return y

def apply_covariate_shift(X, feat_idx, deltas):
    Xs = X.copy()
    Xs[:, feat_idx] = Xs[:, feat_idx] + deltas
    return Xs

def wilson_upper_bound(k, n, z=1.96):
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1.0 + z**2 / n
    center = phat + z**2 / (2*n)
    rad = z * np.sqrt((phat*(1-phat))/n + z**2/(4*n**2))
    return (center + rad) / denom

def policy_cost(y_true, y_pred, abstain_mask, cost_fp=1.0, cost_fn=25.0, c_abs=2.0):
    # y_pred in {0,1}; abstain_mask True => abstained
    cost = np.zeros_like(y_true, dtype=float)
    # abstentions
    cost[abstain_mask] = c_abs
    # mistakes among non-abstained
    idx = ~abstain_mask
    wrong = (y_pred[idx] != y_true[idx])
    fp = (y_pred[idx]==1) & (y_true[idx]==0)
    fn = (y_pred[idx]==0) & (y_true[idx]==1)
    cost[idx & fp] = cost_fp
    cost[idx & fn] = cost_fn
    return cost.mean(), wrong.mean() if idx.sum()>0 else 0.0, idx.mean()


@dataclass
class SCORFParams:
    # spectral/grad
    n_trees_grad:int = 50
    grad_sample:int = 2000
    step:float = 1e-2
    gamma:float = 1e-3
    # augmentation
    rho:float = 0.10
    aug_frac:float = 0.5
    # final forest
    n_trees_final:int = 200
    # selective calibration
    alpha:float = 0.10
    z:float = 1.96
    beta:float = 1.0
    w_conf:float = 0.5
    w_sev:float = 0.5
    # costs
    cost_fp:float = 1.0
    cost_fn:float = 25.0
    c_abs:float = 2.0
    random_state:int = 42

class SCORF:
    def __init__(self, params: SCORFParams):
        self.p = params
        self.rng = check_random_state(self.p.random_state)
        # learned pieces
        self.imp_ = SimpleImputer(strategy="median")
        self.H_ = None
        self.rf_ = None
        # calibration (isotonic) + per-condition stats
        self.iso_ = None
        self.tau_ = None
        self._sev_mu_ = 0.0
        self._sev_sd_ = 1.0
        self._conf_mu_ = 0.0
        self._conf_sd_ = 1.0
        self._mah_mu_ = None
        self._mah_cov_inv_ = None  # for shift severity

    def _rf(self, n):
        return RandomForestClassifier(
            n_estimators=n, max_depth=None, n_jobs=-1, random_state=self.p.random_state
        )

    def _finite_diff_grad(self, clf, X, cls=1):
        """Gradient of P(y=cls|x) via central finite differences."""
        n, d = X.shape
        h = self.p.step
        # build batches for central differences
        Xp = np.repeat(X[:, None, :], d, axis=1)
        Xm = Xp.copy()
        rows = np.arange(n)[:, None]
        cols = np.arange(d)[None, :]
        Xp[rows, cols, cols] += h/2
        Xm[rows, cols, cols] -= h/2
        batch = np.vstack([Xp.reshape(-1, d), Xm.reshape(-1, d)])
        P = clf.predict_proba(batch)[:, 1]  # positive class prob
        Pp = P[:n*d].reshape(n, d)
        Pm = P[n*d:].reshape(n, d)
        G = (Pp - Pm) / h   # n × d
        return G  # rows=samples, cols=features

    def fit(self, X, y):
        X_imp = self.imp_.fit_transform(X)
        n, d = X_imp.shape
        # 1)  model on subsample for gradients
        idx = self.rng.choice(n, size=min(self.p.grad_sample, n), replace=False)
        Xs, ys = X_imp[idx], y[idx]
        rf_g = self._rf(self.p.n_trees_grad).fit(Xs, ys)

        # 2) spectral sensitivity from gradients 
        G = self._finite_diff_grad(rf_g, Xs)          # n_sub × d
        S = (G.T @ G) / G.shape[0]                    # d × d
        vals, vecs = eigh(S)
        self.H_ = vecs @ np.diag(vals / (vals + self.p.gamma)) @ vecs.T  # d×d

        # 3) sensitivity-direction augmentation
        m_aug = int(self.p.aug_frac * n)
        id_aug = self.rng.choice(n, size=m_aug, replace=False)
        Xa = X_imp[id_aug]
        Ga = self._finite_diff_grad(rf_g, Xa)         # m_aug × d
        # direction u(x) = H g(x) / ||H g(x)||
        Hg = Ga @ self.H_.T
        norms = np.linalg.norm(Hg, axis=1, keepdims=True) + 1e-12
        U = Hg / norms
        X_plus  = Xa + self.p.rho * U
        X_minus = Xa - self.p.rho * U
        y_aug = y[id_aug]
        # 4) train final RF on transformed features XH (original + augmented)
        XH_base = X_imp @ self.H_
        XH_plus = X_plus @ self.H_
        XH_minus = X_minus @ self.H_
        XH = np.vstack([XH_base, XH_plus, XH_minus])
        y_all = np.concatenate([y, y_aug, y_aug])
        self.rf_ = self._rf(self.p.n_trees_final).fit(XH, y_all)
        return self

    def _raw_proba(self, X):
        X_imp = self.imp_.transform(X)
        XH = X_imp @ self.H_
        P = self.rf_.predict_proba(XH)[:, 1]
        return P, X_imp, XH

    def predict_proba(self, X):
        P, _, _ = self._raw_proba(X)
        if self.iso_ is not None:
            return self.iso_.transform(P)
        return P

    def _conf_score(self, p):
        eps = 1e-12
        H = -(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))
        Hmax = np.log(2.0)
        return 1.0 - (H / Hmax)

    def _sev_missing(self, miss_frac):
        return miss_frac

    def _sev_noise(self, XH):
        ests = self.rf_.estimators_
        if len(ests) == 0:
            return np.zeros(XH.shape[0])
        probs = np.stack([e.predict_proba(XH)[:,1] for e in ests], axis=1)  # n × T
        return probs.var(axis=1)

    def _sev_shift(self, XH):
        if self._mah_mu_ is None:
            return np.zeros(XH.shape[0])
        VI = self._mah_cov_inv_
        mu = self._mah_mu_
        diffs = XH - mu
        # fast diagonal or full inverse
        if VI.ndim == 1:
            return np.sqrt(((diffs**2) * VI).sum(axis=1))
        return np.sqrt(np.einsum("ni,ij,nj->n", diffs, VI, diffs))

    def calibrate(self, X_cal, y_cal, condition="clean", miss_mask=None):
        """
        Fits isotonic on cal, computes severity z-scores & confidence z-scores,
        and finds tau via Wilson to enforce kept-error <= alpha.
        condition in {"clean","shift","missing","noise"}.
        miss_mask: fraction-missing per sample for 'missing'.
        """
        P_raw, X_imp, XH = self._raw_proba(X_cal)
        self.iso_ = IsotonicRegression(out_of_bounds="clip").fit(P_raw, y_cal)
        P = self.iso_.transform(P_raw)

        # 2) compute per-sample costs & auxiliaries on cal
        e0 = P * self.p.cost_fn                  # E[cost|predict 0] = p*FN
        e1 = (1-P) * self.p.cost_fp              # E[cost|predict 1] = (1-p)*FP
        emin = np.minimum(e0, e1)
        esec = np.maximum(e0, e1)
        margin = esec - emin                     # m(x)

        conf = self._conf_score(P)

        if condition == "missing":
            sev = miss_mask.mean(axis=1) if miss_mask is not None else np.zeros_like(P)
        elif condition == "shift":
            mu = XH.mean(axis=0)
            diffs = XH - mu
            cov = np.var(XH, axis=0) + 1e-6
            self._mah_mu_ = mu
            self._mah_cov_inv_ = 1.0 / cov
            sev = self._sev_shift(XH)
        elif condition == "noise":
            sev = self._sev_noise(XH)
        else:
            sev = np.zeros_like(P)

        # z-score conf & sev on cal
        self._conf_mu_, self._conf_sd_ = conf.mean(), conf.std() + 1e-12
        self._sev_mu_,  self._sev_sd_  = sev.mean(),  sev.std()  + 1e-12
        conf_z = (conf - self._conf_mu_) / self._conf_sd_
        sev_z  = (sev  - self._sev_mu_)  / self._sev_sd_

        s = margin + self.p.beta * (self.p.c_abs - emin) + self.p.w_conf * conf_z - self.p.w_sev * sev_z

        order = np.argsort(-s)
        errs = []
        kept_err_ub = []
        k_best = 0
        for k in range(1, len(order)+1):
            idx = order[:k]
            yhat_kept = (e1[idx] < e0[idx]).astype(int)
            err_k = (yhat_kept != y_cal[idx]).sum()
            ub = wilson_upper_bound(err_k, k, z=self.p.z)
            errs.append(err_k)
            kept_err_ub.append(ub)
            if ub <= self.p.alpha:
                k_best = k

        if k_best == 0:
            self.tau_ = np.inf
        else:
            tau_idx = order[k_best-1]
            self.tau_ = s[tau_idx]
        return self


    def predict_with_abstain(self, X, condition="clean", miss_mask=None):
        P, X_imp, XH = self._raw_proba(X)
        if self.iso_ is not None:
            P = self.iso_.transform(P)

        e0 = P * self.p.cost_fn
        e1 = (1-P) * self.p.cost_fp
        emin = np.minimum(e0, e1)

        if condition == "missing":
            sev = miss_mask.mean(axis=1) if miss_mask is not None else np.zeros_like(P)
        elif condition == "shift":
            sev = self._sev_shift(XH)
        elif condition == "noise":
            sev = self._sev_noise(XH)
        else:
            sev = np.zeros_like(P)

        conf = self._conf_score(P)
        conf_z = (conf - self._conf_mu_) / (self._conf_sd_ + 1e-12)
        sev_z  = (sev  - self._sev_mu_)  / (self._sev_sd_  + 1e-12)
        esec = np.maximum(e0, e1)
        margin = esec - emin
        s = margin + self.p.beta * (self.p.c_abs - emin) + self.p.w_conf * conf_z - self.p.w_sev * sev_z

        # abstain rule: emin > c_abs OR s < tau
        abstain = (emin > self.p.c_abs) | (s < self.tau_)
        yhat = (e1 < e0).astype(int)
        return yhat, abstain

X, y_mult = fetch_covtype(return_X_y=True)
y = (y_mult == 1).astype(int)

# split train/cal/test
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

params = SCORFParams()
scorf = SCORF(params)

def run_clean():
    scorf.fit(X_tr, y_tr)
    scorf.calibrate(X_cal, y_cal, condition="clean")
    yhat, abst = scorf.predict_with_abstain(X_te, condition="clean")
    cost, kept_err, coverage = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
    print(f"[Clean]         cost={cost:.3f}  abstain={abst.mean():.3f}  kept-error={kept_err:.3f}")

def run_shift():
    scorf.fit(X_tr, y_tr)
    var = X_tr.var(axis=0)
    feat_idx = np.argsort(-var)[:8]
    deltas = 0.5 * np.sqrt(var[feat_idx])  # fixed offsets
    X_cal_s = apply_covariate_shift(X_cal, feat_idx, deltas)
    X_te_s  = apply_covariate_shift(X_te,  feat_idx, deltas)
    scorf.calibrate(X_cal_s, y_cal, condition="shift")
    yhat, abst = scorf.predict_with_abstain(X_te_s, condition="shift")
    cost, kept_err, coverage = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
    print(f"[CovariateShift] cost={cost:.3f}  abstain={abst.mean():.3f}  kept-error={kept_err:.3f}")

def run_label_noise():
    y_tr_noisy = flip_labels_asymmetric(y_tr, p10=0.15, p01=0.05, seed=0)
    scorf.fit(X_tr, y_tr_noisy)
    scorf.calibrate(X_cal, y_cal, condition="noise")
    yhat, abst = scorf.predict_with_abstain(X_te, condition="noise")
    cost, kept_err, coverage = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
    print(f"[LabelNoise]    cost={cost:.3f}  abstain={abst.mean():.3f}  kept-error={kept_err:.3f}")

def run_missing():
    X_tr_m, _ = mcar(X_tr, 0.10, seed=1)
    scorf.fit(X_tr_m, y_tr)
    X_cal_m, M_cal = mcar(X_cal, 0.20, seed=2)
    X_te_m,  M_te  = mcar(X_te,  0.20, seed=3)
    scorf.calibrate(X_cal_m, y_cal, condition="missing", miss_mask=M_cal)
    yhat, abst = scorf.predict_with_abstain(X_te_m, condition="missing", miss_mask=M_te)
    cost, kept_err, coverage = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
    print(f"[MCAR Missing]  cost={cost:.3f}  abstain={abst.mean():.3f}  kept-error={kept_err:.3f}")

if __name__ == "__main__":
    run_clean()
    run_shift()
    run_label_noise()
    run_missing()









#NEW BLOCK: 



import os, pandas as pd

DATA_DIR = "data"  

def load_uci_credit(path=os.path.join(DATA_DIR,"UCI_Credit_Card.csv")):
    df = pd.read_csv(path); df = df.drop(columns=[c for c in ["ID"] if c in df.columns])
    y = df["default.payment.next.month"].astype(int).values
    X = df.drop(columns=["default.payment.next.month"]).values
    return X, y

def load_heloc(path=os.path.join(DATA_DIR,"heloc_dataset_v1.csv")):
    df = pd.read_csv(path)
    y = (df["RiskPerformance"].astype(str).str.upper().str.strip()=="BAD").astype(int).values
    X = df.drop(columns=["RiskPerformance"]).apply(pd.to_numeric, errors="coerce").replace({-9:np.nan,-8:np.nan,-7:np.nan})
    return X.values, y

def load_gmsc(path=os.path.join(DATA_DIR,"cs-training.csv")):
    df = pd.read_csv(path).drop(columns=[c for c in ["Unnamed: 0","ID","id"] if c in df.columns])
    y = df["SeriousDlqin2yrs"].astype(int).values
    X = df.drop(columns=["SeriousDlqin2yrs"]).values
    return X, y

LOADERS = {"uci": load_uci_credit, "heloc": load_heloc, "gmsc": load_gmsc}

def temporal_split_indices(n, train_frac=0.8):
    n_tr = int(np.floor(train_frac*n))
    idx = np.arange(n)
    return idx[:n_tr], idx[n_tr:]

def eval_once(X, y, seed, condition, params, temporal=False):
    rng = np.random.RandomState(seed)
    if temporal:
        idx_tr, idx_rest = temporal_split_indices(len(y), 0.8)
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_rest, y_rest = X[idx_rest], y[idx_rest]
        X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=43+seed)
    else:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42+seed)
        X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=43+seed)

    scorf = SCORF(params)
    if condition=="missing":
        X_tr_m,_ = mcar(X_tr, 0.10, seed=rng.randint(1e9)); scorf.fit(X_tr_m, y_tr)
        X_cal_m,Mcal = mcar(X_cal, 0.20, seed=rng.randint(1e9))
        X_te_m, Mte  = mcar(X_te,  0.20, seed=rng.randint(1e9))
        scorf.calibrate(X_cal_m, y_cal, condition="missing", miss_mask=Mcal)
        yhat, abst = scorf.predict_with_abstain(X_te_m, condition="missing", miss_mask=Mte)
    elif condition=="shift":
        scorf.fit(X_tr, y_tr)
        var = X_tr.var(axis=0); feat_idx = np.argsort(-var)[:8]; deltas = 0.5*np.sqrt(var[feat_idx])
        X_cal_s = apply_covariate_shift(X_cal, feat_idx, deltas); X_te_s = apply_covariate_shift(X_te, feat_idx, deltas)
        scorf.calibrate(X_cal_s, y_cal, condition="shift")
        yhat, abst = scorf.predict_with_abstain(X_te_s, condition="shift")
    elif condition=="noise":
        y_tr_noisy = flip_labels_asymmetric(y_tr, p10=0.15, p01=0.05, seed=rng.randint(1e9))
        scorf.fit(X_tr, y_tr_noisy); scorf.calibrate(X_cal, y_cal, condition="noise")
        yhat, abst = scorf.predict_with_abstain(X_te, condition="noise")
    else:
        scorf.fit(X_tr, y_tr); scorf.calibrate(X_cal, y_cal, condition="clean")
        yhat, abst = scorf.predict_with_abstain(X_te, condition="clean")

    cost, kept_err, cov = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
    return cost, abst.mean(), kept_err

def run_paper_like(dataset_key="uci", seeds=10):
    X, y = LOADERS[dataset_key]()
    temporal = (dataset_key=="gmsc")
    params = SCORFParams()
    for name in ["Clean","CovariateShift","LabelNoise","MCAR Missing"]:
        scores = [eval_once(X, y, seed=1000+s,
                            condition=("clean" if name=="Clean" else
                                       "shift" if name=="CovariateShift" else
                                       "noise" if name=="LabelNoise" else
                                       "missing"),
                            params=params, temporal=temporal)
                  for s in range(seeds)]
        arr = np.array(scores)
        m = arr.mean(axis=0); sd = arr.std(axis=0)
        print(f"{dataset_key:5s} | {name:16s} cost={m[0]:.3f}±{sd[0]:.3f}  abstain={m[1]:.3f}±{sd[1]:.3f}  kept-err={m[2]:.3f}±{sd[2]:.3f}")







#NEW BLOCK:




import os, time, numpy as np, pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from rerf.rerfClassifier import rerfClassifier as RerFClassifier
except Exception:
    RerFClassifier = None

try:
    from rotation_forest import RotationForestClassifier
except Exception:
    RotationForestClassifier = None

DATA_DIR = "data"

def load_uci_credit(path=os.path.join(DATA_DIR,"UCI_Credit_Card.csv")):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["ID"] if c in df.columns])
    y = df["default.payment.next.month"].astype(int).values
    X = df.drop(columns=["default.payment.next.month"]).values
    return X, y

def load_heloc(path=os.path.join(DATA_DIR,"heloc_dataset_v1.csv")):
    df = pd.read_csv(path)
    y = (df["RiskPerformance"].astype(str).str.upper().str.strip()=="BAD").astype(int).values
    X = df.drop(columns=["RiskPerformance"]).apply(pd.to_numeric, errors="coerce").replace({-9:np.nan,-8:np.nan,-7:np.nan})
    return X.values, y

def load_gmsc(path=os.path.join(DATA_DIR,"cs-training.csv")):
    df = pd.read_csv(path).drop(columns=[c for c in ["Unnamed: 0","ID","id"] if c in df.columns])
    y = df["SeriousDlqin2yrs"].astype(int).values
    X = df.drop(columns=["SeriousDlqin2yrs"]).values
    return X, y

LOADERS = {"uci": load_uci_credit, "heloc": load_heloc, "gmsc": load_gmsc}

def temporal_split_indices(n, train_frac=0.8):
    n_tr = int(np.floor(train_frac*n))
    idx = np.arange(n)
    return idx[:n_tr], idx[n_tr:]

class CostDeferWrapper:
    def __init__(self, base_clf, params):
        self.base_clf = base_clf
        self.params = params
        self.imp_ = SimpleImputer(strategy="median")
        self.iso_ = None
        self.tau_ = None
        # for optional sev/conf terms (by default we set weights to 0 to match a minimal fair defer)
        self._conf_mu_ = 0.0; self._conf_sd_ = 1.0
        self._sev_mu_  = 0.0; self._sev_sd_  = 1.0

    def _conf_score(self, p):
        eps = 1e-12
        H = -(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))
        return 1.0 - H/np.log(2.0)

    def fit(self, X, y):
        X_imp = self.imp_.fit_transform(X)
        self.base_clf.fit(X_imp, y)
        return self

    def _raw_proba(self, X):
        X_imp = self.imp_.transform(X)
        P = self.base_clf.predict_proba(X_imp)[:,1]
        return P, X_imp

    def calibrate(self, X_cal, y_cal, condition="clean", miss_mask=None, use_conf=False, use_sev=False, sev_vals=None):
        P_raw, Xc = self._raw_proba(X_cal)
        self.iso_ = IsotonicRegression(out_of_bounds="clip").fit(P_raw, y_cal)
        P = self.iso_.transform(P_raw)

        # costs
        e0 = P * self.params.cost_fn
        e1 = (1-P) * self.params.cost_fp
        emin = np.minimum(e0, e1)
        esec = np.maximum(e0, e1)
        margin = esec - emin

        # optional terms
        conf = self._conf_score(P) if use_conf else np.zeros_like(P)
        sev = sev_vals if (use_sev and sev_vals is not None) else np.zeros_like(P)

        # z-normalize when used
        if use_conf:
            self._conf_mu_, self._conf_sd_ = conf.mean(), conf.std()+1e-12
            conf = (conf - self._conf_mu_) / self._conf_sd_
        if use_sev:
            self._sev_mu_, self._sev_sd_ = sev.mean(), sev.std()+1e-12
            sev = (sev - self._sev_mu_) / self._sev_sd_

        s = margin + self.params.beta*(self.params.c_abs - emin) + self.params.w_conf*conf - self.params.w_sev*sev

        order = np.argsort(-s)
        k_best = 0
        for k in range(1, len(order)+1):
            idx = order[:k]
            yhat_kept = (e1[idx] < e0[idx]).astype(int)
            err_k = (yhat_kept != y_cal[idx]).sum()
            ub = wilson_upper_bound(err_k, k, z=self.params.z)
            if ub <= self.params.alpha:
                k_best = k
        self.tau_ = np.inf if k_best==0 else s[order[k_best-1]]
        return self

    def predict_with_abstain(self, X, miss_mask=None, sev_vals=None):
        P_raw, Xi = self._raw_proba(X)
        P = self.iso_.transform(P_raw) if self.iso_ is not None else P_raw
        e0 = P * self.params.cost_fn
        e1 = (1-P) * self.params.cost_fp
        emin = np.minimum(e0, e1)
        conf = np.zeros_like(P)  # (off by default for fairness; can enable)
        sev  = np.zeros_like(P) if sev_vals is None else sev_vals
        # use the same score form used at calibration
        conf = (conf - self._conf_mu_) / (self._conf_sd_ + 1e-12)
        sev  = (sev  - self._sev_mu_)  / (self._sev_sd_  + 1e-12)
        esec = np.maximum(e0, e1); margin = esec - emin
        s = margin + self.params.beta*(self.params.c_abs - emin) + self.params.w_conf*conf - self.params.w_sev*sev

        abstain = (emin > self.params.c_abs) | (s < self.tau_)
        yhat = (e1 < e0).astype(int)
        return yhat, abstain

def crc_select_threshold(probs, y, alpha=0.10, delta=0.05):
    p = probs
    e0 = p * 25.0  # FN cost scale irrelevant here; we only need errors. But keep same flow.
    e1 = (1-p) * 1.0
    # Predict class = argmin expected cost -> same as your rule
    yhat = (e1 < e0).astype(int)
    margin = np.abs(p - 0.5)  # simple confidence margin
    order = np.argsort(-margin)
    tau = np.inf
    k_best = 0
    for k in range(1, len(order)+1):
        idx = order[:k]
        err_k = (yhat[idx] != y[idx]).sum()
        ub = wilson_upper_bound(err_k, k, z=1.96)  # ~95% (delta~0.05)
        if ub <= alpha:
            k_best = k
    if k_best > 0:
        tau = margin[order[k_best-1]]
    return tau

def run_crc_vs_scorf_on_uci_shift_and_missing(seeds=10):
    X, y = LOADERS["uci"]()
    params = SCORFParams()
    out = {"Shift": {"CRC": [], "SCORF": []}, "Missing": {"CRC": [], "SCORF": []}}
    for s in range(seeds):
        # split
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42+s)
        X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=43+s)

        # ---- CRC with Logistic Regression ----
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        logit = LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs")
        logit.fit(X_tr_imp, y_tr)

        # SHIFT
        var = X_tr_imp.var(axis=0)
        feat_idx = np.argsort(-var)[:8]
        deltas = 0.5 * np.sqrt(var[feat_idx])
        def shift(M): 
            Mc = M.copy(); Mc[:, feat_idx] = Mc[:, feat_idx] + deltas; return Mc
        X_cal_s = imp.transform(shift(X_cal)); X_te_s = imp.transform(shift(X_te))
        P_cal = logit.predict_proba(X_cal_s)[:,1]
        tau = crc_select_threshold(P_cal, y_cal, alpha=params.alpha, delta=0.05)
        P_te = logit.predict_proba(X_te_s)[:,1]
        yhat = ( (1-P_te) < (P_te*25/1e-12) ).astype(int)  # same argmin(e0,e1); simplified
        abst = (np.abs(P_te-0.5) < tau)
        cost, kept_err, cov = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
        out["Shift"]["CRC"].append((cost, abst.mean()))

        # MISSING
        def mcar_local(A, p, seed):
            rng = np.random.RandomState(seed); Ac = A.copy()
            mask = rng.rand(*Ac.shape) < p; Ac[mask] = np.nan
            return Ac, mask.astype(float)
        X_tr_m,_ = mcar_local(X_tr, 0.10, 10+s)
        imp2 = SimpleImputer(strategy="median")
        logit2 = LogisticRegression(max_iter=200, solver="lbfgs")
        logit2.fit(imp2.fit_transform(X_tr_m), y_tr)
        X_cal_m, Mcal = mcar_local(X_cal, 0.20, 20+s)
        X_te_m,  Mte  = mcar_local(X_te,  0.20, 30+s)
        P_cal = logit2.predict_proba(imp2.transform(X_cal_m))[:,1]
        tau = crc_select_threshold(P_cal, y_cal, alpha=params.alpha, delta=0.05)
        P_te = logit2.predict_proba(imp2.transform(X_te_m))[:,1]
        yhat = ( (1-P_te) < (P_te*25/1e-12) ).astype(int)
        abst = (np.abs(P_te-0.5) < tau)
        cost, kept_err, cov = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
        out["Missing"]["CRC"].append((cost, abst.mean()))

        # SHIFT
        sc = SCORF(params)
        sc.fit(X_tr, y_tr)
        sc.calibrate(shift(X_cal), y_cal, condition="shift")
        yhat, abst = sc.predict_with_abstain(shift(X_te), condition="shift")
        cost, _, _ = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
        out["Shift"]["SCORF"].append((cost, abst.mean()))
        # MISSING
        sc = SCORF(params)
        X_tr_m,_ = mcar_local(X_tr, 0.10, 40+s)
        sc.fit(X_tr_m, y_tr)
        X_cal_m, Mcal = mcar_local(X_cal, 0.20, 50+s)
        X_te_m,  Mte  = mcar_local(X_te,  0.20, 60+s)
        sc.calibrate(X_cal_m, y_cal, condition="missing", miss_mask=Mcal)
        yhat, abst = sc.predict_with_abstain(X_te_m, condition="missing", miss_mask=Mte)
        cost, _, _ = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
        out["Missing"]["SCORF"].append((cost, abst.mean()))

    def msd(a): a=np.array(a); return a[:,0].mean(), a[:,0].std(), a[:,1].mean()
    print("\n[Table 3] CRC vs SCORF on UCI Credit (α=0.10)")
    for cond in ["Shift","Missing"]:
        m_crc, s_crc, a_crc = msd(out[cond]["CRC"])
        m_sc,  s_sc,  a_sc  = msd(out[cond]["SCORF"])
        print(f"{cond:8s} | CRC: cost={m_crc:.2f}±{s_crc:.2f} (abst={a_crc:.2f})   SCORF: cost={m_sc:.2f}±{s_sc:.2f} (abst={a_sc:.2f})")

def run_oblique_baselines_uci(seeds=10):
    if RotationForestClassifier is None and RerFClassifier is None:
        print("\n[Table 4] Skipped: RotationForest / RerF not installed.")
        return
    X, y = LOADERS["uci"]()
    params = SCORFParams()
    results = { "RotationForest": {"Shift": [], "Missing": []},
                "RerF":          {"Shift": [], "Missing": []},
                "SCORF":         {"Shift": [], "Missing": []}}
    for s in range(seeds):
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42+s)
        X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=43+s)

        var = np.nanvar(X_tr, axis=0); feat_idx = np.argsort(-var)[:8]; deltas = 0.5*np.sqrt(var[feat_idx])
        def shift(M): Mc=M.copy(); Mc[:,feat_idx]=Mc[:,feat_idx]+deltas; return Mc

        # Rotation Forest
        if RotationForestClassifier is not None:
            rot = RotationForestClassifier()   # defaults okay
            rotw = CostDeferWrapper(rot, params).fit(X_tr, y_tr)
            rotw.calibrate(shift(X_cal), y_cal, condition="shift")
            yhat, abst = rotw.predict_with_abstain(shift(X_te))
            cost, _, _ = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
            results["RotationForest"]["Shift"].append((cost, abst.mean()))

            X_tr_m,_ = mcar(X_tr, 0.10, seed=10+s)
            rotw = CostDeferWrapper(RotationForestClassifier(), params).fit(X_tr_m, y_tr)
            X_cal_m, Mcal = mcar(X_cal, 0.20, seed=20+s); X_te_m, Mte = mcar(X_te, 0.20, seed=30+s)
            rotw.calibrate(X_cal_m, y_cal, condition="missing", miss_mask=Mcal)
            yhat, abst = rotw.predict_with_abstain(X_te_m, miss_mask=Mte)
            cost, _, _ = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
            results["RotationForest"]["Missing"].append((cost, abst.mean()))

        # RerF
        if RerFClassifier is not None:
            rf = RerFClassifier()
            rf_w = CostDeferWrapper(rf, params).fit(X_tr, y_tr)
            rf_w.calibrate(shift(X_cal), y_cal, condition="shift")
            yhat, abst = rf_w.predict_with_abstain(shift(X_te))
            cost, _, _ = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
            results["RerF"]["Shift"].append((cost, abst.mean()))

            X_tr_m,_ = mcar(X_tr, 0.10, seed=40+s)
            rf_w = CostDeferWrapper(RerFClassifier(), params).fit(X_tr_m, y_tr)
            X_cal_m, Mcal = mcar(X_cal, 0.20, seed=50+s); X_te_m, Mte = mcar(X_te, 0.20, seed=60+s)
            rf_w.calibrate(X_cal_m, y_cal, condition="missing", miss_mask=Mcal)
            yhat, abst = rf_w.predict_with_abstain(X_te_m, miss_mask=Mte)
            cost, _, _ = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
            results["RerF"]["Missing"].append((cost, abst.mean()))

        # SCORF (reference)
        sc = SCORF(params).fit(X_tr, y_tr)
        sc.calibrate(shift(X_cal), y_cal, condition="shift")
        yhat, abst = sc.predict_with_abstain(shift(X_te), condition="shift")
        cost, _, _ = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
        results["SCORF"]["Shift"].append((cost, abst.mean()))

        sc = SCORF(params)
        X_tr_m,_ = mcar(X_tr, 0.10, seed=70+s)
        sc.fit(X_tr_m, y_tr)
        X_cal_m, Mcal = mcar(X_cal, 0.20, seed=80+s); X_te_m, Mte = mcar(X_te, 0.20, seed=90+s)
        sc.calibrate(X_cal_m, y_cal, condition="missing", miss_mask=Mcal)
        yhat, abst = sc.predict_with_abstain(X_te_m, condition="missing", miss_mask=Mte)
        cost, _, _ = policy_cost(y_te, yhat, abst, params.cost_fp, params.cost_fn, params.c_abs)
        results["SCORF"]["Missing"].append((cost, abst.mean()))

    def msd(lst): lst=np.array(lst); return lst[:,0].mean(), lst[:,0].std(), lst[:,1].mean()
    print("\n[Table 4] UCI Credit: Rotation/RerF vs SCORF (α=0.10)")
    for model in ["RotationForest","RerF","SCORF"]:
        for cond in ["Shift","Missing"]:
            if len(results[model][cond])==0: continue
            m, s, a = msd(results[model][cond])
            print(f"{model:15s} {cond:7s} cost={m:.2f}±{s:.2f} (abst={a:.2f})")

def run_per_dataset_results(seeds=10):
    params = SCORFParams()
    print("\n[Table 5] Per-dataset policy cost (Clean / Shift / Noise / Missing)")
    for ds in ["uci","heloc","gmsc"]:
        X, y = LOADERS[ds]()
        agg = {"HGBT":[], "SCORF":[]}
        for s in range(seeds):
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42+s)
            X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=43+s)

            hgbt = HistGradientBoostingClassifier(max_depth=6)
            hb = CostDeferWrapper(hgbt, params).fit(X_tr, y_tr)

            sc = SCORF(params)

            row_h, row_s = [], []

            hb.calibrate(X_cal, y_cal, condition="clean")
            yh, ab = hb.predict_with_abstain(X_te)
            c,_,_ = policy_cost(y_te, yh, ab, params.cost_fp, params.cost_fn, params.c_abs)
            row_h.append(c)

            sc.fit(X_tr, y_tr)
            sc.calibrate(X_cal, y_cal, condition="clean")
            ys, asb = sc.predict_with_abstain(X_te, condition="clean")
            c,_,_ = policy_cost(y_te, ys, asb, params.cost_fp, params.cost_fn, params.c_abs)
            row_s.append(c)

            var = X_tr.var(axis=0); feat_idx = np.argsort(-var)[:8]; deltas = 0.5*np.sqrt(var[feat_idx])
            shift = lambda M: (M.copy().__setitem__((slice(None),feat_idx), M[:,feat_idx]+deltas) or M)
            hb.calibrate(shift(X_cal), y_cal, condition="shift")
            yh, ab = hb.predict_with_abstain(shift(X_te))
            c,_,_ = policy_cost(y_te, yh, ab, params.cost_fp, params.cost_fn, params.c_abs)
            row_h.append(c)

            sc = SCORF(params).fit(X_tr, y_tr)
            sc.calibrate(shift(X_cal), y_cal, condition="shift")
            ys, asb = sc.predict_with_abstain(shift(X_te), condition="shift")
            c,_,_ = policy_cost(y_te, ys, asb, params.cost_fp, params.cost_fn, params.c_abs)
            row_s.append(c)

            # NOISE
            y_tr_n = flip_labels_asymmetric(y_tr, p10=0.15, p01=0.05, seed=100+s)
            hb = CostDeferWrapper(HistGradientBoostingClassifier(max_depth=6), params).fit(X_tr, y_tr_n)
            hb.calibrate(X_cal, y_cal, condition="noise")
            yh, ab = hb.predict_with_abstain(X_te)
            c,_,_ = policy_cost(y_te, yh, ab, params.cost_fp, params.cost_fn, params.c_abs)
            row_h.append(c)

            sc = SCORF(params).fit(X_tr, y_tr_n)
            sc.calibrate(X_cal, y_cal, condition="noise")
            ys, asb = sc.predict_with_abstain(X_te, condition="noise")
            c,_,_ = policy_cost(y_te, ys, asb, params.cost_fp, params.cost_fn, params.c_abs)
            row_s.append(c)

            # MISSING
            X_tr_m,_ = mcar(X_tr, 0.10, seed=200+s); X_cal_m, Mcal = mcar(X_cal, 0.20, seed=300+s); X_te_m, Mte = mcar(X_te, 0.20, seed=400+s)
            hb = CostDeferWrapper(HistGradientBoostingClassifier(max_depth=6), params).fit(X_tr_m, y_tr)
            hb.calibrate(X_cal_m, y_cal, condition="missing", miss_mask=Mcal)
            yh, ab = hb.predict_with_abstain(X_te_m, miss_mask=Mte)
            c,_,_ = policy_cost(y_te, yh, ab, params.cost_fp, params.cost_fn, params.c_abs)
            row_h.append(c)

            sc = SCORF(params).fit(X_tr_m, y_tr)
            sc.calibrate(X_cal_m, y_cal, condition="missing", miss_mask=Mcal)
            ys, asb = sc.predict_with_abstain(X_te_m, condition="missing", miss_mask=Mte)
            c,_,_ = policy_cost(y_te, ys, asb, params.cost_fp, params.cost_fn, params.c_abs)
            row_s.append(c)

            agg["HGBT"].append(row_h)
            agg["SCORF"].append(row_s)

        H = np.array(agg["HGBT"]).mean(axis=0)
        S = np.array(agg["SCORF"]).mean(axis=0)
        names = ["Clean","Shift","Noise","Missing"]
        print(f"\n{ds.upper()}: ", end="")
        print(" | ".join([f"HGBT {n}:{H[i]:.2f}  SCORF {n}:{S[i]:.2f}" for i,n in enumerate(names)]))

def run_temporal_gmsc(seeds=10):
    params = SCORFParams()
    X, y = LOADERS["gmsc"]()
    out = {"HGBT": [], "XGB": [], "SCORF": []}
    for s in range(seeds):
        idx_tr, idx_rest = temporal_split_indices(len(y), 0.8)
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_rest, y_rest = X[idx_rest], y[idx_rest]
        X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=43+s)

        # HGBT
        hgbt = HistGradientBoostingClassifier(max_depth=6)
        hb = CostDeferWrapper(hgbt, params).fit(X_tr, y_tr)
        hb.calibrate(X_cal, y_cal, condition="clean")
        yh, ab = hb.predict_with_abstain(X_te)
        c,_,_ = policy_cost(y_te, yh, ab, params.cost_fp, params.cost_fn, params.c_abs)
        out["HGBT"].append((c, ab.mean()))

        if XGBClassifier is not None:
            xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1)
            xb = CostDeferWrapper(xgb, params).fit(X_tr, y_tr)
            xb.calibrate(X_cal, y_cal, condition="clean")
            yh, ab = xb.predict_with_abstain(X_te)
            c,_,_ = policy_cost(y_te, yh, ab, params.cost_fp, params.cost_fn, params.c_abs)
            out["XGB"].append((c, ab.mean()))

        # SCORF
        sc = SCORF(params).fit(X_tr, y_tr)
        sc.calibrate(X_cal, y_cal, condition="clean")
        ys, asb = sc.predict_with_abstain(X_te, condition="clean")
        c,_,_ = policy_cost(y_te, ys, asb, params.cost_fp, params.cost_fn, params.c_abs)
        out["SCORF"].append((c, asb.mean()))

    def msd(L): L=np.array(L); return L[:,0].mean(), L[:,0].std(), L[:,1].mean()
    print("\n[Table 6] GMSC temporal split (natural shift):")
    m,s,a = msd(out["HGBT"]); print(f"HGBT  cost={m:.2f}±{s:.2f} abst={a:.2f}")
    if len(out["XGB"]): m,s,a = msd(out["XGB"]); print(f"XGBoost cost={m:.2f}±{s:.2f} abst={a:.2f}")
    m,s,a = msd(out["SCORF"]); print(f"SCORF cost={m:.2f}±{s:.2f} abst={a:.2f}")

def run_runtime_scalability():
    params = SCORFParams()
    X, y = LOADERS["uci"]()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.5, stratify=y, random_state=123)

    def time_fit(model, X, y, label):
        t0 = time.perf_counter()
        model.fit(X, y)
        t1 = time.perf_counter()
        print(f"{label:12s} time={t1-t0:.2f}s")
        return t1-t0

    print("\n[Table 7] Runtime (one run)")
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    t_rf = time_fit(rf, X_tr, y_tr, "RF")
    hgbt = HistGradientBoostingClassifier(max_depth=6)
    t_hg = time_fit(hgbt, X_tr, y_tr, "HGBT")
    if XGBClassifier is not None:
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1)
        t_xg = time_fit(xgb, X_tr, y_tr, "XGBoost")
    sc = SCORF(params)
    t0 = time.perf_counter(); sc.fit(X_tr, y_tr); t1 = time.perf_counter()
    t_sc = t1-t0
    print(f"SCORF        time={t_sc:.2f}s")

    print("\n[Table 7 - right] SCORF scaling with p")
    d = X_tr.shape[1]
    for p in [min(k,d) for k in [23,50,100,200]]:
        cols = np.arange(p)
        t0 = time.perf_counter()
        SCORF(params).fit(X_tr[:,cols], y_tr)
        t1 = time.perf_counter()
        print(f"p={p:3d}  time={t1-t0:.2f}s")

def run_hyperparam_sensitivity_uci_shift(seeds=10):
    X, y = LOADERS["uci"]()
    params = SCORFParams()
    def eval_cost(pmod):
        costs=[]
        for s in range(seeds):
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42+s)
            X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=43+s)
            sc = SCORF(pmod).fit(X_tr, y_tr)
            var = X_tr.var(axis=0); feat_idx = np.argsort(-var)[:8]; deltas = 0.5*np.sqrt(var[feat_idx])
            def shift(M): Mc=M.copy(); Mc[:,feat_idx]=Mc[:,feat_idx]+deltas; return Mc
            sc.calibrate(shift(X_cal), y_cal, condition="shift")
            yh, ab = sc.predict_with_abstain(shift(X_te), condition="shift")
            c,_,_ = policy_cost(y_te, yh, ab, pmod.cost_fp, pmod.cost_fn, pmod.c_abs)
            costs.append(c)
        return np.mean(costs)

    base = params
    print("\n[Table 8] Sensitivity on UCI Credit (Shift) – report cost")
    for name in ["gamma","rho","beta","w_conf","w_sev"]:
        v = getattr(base, name)
        for scale in [0.75, 1.0, 1.25]:
            pmod = dataclasses.replace(base, **{name: v*scale})
            c = eval_cost(pmod)
            print(f"{name:7s} x{scale:>4} -> cost={c:.2f}")

def run_surrogate_ablation_uci_shift(seeds=10, m_for_G=1000):
    X, y = LOADERS["uci"]()
    params = SCORFParams()

    def run_with_surrogate(build_surrogate, label):
        costs, times = [], []
        for s in range(seeds):
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42+s)
            X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=43+s)
            sc = SCORF(params)

            X_imp = sc.imp_.fit_transform(X_tr)
            n = X_imp.shape[0]
            idx = sc.rng.choice(n, size=min(params.grad_sample, n), replace=False)
            Xs, ys = X_imp[idx], y_tr[idx]

            # surrogate
            t0 = time.perf_counter()
            sur = build_surrogate()
            sur.fit(Xs, ys)
            # compute finite-diff gradient using surrogate probs
            G = sc._finite_diff_grad(sur, Xs)
            S = (G.T @ G) / G.shape[0]
            vals, vecs = np.linalg.eigh(S)
            sc.H_ = vecs @ np.diag(vals / (vals + params.gamma)) @ vecs.T
            t1 = time.perf_counter()

            # augmentation + final RF as in SCORF.fit
            m_aug = int(params.aug_frac * X_imp.shape[0])
            id_aug = sc.rng.choice(X_imp.shape[0], size=m_aug, replace=False)
            Xa = X_imp[id_aug]
            Ga = sc._finite_diff_grad(sur, Xa)
            Hg = Ga @ sc.H_.T
            U = Hg / (np.linalg.norm(Hg, axis=1, keepdims=True) + 1e-12)
            X_plus, X_minus = Xa + params.rho*U, Xa - params.rho*U
            y_aug = y_tr[id_aug]
            XH = np.vstack([X_imp @ sc.H_, X_plus @ sc.H_, X_minus @ sc.H_])
            y_all = np.concatenate([y_tr, y_aug, y_aug])
            sc.rf_ = sc._rf(params.n_trees_final).fit(XH, y_all)

            var = X_tr.var(axis=0); feat_idx = np.argsort(-var)[:8]; deltas = 0.5*np.sqrt(var[feat_idx])
            shift = lambda M: (M.copy().__setitem__((slice(None),feat_idx), M[:,feat_idx]+deltas) or M)
            sc.calibrate(shift(X_cal), y_cal, condition="shift")
            yh, ab = sc.predict_with_abstain(shift(X_te), condition="shift")
            c,_,_ = policy_cost(y_te, yh, ab, params.cost_fp, params.cost_fn, params.c_abs)
            costs.append(c); times.append(t1-t0)
        print(f"{label:16s} cost={np.mean(costs):.2f}±{np.std(costs):.2f}  time_for_G≈{np.mean(times):.2f}s")

    print("\n[Table 9] Surrogate for G on UCI Credit (Shift)")
    run_with_surrogate(lambda: RandomForestClassifier(n_estimators=20, max_depth=10, random_state=0), "Shallow RF")
    run_with_surrogate(lambda: LogisticRegression(max_iter=200, solver='lbfgs'), "Logistic")
    run_with_surrogate(lambda: HistGradientBoostingClassifier(max_depth=3), "Shallow HGBT")

if __name__ == "__main__":
    # (C) CRC vs SCORF on UCI (Shift & Missing)
    run_crc_vs_scorf_on_uci_shift_and_missing(seeds=10)

    # (D) Rotation/RerF vs SCORF (UCI)
    run_oblique_baselines_uci(seeds=10)

    # (E) Per-dataset results: HGBT vs SCORF
    run_per_dataset_results(seeds=10)

    # (F) Natural temporal shift on GMSC
    run_temporal_gmsc(seeds=10)

    # (G) Runtime & scalability
    run_runtime_scalability()

    # (H) Hyperparameter sensitivity on UCI (Shift)
    import dataclasses
    run_hyperparam_sensitivity_uci_shift(seeds=10)

    # (I) Surrogate ablation (RF vs Logit vs shallow HGBT) on UCI (Shift)
    run_surrogate_ablation_uci_shift(seeds=10, m_for_G=1000)

