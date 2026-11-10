"""Core of the study: detector, conformal baselines and the proposed
Feedback-Adaptive Conformal Triage (FACT).

A stream of flows arrives in batches. Every method returns a prediction set
per flow. A singleton set is decided automatically; any other set (empty or
ambiguous) is escalated to an analyst, who returns the true label after a
delay, within a per-batch capacity. FACT additionally audits a random
fraction eps of the automatically decided flows, so that the labels it
receives have known inclusion probabilities, and uses inverse-probability-
weighted, class-conditional adaptive conformal updates plus periodic
detector refresh from analyst labels.
"""
import time
import numpy as np
import lightgbm as lgb

# ----------------------------------------------------------------- detector


class Detector:
    """LightGBM multiclass detector over a global label space of size K.
    Classes absent from the training data receive probability 0."""

    def __init__(self, K, threads=16, rounds=200, leaves=63, seed=0):
        self.K, self.threads, self.rounds, self.leaves, self.seed = K, threads, rounds, leaves, seed
        self.known = np.zeros(K, bool)
        self.booster = None

    def fit(self, X, y, w=None):
        cls = np.unique(y)
        self.known[:] = False
        self.known[cls] = True
        self.map = cls
        inv = np.full(self.K, -1)
        inv[cls] = np.arange(len(cls))
        yy = inv[y]
        params = dict(objective="multiclass", num_class=len(cls), num_leaves=self.leaves,
                      learning_rate=0.1, min_data_in_leaf=20, feature_fraction=0.9,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      num_threads=self.threads, seed=self.seed, verbose=-1)
        if len(cls) == 1:
            self.booster = None
            return self
        ds = lgb.Dataset(X, yy, weight=w, free_raw_data=True)
        self.booster = lgb.train(params, ds, num_boost_round=self.rounds)
        return self

    def proba(self, X):
        P = np.zeros((len(X), self.K), np.float32)
        if self.booster is None:
            P[:, self.map[0]] = 1.0
            return P
        P[:, self.map] = self.booster.predict(X, num_threads=self.threads).astype(np.float32)
        return P


class ETDetector(Detector):
    """Extremely randomised trees (scikit-learn) with the same interface, used to
    check that FACT does not depend on the gradient-boosted detector."""

    def fit(self, X, y, w=None):
        from sklearn.ensemble import ExtraTreesClassifier
        cls = np.unique(y)
        self.known[:] = False
        self.known[cls] = True
        self.map = cls
        n_est = 60 if self.rounds <= 100 else 100
        self.booster = ExtraTreesClassifier(n_estimators=n_est, min_samples_leaf=2, max_features="sqrt",
                                            n_jobs=self.threads, random_state=self.seed)
        Xf = np.nan_to_num(X, nan=-1.0)
        self.booster.fit(Xf, y, sample_weight=w)
        return self

    def proba(self, X):
        P = np.zeros((len(X), self.K), np.float32)
        P[:, self.booster.classes_] = self.booster.predict_proba(np.nan_to_num(X, nan=-1.0)).astype(np.float32)
        return P


# --------------------------------------------------------------- utilities


def q_split(s, alpha):
    """Finite-sample split-conformal quantile (inf if too few points)."""
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return np.partition(s, k - 1)[k - 1]


def q_weighted(s, w, level, w_test=0.0):
    """Weighted conformal quantile: smallest t with sum_{s_i<=t} w_i >= level*(W+w_test)."""
    if len(s) == 0:
        return np.inf
    o = np.argsort(s)
    s, w = s[o], w[o]
    cw = np.cumsum(w)
    T = level * (cw[-1] + w_test)
    j = np.searchsorted(cw, T)
    return np.inf if j >= len(s) else s[j]


def lac_scores(P, y):
    return 1.0 - P[np.arange(len(y)), y]


def aps_scores(P, y, u):
    py = P[np.arange(len(y)), y]
    above = (P > py[:, None]).astype(np.float32) * P
    return above.sum(1) + u * py


def aps_sets(P, q, u, lam=0.0, kreg=0):
    """(R)APS sets: class j included if its randomized cumulative score <= q."""
    o = np.argsort(-P, 1)
    Ps = np.take_along_axis(P, o, 1)
    cum = np.cumsum(Ps, 1) - Ps
    ranks = np.arange(1, P.shape[1] + 1)[None, :]
    pen = lam * np.maximum(0, ranks - kreg)
    score_sorted = cum + u[:, None] * Ps + pen
    inc_sorted = score_sorted <= q
    inc_sorted[:, 0] |= False
    S = np.zeros_like(inc_sorted)
    np.put_along_axis(S, o, inc_sorted, 1)
    return S


def raps_scores(P, y, u, lam, kreg):
    o = np.argsort(-P, 1)
    rank = np.argmax(o == y[:, None], 1) + 1
    py = P[np.arange(len(y)), y]
    above = (P > py[:, None]).astype(np.float32) * P
    return above.sum(1) + u * py + lam * np.maximum(0, rank - kreg)


# ------------------------------------------------------------------ methods


class Method:
    name = "base"
    uses_feedback = False
    refreshes = False
    audit = 0.0
    oracle_labels = False

    def __init__(self, alpha):
        self.alpha = alpha

    def setup(self, det, Xcal, ycal, rng):
        self.det, self.rng = det, rng
        self.Xcal, self.ycal = Xcal, ycal
        self.calibrate()

    def calibrate(self):
        pass

    def sets(self, P, Xb):
        raise NotImplementedError

    def observe_unlabeled(self, Xb, P):
        pass

    def feedback(self, X, y, P, S, pi):
        pass


class SplitLAC(Method):
    name = "Split-CP (LAC)"

    def calibrate(self):
        P = self.det.proba(self.Xcal)
        self.q = q_split(lac_scores(P, self.ycal), self.alpha)

    def sets(self, P, Xb):
        return ((1 - P) <= self.q) & self.det.known[None, :]


class Mondrian(Method):
    name = "Mondrian-CP"

    def calibrate(self):
        P = self.det.proba(self.Xcal)
        s = lac_scores(P, self.ycal)
        self.q = np.full(self.det.K, np.inf)
        for c in range(self.det.K):
            if self.det.known[c]:
                self.q[c] = q_split(s[self.ycal == c], self.alpha)

    def sets(self, P, Xb):
        return ((1 - P) <= self.q[None, :]) & self.det.known[None, :]


class APS(Method):
    name = "APS"
    lam, kreg = 0.0, 0

    def calibrate(self):
        P = self.det.proba(self.Xcal)
        u = self.rng.random(len(self.ycal))
        s = raps_scores(P, self.ycal, u, self.lam, self.kreg)
        self.q = q_split(s, self.alpha)

    def sets(self, P, Xb):
        u = self.rng.random(len(P))
        return aps_sets(P, self.q, u, self.lam, self.kreg) & self.det.known[None, :]


class RAPS(APS):
    name = "RAPS"
    lam, kreg = 0.01, 2


class WeightedCP(Method):
    """Covariate-shift conformal (Tibshirani et al., 2019): calibration
    scores reweighted by a density ratio estimated with a domain classifier
    trained on source calibration vs. recent unlabeled deployment flows."""
    name = "Weighted-CP"

    def calibrate(self):
        P = self.det.proba(self.Xcal)
        self.s = lac_scores(P, self.ycal)
        self.win = []
        self.dom = None

    def observe_unlabeled(self, Xb, P):
        self.win.append(Xb)
        if len(self.win) > 10:
            self.win.pop(0)
        self.nb = getattr(self, "nb", 0) + 1
        if self.dom is not None and self.nb % 10:
            return
        Xt = np.concatenate(self.win)
        m = min(len(Xt), len(self.Xcal), 20000)
        a = self.Xcal[self.rng.choice(len(self.Xcal), m, replace=False)]
        b = Xt[self.rng.choice(len(Xt), m, replace=False)]
        X = np.concatenate([a, b]).astype(np.float32)
        y = np.r_[np.zeros(m), np.ones(m)].astype(int)
        params = dict(objective="binary", num_leaves=15, learning_rate=0.1, num_threads=self.det.threads,
                      verbose=-1, seed=0)
        self.dom = lgb.train(params, lgb.Dataset(X, y), num_boost_round=50)
        pc = np.clip(self.dom.predict(self.Xcal, num_threads=self.det.threads), 1e-4, 1 - 1e-4)
        self.wcal = pc / (1 - pc)

    def sets(self, P, Xb):
        if self.dom is None:
            q = q_split(self.s, self.alpha)
            return ((1 - P) <= q) & self.det.known[None, :]
        pt = np.clip(self.dom.predict(Xb, num_threads=self.det.threads), 1e-4, 1 - 1e-4)
        wt = pt / (1 - pt)
        o = np.argsort(self.s)
        s, cw = self.s[o], np.cumsum(self.wcal[o])
        T = (1 - self.alpha) * (cw[-1] + wt)
        j = np.searchsorted(cw, T)
        q = np.where(j >= len(s), np.inf, s[np.minimum(j, len(s) - 1)])
        return ((1 - P) <= q[:, None]) & self.det.known[None, :]


class LabelShiftCP(Method):
    """Label-shift conformal (Podkopaev & Ramdas, 2021): calibration scores
    reweighted by target/source class-prior ratios estimated with BBSE from
    unlabeled deployment flows."""
    name = "LabelShift-CP"

    def calibrate(self):
        P = self.det.proba(self.Xcal)
        self.s = lac_scores(P, self.ycal)
        K = self.det.K
        yhat = P.argmax(1)
        self.C = np.zeros((K, K))
        np.add.at(self.C, (yhat, self.ycal), 1.0)
        self.C /= len(self.ycal)
        self.ps = np.bincount(self.ycal, minlength=K) / len(self.ycal)
        self.r = np.ones(K)
        self.hist = []

    def observe_unlabeled(self, Xb, P):
        self.hist.append(np.bincount(P.argmax(1), minlength=self.det.K))
        if len(self.hist) > 10:
            self.hist.pop(0)
        mu = np.sum(self.hist, 0) / np.sum(self.hist)
        k = self.det.known
        A = self.C[np.ix_(k, k)]
        try:
            w = np.linalg.lstsq(A, mu[k], rcond=None)[0]
        except Exception:
            return
        r = np.ones(self.det.K)
        r[k] = np.clip(np.nan_to_num(w / np.maximum(self.ps[k], 1e-12), nan=1.0), 0.0, 50.0)
        self.r = r

    def sets(self, P, Xb):
        w = self.r[self.ycal]
        q = q_weighted(self.s, w, 1 - self.alpha, w_test=w.max())
        return ((1 - P) <= q) & self.det.known[None, :]


class ACI(Method):
    """Adaptive conformal inference (Gibbs & Candes, 2021) on the LAC score.
    oracle=True: every label is revealed (unrealistic upper reference).
    oracle=False: only analyst labels of escalated flows (no correction)."""
    uses_feedback = True

    def __init__(self, alpha, oracle=True, gamma=0.05):
        super().__init__(alpha)
        self.oracle, self.gamma = oracle, gamma
        self.name = "ACI (all labels)" if oracle else "ACI (analyst labels)"

    def calibrate(self):
        P = self.det.proba(self.Xcal)
        self.s = lac_scores(P, self.ycal)
        self.at = self.alpha

    def _q(self):
        a = self.at
        if a <= 0:
            return np.inf
        if a >= 1:
            return -np.inf
        return q_split(self.s, a)

    def sets(self, P, Xb):
        return ((1 - P) <= self._q()) & self.det.known[None, :]

    def feedback(self, X, y, P, S, pi):
        err = ~S[np.arange(len(y)), y]
        # batch form of the ACI recursion: one step per feedback batch
        self.at = float(np.clip(self.at + self.gamma * (self.alpha - err.mean()), -0.2, 1.0))


class FACT(Method):
    """Feedback-Adaptive Conformal Triage (proposed).

    - class-conditional (Mondrian) prediction sets;
    - per-class adaptive levels alpha_c updated from analyst labels with
      inverse-probability weights (escalated: pi=1, audited: pi=eps), which
      makes the update an unbiased estimate of the per-class coverage error
      even though labels are only revealed for a biased subset of flows;
    - the base quantile mixes source calibration scores with recent labeled
      deployment scores (IPW-weighted);
    - the detector is refreshed from source data plus analyst-labelled
      deployment flows (incl. newly emerging classes)."""
    uses_feedback = True
    refreshes = True

    def __init__(self, alpha, eps=0.02, gamma=0.05, ipw=True, aci=True, refresh=True,
                 window=4000, name="FACT (proposed)", guard=0):
        super().__init__(alpha)
        self.guard = guard  # rare-class guard: alpha_c may not exceed alpha until a class has `guard` labels
        self.audit, self.gamma, self.ipw, self.aci, self.refreshes = eps, gamma, ipw, aci, refresh
        self.window, self.name = window, name

    def calibrate(self):
        P = self.det.proba(self.Xcal)
        self.scal = lac_scores(P, self.ycal)
        K = self.det.K
        if not hasattr(self, "ac"):
            self.ac = np.full(K, self.alpha)
            self.nc = np.zeros(K)
            self.lc = np.zeros(K)
            self.tX, self.ty, self.tw = [], [], []
        self.tscore = self._target_scores()
        self._update_q()

    def _target_scores(self):
        if not self.ty:
            return np.zeros(0, np.float32)
        X = np.concatenate(self.tX)
        y = np.concatenate(self.ty)
        return lac_scores(self.det.proba(X), y)

    def _update_q(self):
        """Base quantile per class: source calibration scores when the class
        has enough of them, otherwise the (IPW-weighted) analyst-labelled
        deployment scores; the adaptive level alpha_c corrects the base."""
        K = self.det.K
        q = np.full(K, np.inf)
        ty = np.concatenate(self.ty) if self.ty else np.zeros(0, int)
        tw = np.concatenate(self.tw) if self.tw else np.zeros(0)
        for c in range(K):
            if not self.det.known[c]:
                continue
            a = self.ac[c]
            if a <= 0:
                continue
            if a >= 1:
                q[c] = -np.inf
                continue
            sc = self.scal[self.ycal == c]
            if len(sc) >= 50:
                q[c] = q_weighted(sc, np.ones(len(sc)), 1 - a, w_test=1.0)
            else:
                m = ty == c
                if self.guard and len(sc) > 0:
                    # rare-class guard: pool the few source scores with the labelled deployment scores;
                    # deployment weights are Hajek-normalised to their count so that the finite-sample
                    # correction reflects the real number of labelled flows
                    wd = tw[m] * (m.sum() / tw[m].sum()) if m.any() else np.zeros(0)
                    q[c] = q_weighted(np.r_[sc, self.tscore[m]], np.r_[np.ones(len(sc)), wd], 1 - a, w_test=1.0)
                else:
                    q[c] = q_weighted(self.tscore[m], tw[m], 1 - a, w_test=1.0) if m.any() else np.inf
        self.q = q

    def sets(self, P, Xb):
        return ((1 - P) <= self.q[None, :]) & self.det.known[None, :]

    def feedback(self, X, y, P, S, pi):
        if len(y) == 0:
            return
        w = (1.0 / pi) if self.ipw else np.ones(len(y))
        err = ~S[np.arange(len(y)), y]
        if self.aci:
            # IPW estimate of the class-c coverage error in this feedback batch,
            # scaled by a smoothed estimate of the class-c flow count per batch
            for c in np.unique(y):
                m = y == c
                self.nc[c] = 0.8 * self.nc[c] + 0.2 * w[m].sum() if self.nc[c] > 0 else w[m].sum()
                # Hajek-type normalisation: one audited flow (weight 1/eps) never moves alpha_c by more than one step
                upd = np.sum(w[m] * (self.alpha - err[m])) / max(self.nc[c], float(w[m].max()), 1.0)
                self.lc[c] += m.sum()
                hi = self.alpha if (self.guard and self.lc[c] < self.guard) else 0.6
                self.ac[c] = float(np.clip(self.ac[c] + self.gamma * upd, -0.2, hi))
        # store labeled deployment flows for calibration and refresh
        self.tX.append(X)
        self.ty.append(y)
        self.tw.append(w)
        n = sum(len(v) for v in self.ty)
        while n - len(self.ty[0]) >= self.window * 3 and len(self.ty) > 1:
            n -= len(self.ty[0])
            self.tX.pop(0), self.ty.pop(0), self.tw.pop(0)
        s_new = lac_scores(P, y)
        self.tscore = np.r_[self.tscore, s_new][-sum(len(v) for v in self.ty):]
        self._update_q()

    def labeled_buffer(self):
        if not self.ty:
            return None, None
        return np.concatenate(self.tX), np.concatenate(self.ty)


class RefreshRandom(Mondrian):
    """Mondrian-CP whose detector is refreshed from the SAME number of
    analyst labels as FACT, but drawn uniformly at random (active-learning
    control without targeted feedback or adaptive levels)."""
    name = "Mondrian + random-label refresh"
    refreshes = True
    uses_feedback = True

    def calibrate(self):
        super().calibrate()
        if not hasattr(self, "tX"):
            self.tX, self.ty = [], []

    def feedback(self, X, y, P, S, pi):
        self.tX.append(X)
        self.ty.append(y)

    def labeled_buffer(self):
        if not self.ty:
            return None, None
        return np.concatenate(self.tX), np.concatenate(self.ty)


class MaxProb(Method):
    """Non-conformal selective classifier: abstain when max probability is
    below the source threshold that accepts 1-alpha of calibration flows."""
    name = "Max-prob selective"

    def calibrate(self):
        P = self.det.proba(self.Xcal)
        self.t = np.quantile(P.max(1), self.alpha)

    def sets(self, P, Xb):
        S = np.zeros_like(P, bool)
        top = P.argmax(1)
        ok = P.max(1) >= self.t
        S[np.arange(len(P)), top] = True
        S[~ok] = True  # ambiguous -> escalate
        S &= self.det.known[None, :]
        return S


# ------------------------------------------------------------------ runner


def run_stream(method, det, src, stream, K, attack_mask, alpha=0.05, batch=2000, capacity=0.10,
               delay=5, refresh_every=50, src_cap=200000, refresh_rounds=100, seed=0, record_every=5,
               P_all=None, label_noise=0.0):
    """Run one method on one deployment stream.

    det: detector already trained on src (a private copy if the method refreshes it).
    src: dict Xtr,ytr,Xcal,ycal; stream: dict X,y (deployment, in arrival order).
    capacity: max fraction of each batch the analyst can label (escalations beyond it are auto-decided).
    delay: analyst labels arrive this many batches later.
    refresh_every: detector refresh period in batches (refreshing methods only).
    P_all: optional cached detector probabilities for the stream (static detector only).
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    method.setup(det, src["Xcal"], src["ycal"], rng)
    X, y = stream["X"], stream["y"]
    n = len(y)
    nb = int(np.ceil(n / batch))
    cap = max(1, int(capacity * batch))
    pending = []
    cov_c = np.zeros(K)
    cnt_c = np.zeros(K)
    auto = auto_ok = esc = aud = 0
    att_auto = att_auto_miss = ben_auto = ben_auto_fa = 0
    total_err = 0
    setsize = 0.0
    series = []
    infer_t = 0.0
    nref = 0
    demand = 0
    silent_c = np.zeros(K)
    for b in range(nb):
        sl = slice(b * batch, min(n, (b + 1) * batch))
        Xb, yb = X[sl], y[sl]
        ti = time.time()
        P = P_all[sl] if (P_all is not None and not method.refreshes) else det.proba(Xb)
        method.observe_unlabeled(Xb, P)
        S = method.sets(P, Xb)
        infer_t += time.time() - ti
        size = S.sum(1)
        idx_w = np.flatnonzero(size != 1)
        demand += len(idx_w)
        if len(idx_w) > cap:
            idx_w = idx_w[np.argsort(P[idx_w].max(1))[:cap]]
        escm = np.zeros(len(yb), bool)
        escm[idx_w] = True
        autom = ~escm
        pred = P.argmax(1)
        audm = (autom & (rng.random(len(yb)) < method.audit)) if method.audit > 0 else np.zeros(len(yb), bool)
        inset = S[np.arange(len(yb)), yb]
        np.add.at(cov_c, yb, inset)
        np.add.at(cnt_c, yb, 1)
        setsize += size.sum()
        auto += autom.sum()
        ok = pred == yb
        auto_ok += (ok & autom).sum()
        esc += escm.sum()
        aud += audm.sum()
        total_err += (~ok & autom).sum()
        np.add.at(silent_c, yb, (~ok & autom))
        isatt = attack_mask[yb]
        predatt = attack_mask[pred]
        att_auto += (isatt & autom).sum()
        att_auto_miss += (isatt & autom & ~predatt).sum()
        ben_auto += (~isatt & autom).sum()
        ben_auto_fa += (~isatt & autom & predatt).sum()
        if method.uses_feedback:
            if (isinstance(method, ACI) and method.oracle) or method.oracle_labels:
                lm = np.ones(len(yb), bool)
                pi = np.ones(len(yb))
            elif isinstance(method, RefreshRandom):
                k = int(escm.sum() + int(round(0.02 * autom.sum())))
                lm = np.zeros(len(yb), bool)
                lm[rng.choice(len(yb), min(k, len(yb)), replace=False)] = True
                pi = np.ones(len(yb))
            else:
                lm = escm | audm
                pi = np.where(escm, 1.0, max(method.audit, 1e-9))
            if lm.any():
                yl_obs = yb[lm].copy()
                if label_noise > 0:  # analyst mislabels a fraction of the revealed flows
                    flip = rng.random(len(yl_obs)) < label_noise
                    yl_obs[flip] = rng.integers(0, K, flip.sum())
                pending.append((b + delay, Xb[lm], yl_obs, P[lm], S[lm], pi[lm]))
        due = [p for p in pending if p[0] <= b]
        pending = [p for p in pending if p[0] > b]
        for _, Xl, yl, Pl, Sl, pil in due:
            method.feedback(Xl, yl, Pl, Sl, pil)
        novel_ready = False
        if method.refreshes and (b + 1) % 5 == 0 and (b + 1) % refresh_every:
            Xl, yl = method.labeled_buffer()
            if yl is not None:
                u, cn = np.unique(yl, return_counts=True)
                novel_ready = bool(np.any((~det.known[u]) & (cn >= 30)))
        if method.refreshes and ((b + 1) % refresh_every == 0 or novel_ready):
            Xl, yl = method.labeled_buffer()
            if yl is not None and len(yl) > 0:
                m = min(src_cap, len(src["ytr"]))
                si = rng.choice(len(src["ytr"]), m, replace=False)
                Xn = np.concatenate([src["Xtr"][si], Xl])
                yn = np.concatenate([src["ytr"][si], yl])
                wn = np.r_[np.ones(m), np.full(len(yl), max(1.0, m / (4 * len(yl))))]
                det.rounds = refresh_rounds
                det.fit(Xn, yn, wn)
                method.calibrate()
                nref += 1
        if b % record_every == 0 or b == nb - 1:
            series.append([int(sl.stop), round(float(autom.mean()), 4), round(float(inset.mean()), 4),
                           round(float((~ok & autom).mean()), 5), round(float(escm.mean()), 4)])
    valid = cnt_c >= 50
    pcov = np.where(cnt_c > 0, cov_c / np.maximum(cnt_c, 1), np.nan)
    return dict(method=method.name, n=int(n), coverage=float(cov_c.sum() / n),
                worst_class_cov=float(np.nanmin(pcov[valid])) if valid.any() else float("nan"),
                mean_class_cov=float(np.nanmean(pcov[valid])) if valid.any() else float("nan"),
                per_class_cov={int(c): round(float(pcov[c]), 4) for c in np.flatnonzero(cnt_c > 0)},
                per_class_n={int(c): int(cnt_c[c]) for c in np.flatnonzero(cnt_c > 0)},
                automation=float(auto / n), auto_acc=float(auto_ok / max(auto, 1)),
                escalation=float(esc / n), audit=float(aud / n), analyst_load=float((esc + aud) / n),
                e2e_error=float(total_err / n), attack_miss=float(att_auto_miss / max(att_auto, 1)),
                false_alarm=float(ben_auto_fa / max(ben_auto, 1)), mean_set=float(setsize / n),
                infer_us_per_flow=float(1e6 * infer_t / n), refreshes=nref,
                demand=float(demand / n),
                worst_class_silent=float(np.nanmax(np.where(valid, silent_c / np.maximum(cnt_c, 1), np.nan)))
                if valid.any() else float("nan"),
                per_class_silent={int(c): round(float(silent_c[c] / cnt_c[c]), 5) for c in np.flatnonzero(cnt_c > 0)},
                wall_s=float(time.time() - t0), series=series)
