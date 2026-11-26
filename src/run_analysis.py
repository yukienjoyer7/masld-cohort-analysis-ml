#!/usr/bin/env python3
"""Reproducible downstream multicohort MASLD microbiome analysis.

The primary estimate uses leave-one-cohort-out validation. Feature filtering,
transformation, dimensionality reduction, model selection, and threshold
selection are fitted without access to the held-out cohort.

The abundance-table inputs are intermediate outputs produced by the study's
upstream NCBI SRA -> QIIME 2 -> DADA2 -> SILVA workflow. This module begins at
genus harmonization and does not reproduce the upstream processing commands.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import torch
from scipy.linalg import pinv
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import clone
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "OTU-table"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "visualization"
SEED = 20260918


@dataclass(frozen=True)
class CohortSpec:
    accession: str
    otu: str
    metadata: str
    country: str
    platform: str
    label_field: str
    control_rule: str


COHORTS = [
    CohortSpec("PRJNA1089365", "PRJNA1089365_otu-table-filtered.tsv", "PRJNA1089365_metadata-filtered.csv", "Thailand", "MiSeq", "Sample Name", "Healthy"),
    CohortSpec("PRJNA1152299", "PRJNA1152299_otu-table-filtered.tsv", "PRJNA1152299_metadata-filtered.csv", "Thailand", "MiniSeq", "Sample Name", "Healthy"),
    CohortSpec("PRJNA1370285", "PRJNA1370285_otu-table.tsv", "PRJNA1370285_metadata.csv", "China", "not reported", "disease", "Control"),
    CohortSpec("PRJNA720274", "PRJNA720274_otu-table.tsv", "PRJNA720274_metadata.csv", "Spain", "MiSeq", "Host_disease", "control"),
    CohortSpec("PRJNA851946", "PRJNA851946_otu-table.tsv", "PRJNA851946_metadata.csv", "China", "NovaSeq 6000", "Sample Name", "CG"),
    CohortSpec("PRJNA962007", "PRJNA962007_otu-table.tsv", "PRJNA962007_metadata.csv", "Spain", "not reported", "disease", "Control"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def extract_genus(taxon: str) -> str | None:
    taxon = str(taxon).strip()
    if not taxon or taxon.lower() == "nan" or taxon.startswith("Unassigned"):
        return None
    parts = [x.strip() for x in taxon.split(";")]
    genus = next((x[3:] for x in parts if x.startswith("g__")), "")
    if genus and genus not in {"Incertae_Sedis", "uncultured", "metagenome"}:
        return genus
    family = next((x[3:] for x in parts if x.startswith("f__")), "")
    if family and family not in {"Incertae_Sedis"}:
        return f"unclassified_{family}"
    return None


def read_otu(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    lines = [line for line in lines if not line.startswith("# Constructed") and line.strip()]
    frame = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t")
    feature_col = frame.columns[0]
    taxonomy_col = "taxonomy" if "taxonomy" in frame.columns else None
    sample_cols = [c for c in frame.columns if c not in {feature_col, taxonomy_col}]
    labels = frame[taxonomy_col].where(frame[taxonomy_col].notna() & (frame[taxonomy_col].astype(str) != ""), frame[feature_col]) if taxonomy_col else frame[feature_col]
    genera = labels.map(extract_genus)
    values = frame[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values.index = genera
    values = values.loc[values.index.notna()]
    values = values.groupby(level=0, sort=True).sum().T
    values.index.name = "sample_id"
    return values


def derive_label(row: pd.Series, spec: CohortSpec) -> int:
    value = str(row[spec.label_field]).strip()
    if spec.label_field == "Sample Name":
        return int(not value.startswith(spec.control_rule))
    return int(value.lower() != spec.control_rule.lower())


def load_data(cohorts: list[CohortSpec] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrices, metadata, audits = [], [], []
    for spec in COHORTS if cohorts is None else cohorts:
        otu = read_otu(DATA_DIR / spec.otu)
        meta = pd.read_csv(DATA_DIR / spec.metadata, dtype=str)
        sample_col = "sample-id" if "sample-id" in meta.columns else "Run"
        meta = meta.set_index(sample_col, drop=False)
        common = [s for s in otu.index if s in meta.index]
        missing_otu = sorted(set(meta.index) - set(otu.index))
        missing_meta = sorted(set(otu.index) - set(meta.index))
        if not common:
            raise ValueError(f"No matching samples for {spec.accession}")
        matched_before_qc = len(common)
        otu = otu.loc[common]
        zero_assigned = otu.index[otu.sum(axis=1) <= 0].tolist()
        if zero_assigned:
            otu = otu.drop(index=zero_assigned)
            common = otu.index.tolist()
        m = meta.loc[common].copy()
        m["sample_id"] = common
        m["cohort"] = spec.accession
        m["country"] = spec.country
        m["platform_short"] = spec.platform
        m["label"] = m.apply(lambda r: derive_label(r, spec), axis=1)
        m["diagnosis"] = m["label"].map({0: "Control", 1: "MASLD"})
        matrices.append(otu)
        metadata.append(m[["sample_id", "cohort", "country", "platform_short", "label", "diagnosis"]])
        audits.append({
            "cohort": spec.accession,
            "metadata_n": int(len(meta)),
            "otu_n": int(matched_before_qc),
            "matched_before_qc": int(matched_before_qc),
            "analyzed_n": int(len(common)),
            "cases": int(m["label"].sum()),
            "controls": int((1 - m["label"]).sum()),
            "raw_genera": int(otu.shape[1]),
            "missing_from_otu": missing_otu,
            "missing_from_metadata": missing_meta,
            "excluded_zero_assigned_abundance": zero_assigned,
        })
    all_features = sorted(set().union(*(x.columns for x in matrices)))
    x = pd.concat([m.reindex(columns=all_features, fill_value=0.0) for m in matrices], axis=0)
    meta = pd.concat(metadata, axis=0).set_index("sample_id").loc[x.index]
    audit = pd.DataFrame(audits)
    return x.astype(float), meta, audit


def counts_to_clr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    totals = x.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("At least one sample has zero total abundance")
    rel = x / totals
    replacement = 0.5 / totals
    rel = np.where(x > 0, rel, replacement)
    rel /= rel.sum(axis=1, keepdims=True)
    logged = np.log(rel)
    return logged - logged.mean(axis=1, keepdims=True)


class FeatureTransformer:
    def __init__(self, method: str, k: int, seed: int = SEED, min_cohorts: int = 2):
        self.method, self.k, self.seed, self.min_cohorts = method, k, seed, min_cohorts

    def fit(self, x: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
        any_training = (x > 0).any(axis=0).to_numpy()
        self.base_columns_ = x.columns[any_training].tolist()
        base_values = x[self.base_columns_].to_numpy()
        present = base_values > 0
        min_n = max(5, int(math.ceil(0.10 * len(x))))
        cohort_presence = np.vstack([present[groups == g].any(axis=0) for g in np.unique(groups)]).sum(axis=0)
        keep = (present.sum(axis=0) >= min_n) & (cohort_presence >= self.min_cohorts)
        if keep.sum() < 5:
            raise ValueError("Too few genera after training-only prevalence filtering")
        self.keep_indices_ = np.flatnonzero(keep)
        self.columns_ = np.asarray(self.base_columns_)[keep].tolist()
        clr = counts_to_clr(base_values)[:, self.keep_indices_]
        self.scaler_ = StandardScaler().fit(clr)
        z = self.scaler_.transform(clr)
        n_components = min(self.k, z.shape[1], z.shape[0] - 2)
        if self.method == "Differential":
            scores, _ = f_classif(z, y)
            scores = np.nan_to_num(scores, nan=-np.inf)
            self.indices_ = np.argsort(scores)[::-1][:n_components]
            self.reducer_ = None
        elif self.method == "PCA":
            self.reducer_ = PCA(n_components=n_components, random_state=self.seed).fit(z)
        elif self.method == "ICA":
            self.reducer_ = FastICA(n_components=n_components, whiten="unit-variance", max_iter=1000, tol=0.002, random_state=self.seed).fit(z)
        elif self.method == "Random projection":
            self.reducer_ = GaussianRandomProjection(n_components=n_components, random_state=self.seed).fit(z)
        else:
            raise ValueError(self.method)
        self.output_dim_ = n_components
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        clr = counts_to_clr(x[self.base_columns_].to_numpy())[:, self.keep_indices_]
        z = self.scaler_.transform(clr)
        if self.method == "Differential":
            return z[:, self.indices_]
        return self.reducer_.transform(z)


class TinyCNN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(12, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(min(8, n_features)),
            nn.Flatten(),
            nn.Linear(20 * min(8, n_features), 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x[:, None, :]).squeeze(1)


class CNNClassifier:
    def __init__(self, seed: int = SEED, epochs: int = 35):
        self.seed, self.epochs = seed, epochs

    def fit(self, x: np.ndarray, y: np.ndarray):
        set_seed(self.seed)
        order = np.arange(len(y))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(order)
        n_val = max(12, int(round(0.15 * len(y))))
        val_idx, train_idx = order[:n_val], order[n_val:]
        self.model_ = TinyCNN(x.shape[1])
        n_pos = max(1, int(y[train_idx].sum()))
        n_neg = max(1, int(len(train_idx) - n_pos))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32))
        opt = torch.optim.AdamW(self.model_.parameters(), lr=0.003, weight_decay=1e-3)
        ds = TensorDataset(torch.tensor(x[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.float32))
        loader = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)
        xv = torch.tensor(x[val_idx], dtype=torch.float32)
        yv = y[val_idx]
        best_auc, best_state, stale = -np.inf, None, 0
        for _ in range(self.epochs):
            self.model_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                pv = torch.sigmoid(self.model_(xv)).numpy()
            auc = roc_auc_score(yv, pv) if len(np.unique(yv)) == 2 else -float(loss.item())
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 6:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.tensor(x, dtype=torch.float32))).numpy()
        return np.column_stack([1 - p, p])


def make_model(name: str, seed: int):
    if name == "SVM":
        return SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=seed)
    if name == "Random forest":
        return RandomForestClassifier(n_estimators=350, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 24), alpha=0.01, learning_rate_init=0.001, early_stopping=True, validation_fraction=0.15, max_iter=350, random_state=seed)
    if name == "CNN":
        return CNNClassifier(seed=seed)
    raise ValueError(name)


def fit_bundle(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, method: str, k: int, model_name: str, seed: int, min_cohorts: int = 2):
    transformer = FeatureTransformer(method, k, seed, min_cohorts).fit(x, y, groups)
    z = transformer.transform(x)
    model = make_model(model_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(z, y)
    return transformer, model


def predict_bundle(bundle, x: pd.DataFrame) -> np.ndarray:
    transformer, model = bundle
    return model.predict_proba(transformer.transform(x))[:, 1]


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds)
    idx = np.argmax((tpr - fpr)[valid])
    return float(thresholds[valid][idx])


def nested_loco(x: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = meta["label"].to_numpy(int)
    groups = meta["cohort"].to_numpy()
    methods = ["Differential", "PCA", "ICA", "Random projection"]
    dimensions = [20, 40, 60, 80]
    models = ["SVM", "Random forest", "MLP", "CNN"]
    predictions, selections, searches = [], [], []
    for outer_i, heldout in enumerate(np.unique(groups)):
        train = groups != heldout
        test = ~train
        inner_groups = groups[train]
        inner_cv = list(GroupKFold(n_splits=3).split(x.loc[train], y[train], inner_groups))
        for model_i, model_name in enumerate(models):
            candidate_rows = []
            for method in methods:
                for k in dimensions:
                    fold_scores = []
                    for inner_i, (tr, va) in enumerate(inner_cv):
                        seed = SEED + outer_i * 1000 + model_i * 100 + inner_i
                        try:
                            bundle = fit_bundle(x.loc[train].iloc[tr], y[train][tr], inner_groups[tr], method, k, model_name, seed)
                            p = predict_bundle(bundle, x.loc[train].iloc[va])
                            fold_scores.append(roc_auc_score(y[train][va], p))
                        except Exception as exc:
                            warnings.warn(f"{heldout} {model_name} {method} {k}: {exc}")
                            fold_scores.append(np.nan)
                    score = float(np.nanmean(fold_scores)) if np.isfinite(fold_scores).any() else -np.inf
                    row = {"heldout_cohort": heldout, "model": model_name, "method": method, "features": k, "inner_mean_auc": score}
                    candidate_rows.append(row)
                    searches.append(row)
            candidate_rows.sort(key=lambda r: (-r["inner_mean_auc"], r["features"], r["method"]))
            chosen = candidate_rows[0]
            inner_oof = np.full(train.sum(), np.nan)
            for inner_i, (tr, va) in enumerate(inner_cv):
                seed = SEED + 50000 + outer_i * 1000 + model_i * 100 + inner_i
                bundle = fit_bundle(x.loc[train].iloc[tr], y[train][tr], inner_groups[tr], chosen["method"], chosen["features"], model_name, seed)
                inner_oof[va] = predict_bundle(bundle, x.loc[train].iloc[va])
            threshold = best_threshold(y[train], inner_oof)
            final_bundle = fit_bundle(x.loc[train], y[train], inner_groups, chosen["method"], chosen["features"], model_name, SEED + 90000 + outer_i * 100 + model_i)
            p_test = predict_bundle(final_bundle, x.loc[test])
            selections.append({**chosen, "threshold": threshold, "effective_features": final_bundle[0].output_dim_, "training_genera": len(final_bundle[0].columns_)})
            for sample_id, yi, pi in zip(x.index[test], y[test], p_test):
                predictions.append({"sample_id": sample_id, "cohort": heldout, "label": int(yi), "model": model_name, "probability": float(pi), "threshold": threshold, "prediction": int(pi >= threshold)})
            print(f"LOCO {heldout:12s} {model_name:13s} {chosen['method']:17s} k={chosen['features']:2d} inner AUC={chosen['inner_mean_auc']:.3f}", flush=True)
    return pd.DataFrame(predictions), pd.DataFrame(selections), pd.DataFrame(searches)


def metric_row(frame: pd.DataFrame) -> dict:
    y, p, pred = frame["label"].to_numpy(), frame["probability"].to_numpy(), frame["prediction"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(frame),
        "cases": int(y.sum()),
        "controls": int((1 - y).sum()),
        "auc_roc": roc_auc_score(y, p),
        "auc_pr": average_precision_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "brier": brier_score_loss(y, p),
    }


def bootstrap_auc(frame: pd.DataFrame, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    cohorts = list(frame["cohort"].unique())
    values = []
    for _ in range(n_boot):
        parts = []
        for cohort in cohorts:
            cf = frame[frame["cohort"] == cohort]
            idx = rng.integers(0, len(cf), len(cf))
            parts.append(cf.iloc[idx])
        boot = pd.concat(parts)
        if boot["label"].nunique() == 2:
            values.append(roc_auc_score(boot["label"], boot["probability"]))
    return tuple(np.quantile(values, [0.025, 0.975]))


def bootstrap_macro_auc(cohort_aucs: list[float], n_boot: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + 1)
    values = np.asarray(cohort_aucs, dtype=float)
    boot = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def summarize_performance(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_rows, summary_rows = [], []
    for model, mf in pred.groupby("model"):
        for cohort, cf in mf.groupby("cohort"):
            cohort_rows.append({"model": model, "cohort": cohort, **metric_row(cf)})
        pooled = metric_row(mf)
        lo, hi = bootstrap_auc(mf)
        cohort_aucs = [r["auc_roc"] for r in cohort_rows if r["model"] == model]
        macro_lo, macro_hi = bootstrap_macro_auc(cohort_aucs)
        summary_rows.append({"model": model, **pooled, "auc_ci_low": lo, "auc_ci_high": hi, "macro_cohort_auc": float(np.mean(cohort_aucs)), "macro_auc_ci_low": macro_lo, "macro_auc_ci_high": macro_hi, "min_cohort_auc": float(np.min(cohort_aucs)), "max_cohort_auc": float(np.max(cohort_aucs))})
    return pd.DataFrame(summary_rows).sort_values("auc_roc", ascending=False), pd.DataFrame(cohort_rows)


def random_split_benchmarks(x: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    y = meta["label"].to_numpy(int)
    groups = meta["cohort"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_prob = np.zeros(len(y))
    prior_prob = np.zeros(len(y))
    for fold, (tr, va) in enumerate(cv.split(x, y)):
        bundle = fit_bundle(x.iloc[tr], y[tr], groups[tr], "Differential", 40, "Random forest", SEED + fold)
        rf_prob[va] = predict_bundle(bundle, x.iloc[va])
        global_prior = y[tr].mean()
        priors = {g: y[tr][groups[tr] == g].mean() for g in np.unique(groups[tr])}
        prior_prob[va] = [priors.get(g, global_prior) for g in groups[va]]
    return pd.DataFrame([
        {"validation": "Random 5-fold", "predictor": "Microbiome random forest", "auc_roc": roc_auc_score(y, rf_prob)},
        {"validation": "Random 5-fold", "predictor": "Cohort prevalence only", "auc_roc": roc_auc_score(y, prior_prob)},
    ])


def pcoa(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = distance.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    g = -0.5 * j @ (distance ** 2) @ j
    vals, vecs = np.linalg.eigh(g)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    positive = vals > 0
    coords = vecs[:, positive] * np.sqrt(vals[positive])
    explained = vals[positive] / vals[positive].sum()
    return coords, explained


def design_matrix(values: np.ndarray, categorical: bool = True) -> np.ndarray:
    if categorical:
        dummies = pd.get_dummies(pd.Series(values), drop_first=True, dtype=float).to_numpy()
        return np.column_stack([np.ones(len(values)), dummies])
    return np.column_stack([np.ones(len(values)), values.astype(float)])


def sequential_permanova(gower: np.ndarray, base: np.ndarray, full: np.ndarray) -> tuple[float, float, int, int]:
    h0 = base @ pinv(base)
    h1 = full @ pinv(full)
    rank0, rank1 = np.linalg.matrix_rank(base), np.linalg.matrix_rank(full)
    df_effect = rank1 - rank0
    df_resid = len(base) - rank1
    ss_effect = float(np.trace((h1 - h0) @ gower))
    ss_resid = float(np.trace((np.eye(len(base)) - h1) @ gower))
    f = (ss_effect / df_effect) / (ss_resid / df_resid)
    r2 = ss_effect / float(np.trace(gower))
    return f, r2, df_effect, df_resid


def permanova_tests(distance: np.ndarray, meta: pd.DataFrame, permutations: int = 999) -> pd.DataFrame:
    n = len(meta)
    j = np.eye(n) - np.ones((n, n)) / n
    gower = -0.5 * j @ (distance ** 2) @ j
    cohort = meta["cohort"].to_numpy()
    disease = meta["label"].to_numpy(float)
    x_disease = design_matrix(disease, categorical=False)
    x_cohort = design_matrix(cohort, categorical=True)
    x_full = np.column_stack([x_cohort, disease])
    f_dis, r2_dis, dfe, dfr = sequential_permanova(gower, x_cohort, x_full)
    rng = np.random.default_rng(SEED)
    null_dis = []
    for _ in range(permutations):
        yp = disease.copy()
        for group in np.unique(cohort):
            idx = np.where(cohort == group)[0]
            yp[idx] = rng.permutation(yp[idx])
        null_dis.append(sequential_permanova(gower, x_cohort, np.column_stack([x_cohort, yp]))[0])
    f_coh, r2_coh, dfce, dfcr = sequential_permanova(gower, x_disease, np.column_stack([x_disease, design_matrix(cohort, True)[:, 1:]]))
    null_coh = []
    for _ in range(permutations):
        cp = rng.permutation(cohort)
        null_coh.append(sequential_permanova(gower, x_disease, np.column_stack([x_disease, design_matrix(cp, True)[:, 1:]]))[0])
    return pd.DataFrame([
        {"term": "Diagnosis adjusted for cohort", "pseudo_f": f_dis, "r2": r2_dis, "df_effect": dfe, "df_residual": dfr, "permutations": permutations, "p_value": (1 + np.sum(np.asarray(null_dis) >= f_dis)) / (permutations + 1)},
        {"term": "Cohort adjusted for diagnosis", "pseudo_f": f_coh, "r2": r2_coh, "df_effect": dfce, "df_residual": dfcr, "permutations": permutations, "p_value": (1 + np.sum(np.asarray(null_coh) >= f_coh)) / (permutations + 1)},
    ])


def differential_meta_analysis(x: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    clr = pd.DataFrame(counts_to_clr(x.to_numpy()), index=x.index, columns=x.columns)
    rows = []
    for taxon in x.columns:
        effects, variances = [], []
        signs = []
        for cohort, idx in meta.groupby("cohort").groups.items():
            present = x.loc[idx, taxon] > 0
            yc = meta.loc[idx, "label"]
            if present.sum() < max(4, int(0.10 * len(idx))) or ((present & (yc == 1)).sum() < 2) or ((present & (yc == 0)).sum() < 2):
                continue
            a = clr.loc[idx, taxon][yc == 1].to_numpy()
            b = clr.loc[idx, taxon][yc == 0].to_numpy()
            n1, n0 = len(a), len(b)
            pooled_var = ((n1 - 1) * a.var(ddof=1) + (n0 - 1) * b.var(ddof=1)) / (n1 + n0 - 2)
            if pooled_var <= 0:
                continue
            d = (a.mean() - b.mean()) / math.sqrt(pooled_var)
            correction = 1 - 3 / (4 * (n1 + n0) - 9)
            g = correction * d
            var_g = (n1 + n0) / (n1 * n0) + g * g / (2 * (n1 + n0 - 2))
            effects.append(g)
            variances.append(var_g)
            signs.append(np.sign(g))
        if len(effects) < 3:
            continue
        effects, variances = np.asarray(effects), np.asarray(variances)
        w = 1 / variances
        fixed = np.sum(w * effects) / np.sum(w)
        q = np.sum(w * (effects - fixed) ** 2)
        c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max(0.0, (q - (len(effects) - 1)) / c)
        wr = 1 / (variances + tau2)
        pooled = np.sum(wr * effects) / np.sum(wr)
        se = math.sqrt(1 / np.sum(wr))
        z = pooled / se
        p = 2 * norm.sf(abs(z))
        i2 = max(0.0, (q - (len(effects) - 1)) / q) if q > 0 else 0.0
        consistency = max(np.mean(np.asarray(signs) > 0), np.mean(np.asarray(signs) < 0))
        rows.append({"taxon": taxon, "cohorts": len(effects), "hedges_g": pooled, "ci_low": pooled - 1.96 * se, "ci_high": pooled + 1.96 * se, "p_value": p, "tau2": tau2, "i2": i2, "direction_consistency": consistency})
    result = pd.DataFrame(rows)
    if len(result):
        result["q_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
        result["stable"] = (result["q_value"] < 0.05) & (result["direction_consistency"] >= 0.75)
        result = result.sort_values(["stable", "q_value", "hedges_g"], ascending=[False, True, False])
    return result


def save_figure(fig: plt.Figure, stem: str, output_dir: Path = FIG_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_graphical_overview() -> None:
    fig = plt.figure(figsize=(12.4, 4.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    navy, teal, coral = "#203D4B", "#2A9D8F", "#E76F51"

    illustration = plt.imread(PAPER_FIG_DIR / "fig0_niddk_digestive_system.png")
    image_ax = fig.add_axes([0.015, 0.115, 0.285, 0.76])
    image_ax.imshow(illustration)
    image_ax.axis("off")
    ax.plot([0.315, 0.315], [0.09, 0.91], color="#DCE5E9", lw=1.2)
    ax.text(0.025, 0.925, "A  ANATOMI USUS DAN HATI", color=navy, fontsize=14, fontweight="bold", va="center")
    ax.text(0.025, 0.065, "Enam BioProject publik  |  390 sampel", color=navy, fontsize=11.5, va="center")

    ax.text(0.345, 0.925, "B  DARI SEKUENS KE VALIDASI LINTAS KOHOR", color=navy, fontsize=14, fontweight="bold", va="center")
    stages = [
        ("NCBI SRA", "Raw 16S +\nmetadata"),
        ("QIIME 2 / DADA2", "QC, denoising,\nfilter kimera"),
        ("SILVA", "Anotasi\ntaksonomi"),
        ("Tabel genus", "Harmonisasi\nantarkohor"),
        ("LOCO tersarang", "5 kohor latih,\n1 kohor uji"),
    ]
    centers = [0.388, 0.525, 0.662, 0.799, 0.936]
    track_y = 0.735
    ax.plot([centers[0], centers[-1]], [track_y, track_y], color="#B9CDD1", lw=1.5, zorder=1)
    for index, (center, (heading, detail)) in enumerate(zip(centers, stages), start=1):
        final = index == len(stages)
        ax.scatter(center, track_y, s=430, marker="o", facecolor=navy if final else "white",
                   edgecolor=navy if final else teal, linewidth=1.55, zorder=2)
        ax.text(center, track_y, f"{index:02d}", ha="center", va="center",
                color="white" if final else teal, fontsize=9.2, fontweight="bold", zorder=3)
        ax.text(center, 0.822, heading, ha="center", va="center", color=navy,
                fontsize=11.5, fontweight="bold")
        ax.text(center, 0.635, detail, ha="center", va="center", color="#52636C",
                fontsize=10.3, linespacing=1.15)

    ax.text(0.345, 0.505, "C  VARIASI KOMUNITAS BRAY-CURTIS", color=navy, fontsize=14, fontweight="bold", va="center")
    ax.text(0.345, 0.445, "R² PERMANOVA TERSESUAIKAN", color="#5A6A73", fontsize=10.5, va="center")
    bar_start, bar_width = 0.505, 0.365
    for y, label, value, color in [(0.345, "Kohor", 0.434, teal), (0.225, "Diagnosis", 0.007, coral)]:
        ax.text(0.345, y, label, va="center", color=navy, fontsize=12)
        ax.add_patch(Rectangle((bar_start, y - 0.018), bar_width, 0.036, facecolor="#E8EEF0", edgecolor="none"))
        ax.add_patch(Rectangle((bar_start, y - 0.018), bar_width * value / 0.5, 0.036, facecolor=color, edgecolor="none"))
        ax.text(0.895, y, f"{100 * value:.1f}%".replace(".", ","), va="center", color=navy, fontsize=13, fontweight="bold")
    ax.text(0.345, 0.095, "Setiap kohor diuji saat seluruh sampelnya ditahan dari pelatihan model.", color="#536771", fontsize=11.5, va="center")
    save_figure(fig, "fig0_graphical_overview", PAPER_FIG_DIR)


def plot_figures(x: pd.DataFrame, meta: pd.DataFrame, audit: pd.DataFrame, pred: pd.DataFrame, perf: pd.DataFrame, cohort_perf: pd.DataFrame, benchmarks: pd.DataFrame, differential: pd.DataFrame, distance: np.ndarray) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    colors = {"Control": "#2A9D8F", "MASLD": "#E76F51"}
    cohort_order = audit["cohort"].tolist()
    if (PAPER_FIG_DIR / "fig0_niddk_digestive_system.png").exists():
        plot_graphical_overview()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), gridspec_kw={"width_ratios": [1.05, 1.95]})
    axes[0].barh(cohort_order, audit["controls"], color=colors["Control"], label="Kontrol")
    axes[0].barh(cohort_order, audit["cases"], left=audit["controls"], color=colors["MASLD"], label="MASLD")
    axes[0].invert_yaxis(); axes[0].set_xlabel("Jumlah sampel"); axes[0].set_title("A  Komposisi kohor", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    axes[1].axis("off")
    steps = [(0.005, "Raw reads\nNCBI SRA"), (0.205, "QIIME 2\n+ DADA2"), (0.405, "Taksonomi\nSILVA"), (0.605, "Harmonisasi\ngenus"), (0.805, "Uji LOCO\ntersarang")]
    for i, (x0, label) in enumerate(steps):
        axes[1].add_patch(plt.Rectangle((x0, 0.40), 0.15, 0.24, transform=axes[1].transAxes, facecolor="#F2F4F7", edgecolor="#264653", linewidth=1.2))
        axes[1].text(x0 + 0.075, 0.52, label, ha="center", va="center", transform=axes[1].transAxes, fontsize=7.5)
        if i < len(steps) - 1:
            axes[1].annotate("", xy=(x0 + 0.195, 0.52), xytext=(x0 + 0.155, 0.52), xycoords=axes[1].transAxes, arrowprops=dict(arrowstyle="->", color="#264653", lw=1.5))
    axes[1].text(0.005, 0.87, "B  Provenance dan alur analisis", transform=axes[1].transAxes, fontweight="bold")
    axes[1].text(0.005, 0.18, "Pemrosesan hulu menghasilkan tabel genus; seluruh preprocessing ML mengecualikan kohor uji.", transform=axes[1].transAxes, fontsize=8.5, color="#4B5563")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.28, top=0.88, wspace=0.12)
    save_figure(fig, "fig1_study_design")

    coords, explained = pcoa(distance)
    plot_df = meta.copy(); plot_df["PCo1"], plot_df["PCo2"] = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cohorts = sorted(plot_df["cohort"].unique())
    cohort_colors = dict(zip(cohorts, sns.color_palette("colorblind", n_colors=len(cohorts))))
    for cohort in cohorts:
        for diagnosis in ["MASLD", "Control"]:
            part = plot_df[(plot_df["cohort"] == cohort) & (plot_df["diagnosis"] == diagnosis)]
            ax.scatter(part["PCo1"], part["PCo2"], s=18, marker="o",
                       facecolors=cohort_colors[cohort] if diagnosis == "MASLD" else "white",
                       edgecolors=cohort_colors[cohort], linewidths=0.65, alpha=0.82, zorder=3 if diagnosis == "Control" else 2)
    ax.set_title("PCoA Bray-Curtis lintas kohor", loc="left", fontweight="bold")
    ax.set_xlabel(f"PCo1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PCo2 ({explained[1]*100:.1f}%)")
    cohort_handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cohort_colors[c],
                             markeredgecolor=cohort_colors[c], markersize=5, label=c.removeprefix("PRJNA")) for c in cohorts]
    cohort_legend = ax.legend(handles=cohort_handles, title="BioProject PRJNA", frameon=False, fontsize=7.3,
                              title_fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                              columnspacing=0.8, handletextpad=0.2)
    ax.add_artist(cohort_legend)
    diagnosis_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#264653", markersize=5, label="Kontrol"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#264653", markeredgecolor="#264653", markersize=5, label="MASLD"),
    ]
    ax.legend(handles=diagnosis_handles, frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.41))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.35, top=0.89)
    save_figure(fig, "fig2_ordination")

    order = perf.sort_values("auc_roc")["model"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), gridspec_kw={"width_ratios": [1, 1.4]})
    pmap = perf.set_index("model")
    for i, model in enumerate(order):
        row = pmap.loc[model]
        axes[0].errorbar(row["auc_roc"], i, xerr=[[row["auc_roc"] - row["auc_ci_low"]], [row["auc_ci_high"] - row["auc_roc"]]], fmt="o", color="#264653", capsize=3)
    axes[0].set_yticks(range(len(order)), order); axes[0].axvline(0.5, color="#9CA3AF", ls="--", lw=1); axes[0].set_xlim(0.35, 1.0); axes[0].set_xlabel("AUROC gabungan (IK bootstrap 95%)"); axes[0].set_title("A  Kinerja pada kohor eksternal", loc="left", fontweight="bold")
    heat = cohort_perf.pivot(index="model", columns="cohort", values="auc_roc").loc[order]
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="vlag", center=0.5, vmin=0, vmax=1, cbar_kws={"label": "AUROC"}, ax=axes[1])
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
    axes[1].set_xticklabels([c.removeprefix("PRJNA") for c in heat.columns], rotation=45, ha="right", rotation_mode="anchor")
    axes[1].set_title("B  AUROC setiap kohor uji", loc="left", fontweight="bold"); axes[1].set_xlabel("BioProject PRJNA"); axes[1].set_ylabel("")
    fig.tight_layout(); save_figure(fig, "fig3_model_performance")

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for model, mf in pred.groupby("model"):
        fpr, tpr, _ = roc_curve(mf["label"], mf["probability"])
        auc = roc_auc_score(mf["label"], mf["probability"])
        ax.plot(fpr, tpr, lw=1.8, label=f"{model} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#9CA3AF", lw=1)
    ax.set(xlabel="Laju positif palsu", ylabel="Laju positif benar", xlim=(0, 1), ylim=(0, 1), title="Kurva ROC gabungan luar kohor")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.24, top=0.91)
    save_figure(fig, "fig4_roc_curves")

    stable = differential[differential["stable"]].copy()
    if stable.empty:
        stable = differential.head(12).copy()
    elif len(stable) > 12:
        stable = pd.concat([stable.nsmallest(6, "hedges_g"), stable.nlargest(6, "hedges_g")]).drop_duplicates()
    stable = stable.sort_values("hedges_g")
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    yloc = np.arange(len(stable))
    tax_colors = np.where(stable["hedges_g"] > 0, colors["MASLD"], colors["Control"])
    ax.hlines(yloc, stable["ci_low"], stable["ci_high"], color=tax_colors, lw=1.5)
    ax.scatter(stable["hedges_g"], yloc, c=tax_colors, s=30, zorder=3)
    taxon_labels = stable["taxon"].replace({
        "unclassified_[Eubacterium]_coprostanoligenes_group": "unclassified_[Eubacterium]_\ncoprostanoligenes_group",
        "unclassified_Ruminococcaceae": "unclassified_\nRuminococcaceae",
        "Christensenellaceae_R-7_group": "Christensenellaceae_\nR-7_group",
    })
    ax.axvline(0, color="#6B7280", lw=1); ax.set_yticks(yloc, taxon_labels); ax.set_xlabel("Hedges g efek acak (MASLD dikurangi kontrol)"); ax.set_title("Tanda tangan genus lintas kohor", loc="left", fontweight="bold")
    fig.tight_layout(); save_figure(fig, "fig5_signatures")

    best_rf = perf.loc[perf["model"] == "Random forest", "auc_roc"].iloc[0]
    gap = pd.concat([benchmarks, pd.DataFrame([{"validation": "Leave-one-cohort-out", "predictor": "Microbiome random forest", "auc_roc": best_rf}])], ignore_index=True)
    gap["validation"] = gap["validation"].replace({"Random 5-fold": "Acak 5 lipatan", "Leave-one-cohort-out": "Leave-one-cohort-out"})
    gap["predictor"] = gap["predictor"].replace({"Microbiome random forest": "Random forest mikrobioma", "Cohort prevalence only": "Prevalensi kohor saja"})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    validation_colors = {"Acak 5 lipatan": "#C49A45", "Leave-one-cohort-out": "#264653"}
    positions = {"Random forest mikrobioma": 1.0, "Prevalensi kohor saja": 0.0}
    for _, row in gap.iterrows():
        y = positions[row["predictor"]] + (0.13 if row["validation"] == "Acak 5 lipatan" else -0.13)
        color = validation_colors[row["validation"]]
        ax.plot([0.5, row["auc_roc"]], [y, y], color=color, lw=2.3, solid_capstyle="round")
        ax.scatter(row["auc_roc"], y, color=color, s=62, zorder=3)
        ax.text(row["auc_roc"] + 0.008, y, f"{row['auc_roc']:.3f}".replace(".", ","), va="center", fontsize=9, color="#253744")
    ax.axvline(0.5, color="#9CA3AF", ls="--", lw=1)
    ax.set_yticks([0, 1], ["Prevalensi kohor saja", "Random forest mikrobioma"])
    ax.set(xlim=(0.47, 0.84), ylim=(-0.4, 1.4), xlabel="AUROC", ylabel="")
    ax.set_title("Pembagian acak melebihkan generalisasi", loc="left", fontweight="bold")
    handles = [Line2D([0], [0], color=color, marker="o", lw=2, label=label) for label, color in validation_colors.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.26, top=0.87)
    save_figure(fig, "fig6_validation_gap")


def write_manifest() -> None:
    text = """# Daftar visualisasi

- `fig1_study_design`: provenance SRA-QIIME2-DADA2-SILVA, komposisi kohor, dan alur yang tahan kebocoran data.
- `fig2_ordination`: PCoA Bray-Curtis menurut diagnosis dan kohor.
- `fig3_model_performance`: interval kepercayaan gabungan dan peta panas AUROC kohor uji.
- `fig4_roc_curves`: prediksi gabungan yang diperoleh ketika kohor setiap sampel ditahan.
- `fig5_signatures`: meta-analisis efek acak terhadap efek CLR tingkat genus.
- `fig6_validation_gap`: perbandingan pembagian acak, prediktor kohor, dan leave-one-cohort-out.

Gambar analitik tersedia sebagai PNG 320 dpi dan PDF. Jika aset ilustrasi lokal tersedia di `paper/visualization/`, skrip juga dapat memperbarui ikhtisar naskah di folder tersebut. Folder `paper/` tidak dipublikasikan dalam repositori eksperimen.
"""
    (FIG_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-models", action="store_true", help="Only run data audit, ordination, and differential analysis")
    parser.add_argument("--figures-only", action="store_true", help="Regenerate figures from existing results without rerunning analysis")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    x, meta, audit = load_data()
    if args.figures_only:
        rel = x.div(x.sum(axis=1), axis=0)
        global_keep = ((x > 0).sum(axis=0) >= max(5, int(0.05 * len(x))))
        distance = squareform(pdist(rel.loc[:, global_keep], metric="braycurtis"))
        plot_figures(
            x, meta, audit,
            pd.read_csv(RESULTS_DIR / "nested_loco_predictions.csv"),
            pd.read_csv(RESULTS_DIR / "performance_summary.csv"),
            pd.read_csv(RESULTS_DIR / "per_cohort_performance.csv"),
            pd.read_csv(RESULTS_DIR / "random_split_benchmarks.csv"),
            pd.read_csv(RESULTS_DIR / "meta_analysis_taxa.csv"),
            distance,
        )
        write_manifest()
        return
    audit.to_csv(RESULTS_DIR / "cohort_summary.csv", index=False)
    meta.to_csv(RESULTS_DIR / "sample_manifest.csv")
    data_audit = {
        "samples": int(len(x)), "cases": int(meta["label"].sum()), "controls": int((1 - meta["label"]).sum()),
        "cohorts": int(meta["cohort"].nunique()), "union_genera": int(x.shape[1]),
        "duplicate_sample_ids": int(x.index.duplicated().sum()), "zero_library_samples": int((x.sum(axis=1) == 0).sum()),
        "class_prevalence": float(meta["label"].mean()), "cohort_records": audit.to_dict(orient="records"),
        "excluded_metadata_only_cohort": "PRJNA914906 has metadata but no abundance table used in the final analysis",
    }
    (RESULTS_DIR / "data_audit.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")

    rel = x.div(x.sum(axis=1), axis=0)
    global_keep = ((x > 0).sum(axis=0) >= max(5, int(0.05 * len(x))))
    distance = squareform(pdist(rel.loc[:, global_keep], metric="braycurtis"))
    permanova = permanova_tests(distance, meta)
    permanova.to_csv(RESULTS_DIR / "permanova.csv", index=False)
    differential = differential_meta_analysis(x, meta)
    differential.to_csv(RESULTS_DIR / "meta_analysis_taxa.csv", index=False)
    if args.skip_models:
        print(audit.to_string(index=False))
        print(permanova.to_string(index=False))
        return

    predictions, selections, searches = nested_loco(x, meta)
    performance, cohort_performance = summarize_performance(predictions)
    benchmarks = random_split_benchmarks(x, meta)
    predictions.to_csv(RESULTS_DIR / "nested_loco_predictions.csv", index=False)
    selections.to_csv(RESULTS_DIR / "selected_configurations.csv", index=False)
    searches.to_csv(RESULTS_DIR / "inner_search_results.csv", index=False)
    performance.to_csv(RESULTS_DIR / "performance_summary.csv", index=False)
    cohort_performance.to_csv(RESULTS_DIR / "per_cohort_performance.csv", index=False)
    benchmarks.to_csv(RESULTS_DIR / "random_split_benchmarks.csv", index=False)
    sensitivity = predictions[~predictions["cohort"].eq("PRJNA962007")].groupby("model").apply(lambda f: pd.Series(metric_row(f)), include_groups=False).reset_index()
    sensitivity.to_csv(RESULTS_DIR / "sensitivity_excluding_hiv_cohort.csv", index=False)
    plot_figures(x, meta, audit, predictions, performance, cohort_performance, benchmarks, differential, distance)
    write_manifest()
    summary = {
        "best_model_by_pooled_auc": performance.iloc[0].to_dict(),
        "permanova": permanova.to_dict(orient="records"),
        "stable_taxa_fdr_0_05": int(differential.get("stable", pd.Series(dtype=bool)).sum()),
        "sensitivity_excluding_hiv": sensitivity.to_dict(orient="records"),
    }
    (RESULTS_DIR / "statistical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nPerformance\n", performance.to_string(index=False))
    print("\nPERMANOVA\n", permanova.to_string(index=False))
    print(f"\nStable taxa: {summary['stable_taxa_fdr_0_05']}")


if __name__ == "__main__":
    main()
