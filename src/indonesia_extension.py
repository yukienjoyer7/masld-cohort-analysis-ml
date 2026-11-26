#!/usr/bin/env python3
"""Pre-specified extensions for an independent Indonesian 16S cohort.

This module does not contain Indonesian data. Source models are frozen without
access to Indonesian outcomes. Local Indonesian inputs and outputs remain outside
Git through the repository's ignore rules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from statsmodels.stats.multitest import multipletests

from .run_analysis import (
    COHORTS,
    DATA_DIR,
    RESULTS_DIR,
    ROOT,
    SEED,
    best_threshold,
    counts_to_clr,
    extract_genus,
    fit_bundle,
    load_data,
    metric_row,
    pcoa,
    predict_bundle,
    sequential_permanova,
)


METHODS = ("Differential", "PCA", "ICA", "Random projection")
DIMENSIONS = (20, 40, 60, 80)
MODELS = ("SVM", "Random forest", "MLP", "CNN")
OUTPUT_DIR = RESULTS_DIR / "indonesia"
BASELINE_ACCESSIONS = frozenset({
    "PRJNA1089365", "PRJNA1152299", "PRJNA1370285",
    "PRJNA720274", "PRJNA851946", "PRJNA962007",
})
ASIAN_ACCESSIONS = frozenset({
    "PRJNA1089365", "PRJNA1152299", "PRJNA1370285", "PRJNA851946",
})
CANDIDATE_FILE = ROOT / "config" / "pre_indonesia_meta_analysis_taxa.csv"


@dataclass(frozen=True)
class TargetData:
    counts: pd.DataFrame
    metadata: pd.DataFrame
    audit: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_target(
    otu_path: Path,
    metadata_path: Path,
    table_format: str,
    sample_column: str,
    label_column: str,
    case_value: str,
    control_value: str,
) -> TargetData:
    """Validate and load one count table with exact sample-level metadata matching."""
    if case_value.casefold() == control_value.casefold():
        raise ValueError("Case and control values must differ")
    lines = [line for line in otu_path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.startswith("# Constructed")]
    if not lines:
        raise ValueError("The abundance table is empty")
    header = next(csv.reader([lines[0]], delimiter="\t"))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate columns in the abundance table header")
    table = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", low_memory=False)
    feature_column = table.columns[0]
    taxonomy_column = "taxonomy" if "taxonomy" in table.columns else None
    sample_columns = [column for column in table.columns if column not in {feature_column, taxonomy_column}]
    if not sample_columns or any(not str(column).strip() for column in sample_columns):
        raise ValueError("The abundance table needs nonempty sample-ID columns")
    values = table[sample_columns].apply(pd.to_numeric, errors="raise")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Abundance values must be finite and nonnegative")
    if not np.allclose(numeric, np.rint(numeric)):
        raise ValueError("This pipeline expects count data, not fractional relative abundance")
    if table_format == "taxonomy":
        taxon = table[taxonomy_column].where(table[taxonomy_column].notna(), table[feature_column]) if taxonomy_column else table[feature_column]
        genera = taxon.map(extract_genus)
    elif table_format == "genus":
        genera = table[feature_column].astype("string").str.strip().str.removeprefix("g__")
        genera = genera.mask(genera.isna() | genera.eq("") | genera.str.lower().eq("unassigned"))
    else:
        raise ValueError("table_format must be taxonomy or genus")
    values.index = genera
    values = values.loc[values.index.notna()]
    if values.empty:
        raise ValueError("No assigned genera were found")
    counts = values.groupby(level=0, sort=True).sum().T.astype(float)
    counts.index.name = "sample_id"
    if (counts.sum(axis=1) <= 0).any():
        raise ValueError("At least one sample has zero assigned-genus abundance")

    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    for column in (sample_column, label_column):
        if column not in metadata.columns:
            raise ValueError(f"Metadata column not found: {column}")
    metadata[sample_column] = metadata[sample_column].str.strip()
    if metadata[sample_column].eq("").any() or metadata[sample_column].duplicated().any():
        raise ValueError("Metadata sample IDs must be nonempty and unique")
    table_ids, metadata_ids = set(counts.index), set(metadata[sample_column])
    if table_ids != metadata_ids:
        only_table = sorted(table_ids - metadata_ids)
        only_metadata = sorted(metadata_ids - table_ids)
        raise ValueError(f"Sample IDs differ: table-only={only_table[:5]}, metadata-only={only_metadata[:5]}")
    metadata = metadata.set_index(sample_column, drop=False).loc[counts.index].copy()
    labels = metadata[label_column].str.strip().str.casefold()
    mapping = {case_value.casefold(): 1, control_value.casefold(): 0}
    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise ValueError(f"Unexpected diagnostic labels: {unknown}")
    metadata["label"] = labels.map(mapping).astype(int)
    if metadata["label"].nunique() != 2:
        raise ValueError("Both case and control samples are required")
    metadata["cohort"] = "INDONESIA"
    audit = {
        "samples": int(len(counts)),
        "cases": int(metadata["label"].sum()),
        "controls": int((1 - metadata["label"]).sum()),
        "assigned_genera": int(counts.shape[1]),
        "otu_sha256": sha256_file(otu_path),
        "metadata_sha256": sha256_file(metadata_path),
        "table_format": table_format,
        "sample_column": sample_column,
        "label_column": label_column,
        "case_value": case_value,
        "control_value": control_value,
    }
    return TargetData(counts, metadata, audit)


def source_data(source_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source_set not in {"all", "asia"}:
        raise ValueError(f"Unknown source set: {source_set}")
    accessions = BASELINE_ACCESSIONS if source_set == "all" else ASIAN_ACCESSIONS
    selected = [spec for spec in COHORTS if spec.accession in accessions]
    if {spec.accession for spec in selected} != accessions:
        raise ValueError("One or more frozen source BioProjects are missing from COHORTS")
    counts, metadata, _ = load_data(selected)
    if metadata["cohort"].nunique() < 3:
        raise ValueError("At least three source cohorts are required for grouped tuning")
    return counts, metadata


def source_input_hashes(source_set: str) -> dict[str, str]:
    accessions = BASELINE_ACCESSIONS if source_set == "all" else ASIAN_ACCESSIONS
    selected = [spec for spec in COHORTS if spec.accession in accessions]
    paths = [DATA_DIR / name for spec in selected for name in (spec.otu, spec.metadata)]
    return {path.name: sha256_file(path) for path in paths}


def validation_folds(counts: pd.DataFrame, labels: np.ndarray, groups: np.ndarray, min_cohorts: int):
    if min_cohorts == 1:
        if np.bincount(labels, minlength=2).min() < 3:
            raise ValueError("Indonesia-only tuning needs at least three cases and three controls in its training split")
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(counts, labels))
    else:
        if len(np.unique(groups)) < 3:
            raise ValueError("Grouped tuning needs at least three independent cohorts")
        folds = list(GroupKFold(n_splits=3).split(counts, labels, groups))
    for train, validation in folds:
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[validation])) != 2:
            raise ValueError("Every internal training and validation fold must contain both classes")
    return folds


def select_and_fit(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    model_name: str,
    min_cohorts: int = 2,
    methods: tuple[str, ...] = METHODS,
    dimensions: tuple[int, ...] = DIMENSIONS,
):
    """Select representation and threshold using only the supplied training data."""
    labels = metadata.loc[counts.index, "label"].to_numpy(dtype=int)
    groups = metadata.loc[counts.index, "cohort"].to_numpy()
    folds = validation_folds(counts, labels, groups, min_cohorts)
    candidates = []
    failures = []
    for method in methods:
        for dimension in dimensions:
            scores = []
            for fold_index, (train, validation) in enumerate(folds):
                try:
                    bundle = fit_bundle(counts.iloc[train], labels[train], groups[train], method,
                                        dimension, model_name, SEED + fold_index, min_cohorts)
                    probabilities = predict_bundle(bundle, counts.iloc[validation])
                    scores.append(roc_auc_score(labels[validation], probabilities))
                except (ValueError, RuntimeError) as exc:
                    failures.append(f"{method}/{dimension}/fold{fold_index}: {exc}")
                    break
            if len(scores) == len(folds):
                candidates.append((float(np.mean(scores)), dimension, method))
    if not candidates:
        raise ValueError(f"No valid training-only configuration. First failure: {failures[0] if failures else 'none'}")
    score, dimension, method = sorted(candidates, key=lambda row: (-row[0], row[1], row[2]))[0]
    out_of_fold = np.full(len(counts), np.nan)
    for fold_index, (train, validation) in enumerate(folds):
        bundle = fit_bundle(counts.iloc[train], labels[train], groups[train], method,
                            dimension, model_name, SEED + 50000 + fold_index, min_cohorts)
        out_of_fold[validation] = predict_bundle(bundle, counts.iloc[validation])
    if not np.isfinite(out_of_fold).all():
        raise ValueError("Internal out-of-fold predictions are incomplete")
    threshold = best_threshold(labels, out_of_fold)
    bundle = fit_bundle(counts, labels, groups, method, dimension, model_name,
                        SEED + 90000, min_cohorts)
    info = {
        "model": model_name,
        "method": method,
        "requested_features": dimension,
        "effective_features": int(bundle[0].output_dim_),
        "inner_mean_auroc": score,
        "threshold": threshold,
        "training_samples": int(len(counts)),
        "training_cases": int(labels.sum()),
        "training_controls": int((1 - labels).sum()),
        "training_cohorts": sorted(set(groups)),
        "training_sample_ids": sorted(map(str, counts.index)),
        "feature_filter_min_cohorts": min_cohorts,
        "internal_cv": "stratified_3_fold" if min_cohorts == 1 else "grouped_3_fold",
        "seed": SEED,
    }
    return bundle, info


def new_outputs(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError(f"Refusing to overwrite existing analysis outputs: {existing}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def freeze_source(source_sets: list[str], model_name: str, output_dir: Path) -> None:
    if len(source_sets) != len(set(source_sets)):
        raise ValueError("Source sets must not be repeated")
    planned = []
    for source_set in source_sets:
        slug = model_name.lower().replace(" ", "_")
        planned.extend([output_dir / f"{source_set}_{slug}.pkl", output_dir / f"{source_set}_{slug}.lock.json"])
    new_outputs(*planned)
    fitted = []
    for source_set in source_sets:
        counts, metadata = source_data(source_set)
        bundle, info = select_and_fit(counts, metadata, model_name)
        fitted.append((source_set, bundle, info, source_input_hashes(source_set)))
    for source_set, bundle, info, input_hashes in fitted:
        slug = model_name.lower().replace(" ", "_")
        model_path = output_dir / f"{source_set}_{slug}.pkl"
        lock_path = output_dir / f"{source_set}_{slug}.lock.json"
        with model_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        lock = {
            "lock_version": 1,
            "source_set": source_set,
            "model_file": model_path.name,
            "model_sha256": sha256_file(model_path),
            "source_input_sha256": input_hashes,
            **info,
        }
        lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Frozen {source_set} {model_name}: {lock_path}")


def load_locked_model(lock_path: Path):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("lock_version") != 1:
        raise ValueError("Unsupported model-lock version")
    model_path = lock_path.parent / lock["model_file"]
    if sha256_file(model_path) != lock["model_sha256"]:
        raise ValueError(f"Model artifact changed after freezing: {model_path}")
    # Pickle is for trusted, locally generated model artifacts only.
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    return bundle, lock


def score_model(bundle, info: dict, target: TargetData, comparison: str, bootstraps: int):
    overlap = set(info.get("training_sample_ids", [])) & set(target.counts.index)
    if overlap:
        raise ValueError(f"Training and target sample IDs overlap in {comparison}: {sorted(overlap)[:5]}")
    base_columns = bundle[0].base_columns_
    aligned = target.counts.reindex(columns=base_columns, fill_value=0.0)
    if (aligned.sum(axis=1) <= 0).any():
        raise ValueError(f"At least one target sample has no abundance in the {comparison} training feature universe")
    probabilities = predict_bundle(bundle, aligned)
    threshold = float(info["threshold"])
    predictions = pd.DataFrame({
        "sample_id": target.counts.index,
        "label": target.metadata.loc[target.counts.index, "label"].to_numpy(dtype=int),
        "comparison": comparison,
        "probability": probabilities,
        "threshold": threshold,
        "prediction": (probabilities >= threshold).astype(int),
    })
    metrics = metric_row(predictions)
    metrics.update({
        "comparison": comparison,
        "case_prevalence": float(predictions["label"].mean()),
        "auprc_baseline": float(predictions["label"].mean()),
        "calibration_observed_minus_predicted": float(predictions["label"].mean() - probabilities.mean()),
        "shared_training_genera": int(len(set(target.counts.columns) & set(base_columns))),
        "training_base_genera": int(len(base_columns)),
    })
    rng = np.random.default_rng(SEED)
    labels = predictions["label"].to_numpy(dtype=int)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    auc_values, auprc_values = [], []
    for _ in range(bootstraps):
        sampled = np.concatenate([rng.choice(positive, len(positive), replace=True),
                                  rng.choice(negative, len(negative), replace=True)])
        auc_values.append(roc_auc_score(labels[sampled], probabilities[sampled]))
        auprc_values.append(average_precision_score(labels[sampled], probabilities[sampled]))
    metrics["auroc_ci_low"], metrics["auroc_ci_high"] = np.quantile(auc_values, [0.025, 0.975])
    metrics["auprc_ci_low"], metrics["auprc_ci_high"] = np.quantile(auprc_values, [0.025, 0.975])
    bins = pd.cut(probabilities, bins=np.linspace(0, 1, 6), include_lowest=True)
    calibration = predictions.groupby(bins, observed=True).agg(
        n=("label", "size"), observed_rate=("label", "mean"), mean_probability=("probability", "mean")
    ).reset_index(drop=True)
    calibration.insert(0, "comparison", comparison)
    return predictions, metrics, calibration


def evaluate_external(lock_paths: list[Path], target: TargetData, output_dir: Path, bootstraps: int) -> None:
    metric_path = output_dir / "external_metrics.csv"
    prediction_path = output_dir / "external_predictions.csv"
    calibration_path = output_dir / "external_calibration.csv"
    audit_path = output_dir / "external_input_audit.json"
    new_outputs(metric_path, prediction_path, calibration_path, audit_path)
    all_predictions, all_metrics, all_calibration = [], [], []
    comparisons = set()
    for lock_path in lock_paths:
        bundle, lock = load_locked_model(lock_path)
        comparison = f"{lock['source_set']}_{lock['model'].lower().replace(' ', '_')}"
        if comparison in comparisons:
            raise ValueError(f"Duplicate locked comparison: {comparison}")
        comparisons.add(comparison)
        predictions, metrics, calibration = score_model(bundle, lock, target, comparison, bootstraps)
        metrics["lock_sha256"] = sha256_file(lock_path)
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        all_calibration.append(calibration)
    pd.concat(all_predictions, ignore_index=True).to_csv(prediction_path, index=False)
    pd.DataFrame(all_metrics).to_csv(metric_path, index=False)
    pd.concat(all_calibration, ignore_index=True).to_csv(calibration_path, index=False)
    audit_path.write_text(json.dumps(target.audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compare_target(target: TargetData, split_column: str, train_value: str, test_value: str,
                   model_name: str, output_dir: Path, bootstraps: int) -> None:
    if split_column not in target.metadata.columns:
        raise ValueError(f"Split column not found: {split_column}")
    if train_value == test_value:
        raise ValueError("Training and test split values must differ")
    split = target.metadata[split_column].str.strip()
    if set(split) != {train_value, test_value}:
        raise ValueError("Split column must contain exactly the pre-specified train and test values")
    train_ids = split.index[split == train_value]
    test_ids = split.index[split == test_value]
    test = TargetData(target.counts.loc[test_ids], target.metadata.loc[test_ids], target.audit)
    indonesia_train = target.counts.loc[train_ids]
    indonesia_meta = target.metadata.loc[train_ids]
    for name, frame in (("Indonesian training", indonesia_meta), ("Indonesian test", test.metadata)):
        if frame["label"].nunique() != 2:
            raise ValueError(f"{name} split must contain cases and controls")
    asia_counts, asia_meta = source_data("asia")
    if set(target.counts.index) & set(asia_counts.index):
        raise ValueError("Indonesian and source sample IDs overlap")
    union = sorted(set(indonesia_train.columns) | set(asia_counts.columns))
    augmented_counts = pd.concat([
        asia_counts.reindex(columns=union, fill_value=0.0),
        indonesia_train.reindex(columns=union, fill_value=0.0),
    ])
    augmented_meta = pd.concat([asia_meta, indonesia_meta], join="inner")
    comparisons = [
        ("indonesia_only", indonesia_train, indonesia_meta, 1),
        ("indonesia_plus_asia", augmented_counts, augmented_meta, 2),
    ]
    metric_path = output_dir / "target_comparison_metrics.csv"
    prediction_path = output_dir / "target_comparison_predictions.csv"
    calibration_path = output_dir / "target_comparison_calibration.csv"
    config_path = output_dir / "target_comparison_config.json"
    new_outputs(metric_path, prediction_path, calibration_path, config_path)
    all_predictions, all_metrics, all_calibration, configurations = [], [], [], {}
    for name, counts, metadata, min_cohorts in comparisons:
        bundle, info = select_and_fit(counts, metadata, model_name, min_cohorts)
        predictions, metrics, calibration = score_model(bundle, info, test, name, bootstraps)
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        all_calibration.append(calibration)
        configurations[name] = info
    pd.concat(all_predictions, ignore_index=True).to_csv(prediction_path, index=False)
    pd.DataFrame(all_metrics).to_csv(metric_path, index=False)
    pd.concat(all_calibration, ignore_index=True).to_csv(calibration_path, index=False)
    config_path.write_text(json.dumps({"split_column": split_column, "train_value": train_value,
                                       "test_value": test_value, "test_sample_ids_sha256": hashlib.sha256(
                                           "\n".join(sorted(test_ids)).encode()).hexdigest(),
                                       "target_input_audit": target.audit, "configurations": configurations},
                                      indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def profile_target(target: TargetData, output_dir: Path, permutations: int) -> None:
    if len(target.counts) < 3:
        raise ValueError("At least three Indonesian samples are required for the diversity profile")
    alpha_path = output_dir / "profile_alpha.csv"
    genus_path = output_dir / "profile_top_genera.csv"
    pcoa_path = output_dir / "profile_pcoa.csv"
    beta_path = output_dir / "profile_beta_permanova.csv"
    audit_path = output_dir / "profile_input_audit.json"
    new_outputs(alpha_path, genus_path, pcoa_path, beta_path, audit_path)
    relative = target.counts.div(target.counts.sum(axis=1), axis=0)
    alpha = pd.DataFrame({
        "sample_id": target.counts.index,
        "label": target.metadata["label"].to_numpy(dtype=int),
        "observed_genera": (target.counts > 0).sum(axis=1).to_numpy(dtype=int),
        "shannon": -(relative.where(relative > 0, 1.0) * np.log(relative.where(relative > 0, 1.0))).sum(axis=1).to_numpy(),
        "gini_simpson": (1 - (relative ** 2).sum(axis=1)).to_numpy(),
    })
    overall = relative.mean(axis=0).sort_values(ascending=False).head(10)
    top_rows = []
    for rank, genus in enumerate(overall.index, start=1):
        for label in (0, 1):
            group = relative.loc[target.metadata["label"] == label, genus]
            top_rows.append({"rank_overall": rank, "genus": genus, "label": label,
                             "mean_relative_abundance": float(group.mean()),
                             "prevalence": float((group > 0).mean())})
    distance = squareform(pdist(relative.to_numpy(), metric="braycurtis"))
    coordinates, explained = pcoa(distance)
    pcoa_frame = pd.DataFrame({
        "sample_id": target.counts.index,
        "label": target.metadata["label"].to_numpy(dtype=int),
        "PCo1": coordinates[:, 0] if coordinates.shape[1] > 0 else np.zeros(len(relative)),
        "PCo2": coordinates[:, 1] if coordinates.shape[1] > 1 else np.zeros(len(relative)),
    })
    n = len(relative)
    center = np.eye(n) - np.ones((n, n)) / n
    gower = -0.5 * center @ (distance ** 2) @ center
    labels = target.metadata["label"].to_numpy(dtype=float)
    intercept = np.ones((n, 1))
    observed_f, observed_r2, df_effect, df_residual = sequential_permanova(
        gower, intercept, np.column_stack([intercept, labels]))
    rng = np.random.default_rng(SEED)
    null_f = [sequential_permanova(gower, intercept,
              np.column_stack([intercept, rng.permutation(labels)]))[0] for _ in range(permutations)]
    beta = pd.DataFrame([{
        "term": "Diagnosis within Indonesian cohort (unadjusted)",
        "pseudo_f": observed_f,
        "r2": observed_r2,
        "df_effect": df_effect,
        "df_residual": df_residual,
        "permutations": permutations,
        "p_value": (1 + sum(np.asarray(null_f) >= observed_f)) / (permutations + 1),
        "PCo1_variance": float(explained[0]) if len(explained) else np.nan,
        "PCo2_variance": float(explained[1]) if len(explained) > 1 else np.nan,
    }])
    alpha.to_csv(alpha_path, index=False)
    pd.DataFrame(top_rows).to_csv(genus_path, index=False)
    pcoa_frame.to_csv(pcoa_path, index=False)
    beta.to_csv(beta_path, index=False)
    audit_path.write_text(json.dumps(target.audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def geography_target(target: TargetData, output_dir: Path) -> None:
    """Describe geographic structure after adding Indonesia, without a causal test."""
    coordinate_path = output_dir / "geography_pcoa.csv"
    summary_path = output_dir / "geography_cohorts.csv"
    audit_path = output_dir / "geography_input_audit.json"
    new_outputs(coordinate_path, summary_path, audit_path)
    source_counts, source_meta = source_data("all")
    if set(source_counts.index) & set(target.counts.index):
        raise ValueError("Indonesian and source sample IDs overlap")
    features = sorted(set(source_counts.columns) | set(target.counts.columns))
    combined = pd.concat([
        source_counts.reindex(columns=features, fill_value=0.0),
        target.counts.reindex(columns=features, fill_value=0.0),
    ])
    target_meta = target.metadata.copy()
    target_meta["country"] = "Indonesia"
    metadata = pd.concat([source_meta[["cohort", "country", "label"]],
                          target_meta[["cohort", "country", "label"]]])
    relative = combined.div(combined.sum(axis=1), axis=0)
    distance = squareform(pdist(relative.to_numpy(), metric="braycurtis"))
    coordinates, explained = pcoa(distance)
    frame = metadata.loc[combined.index].reset_index(drop=True).copy()
    frame.insert(0, "sample_id", combined.index)
    frame["PCo1"] = coordinates[:, 0] if coordinates.shape[1] else 0.0
    frame["PCo2"] = coordinates[:, 1] if coordinates.shape[1] > 1 else 0.0
    frame["PCo1_variance"] = float(explained[0]) if len(explained) else np.nan
    frame["PCo2_variance"] = float(explained[1]) if len(explained) > 1 else np.nan
    summary = frame.groupby(["country", "cohort"], as_index=False).agg(
        n=("sample_id", "size"), cases=("label", "sum"))
    summary["controls"] = summary["n"] - summary["cases"]
    frame.to_csv(coordinate_path, index=False)
    summary.to_csv(summary_path, index=False)
    audit_path.write_text(json.dumps(target.audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def replicate_candidates(target: TargetData, output_dir: Path) -> None:
    output_path = output_dir / "candidate_genus_replication.csv"
    audit_path = output_dir / "replication_input_audit.json"
    new_outputs(output_path, audit_path)
    source_counts, _ = source_data("all")
    source_taxa = pd.read_csv(CANDIDATE_FILE)
    stable = source_taxa.loc[source_taxa["stable"].astype(str).str.lower() == "true"]
    if stable.empty:
        raise ValueError("No pre-specified stable taxa found in the six-cohort results")
    aligned = target.counts.reindex(columns=source_counts.columns, fill_value=0.0)
    if (aligned.sum(axis=1) <= 0).any():
        raise ValueError("At least one target sample has no abundance in the source genus universe")
    clr = pd.DataFrame(counts_to_clr(aligned.to_numpy()), index=aligned.index, columns=aligned.columns)
    labels = target.metadata["label"]
    rows = []
    for record in stable.itertuples(index=False):
        taxon = record.taxon
        row = {"taxon": taxon, "source_hedges_g": float(record.hedges_g),
               "target_hedges_g": np.nan, "ci_low": np.nan, "ci_high": np.nan,
               "p_value": 1.0, "direction_match": False, "status": "not_testable"}
        if taxon in target.counts.columns and taxon in clr.columns:
            present = target.counts[taxon] > 0
            sufficient = (present.sum() >= max(4, int(math.ceil(0.10 * len(target.counts))))
                          and (present & (labels == 1)).sum() >= 2
                          and (present & (labels == 0)).sum() >= 2)
            if sufficient:
                cases = clr.loc[labels == 1, taxon].to_numpy()
                controls = clr.loc[labels == 0, taxon].to_numpy()
                n_case, n_control = len(cases), len(controls)
                pooled_variance = ((n_case - 1) * cases.var(ddof=1) +
                                   (n_control - 1) * controls.var(ddof=1)) / (n_case + n_control - 2)
                if pooled_variance > 0:
                    d = (cases.mean() - controls.mean()) / math.sqrt(pooled_variance)
                    correction = 1 - 3 / (4 * (n_case + n_control) - 9)
                    effect = correction * d
                    variance = ((n_case + n_control) / (n_case * n_control) +
                                effect ** 2 / (2 * (n_case + n_control - 2)))
                    se = math.sqrt(variance)
                    row.update({"target_hedges_g": effect, "ci_low": effect - 1.96 * se,
                                "ci_high": effect + 1.96 * se,
                                "p_value": float(2 * norm.sf(abs(effect / se))),
                                "direction_match": bool(np.sign(effect) == np.sign(record.hedges_g)),
                                "status": "estimated"})
        rows.append(row)
    result = pd.DataFrame(rows)
    result["q_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
    result.to_csv(output_path, index=False)
    audit_path.write_text(json.dumps(target.audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def target_from_args(args) -> TargetData:
    return load_target(args.otu, args.metadata, args.table_format, args.sample_column,
                       args.label_column, args.case_value, args.control_value)


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--otu", type=Path, required=True, help="Local Indonesian count table in TSV format")
    parser.add_argument("--metadata", type=Path, required=True, help="Local Indonesian metadata in CSV format")
    parser.add_argument("--table-format", choices=("taxonomy", "genus"), required=True)
    parser.add_argument("--sample-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--case-value", required=True)
    parser.add_argument("--control-value", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validate local input and print a non-identifying audit")
    add_target_arguments(check)

    freeze = commands.add_parser("freeze", help="Train and lock source-only models before Indonesian evaluation")
    freeze.add_argument("--source-sets", nargs="+", choices=("all", "asia"), default=["all", "asia"])
    freeze.add_argument("--model", choices=MODELS, default="SVM")
    freeze.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    external = commands.add_parser("external", help="Evaluate pre-frozen models on untouched Indonesian data")
    add_target_arguments(external)
    external.add_argument("--locks", nargs="+", type=Path, required=True)
    external.add_argument("--bootstraps", type=int, default=2000)
    external.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    compare = commands.add_parser("compare", help="Compare Indonesia-only with Indonesia-plus-Asia training")
    add_target_arguments(compare)
    compare.add_argument("--split-column", required=True)
    compare.add_argument("--train-value", required=True)
    compare.add_argument("--test-value", required=True)
    compare.add_argument("--model", choices=MODELS, default="SVM")
    compare.add_argument("--bootstraps", type=int, default=2000)
    compare.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    profile = commands.add_parser("profile", help="Export Indonesian genus and alpha/beta diversity profiles")
    add_target_arguments(profile)
    profile.add_argument("--permutations", type=int, default=999)
    profile.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    geography = commands.add_parser("geography", help="Export combined source-and-Indonesia Bray-Curtis PCoA")
    add_target_arguments(geography)
    geography.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    replicate = commands.add_parser("replicate", help="Test the seven pre-selected six-cohort genera in Indonesia")
    add_target_arguments(replicate)
    replicate.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            freeze_source(args.source_sets, args.model, args.output_dir)
            return
        target = target_from_args(args)
        if args.command == "check":
            print(json.dumps({key: value for key, value in target.audit.items()
                              if not key.endswith("sha256")}, indent=2, ensure_ascii=False))
        elif args.command == "external":
            if args.bootstraps < 1:
                raise ValueError("bootstraps must be positive")
            evaluate_external(args.locks, target, args.output_dir, args.bootstraps)
        elif args.command == "compare":
            if args.bootstraps < 1:
                raise ValueError("bootstraps must be positive")
            compare_target(target, args.split_column, args.train_value, args.test_value,
                           args.model, args.output_dir, args.bootstraps)
        elif args.command == "profile":
            if args.permutations < 1:
                raise ValueError("permutations must be positive")
            profile_target(target, args.output_dir, args.permutations)
        elif args.command == "geography":
            geography_target(target, args.output_dir)
        elif args.command == "replicate":
            replicate_candidates(target, args.output_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
