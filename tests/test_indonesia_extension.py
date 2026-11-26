"""Synthetic tests for the future Indonesian-cohort analysis commands."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import indonesia_extension as extension


def synthetic_data(cohorts: int = 4, samples_per_cohort: int = 12, prefix: str = "SRC"):
    rows, metadata = [], []
    genera = ["Faecalibacterium", "Ruminococcus"] + [f"Genus_{index}" for index in range(8)]
    for cohort in range(cohorts):
        for sample in range(samples_per_cohort):
            label = sample % 2
            sample_id = f"{prefix}_{cohort}_{sample}"
            rows.append([3 + ((sample + genus * 2 + cohort) % 7) + (2 * label if genus == 0 else 0)
                         for genus in range(len(genera))])
            metadata.append({"sample_id": sample_id, "label": label,
                             "cohort": f"{prefix}_{cohort}", "country": "Thailand"})
    counts = pd.DataFrame(rows, index=[record["sample_id"] for record in metadata], columns=genera, dtype=float)
    meta = pd.DataFrame(metadata).set_index("sample_id")
    return counts, meta


class TargetInputTests(unittest.TestCase):
    def test_genus_table_and_metadata_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            otu = root / "otu.tsv"
            metadata = root / "metadata.csv"
            otu.write_text("genus\tI1\tI2\tI3\tI4\nAkkermansia\t2\t4\t3\t5\n"
                           "Faecalibacterium\t3\t1\t4\t2\n", encoding="utf-8")
            metadata.write_text("sample_id,diagnosis,split\nI1,Control,train\nI2,MASLD,train\n"
                                "I3,Control,test\nI4,MASLD,test\n", encoding="utf-8")
            target = extension.load_target(otu, metadata, "genus", "sample_id", "diagnosis", "MASLD", "Control")
            self.assertEqual(target.counts.shape, (4, 2))
            self.assertEqual(target.audit["cases"], 2)
            self.assertEqual(target.metadata.loc["I4", "label"], 1)

    def test_rejects_unmatched_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            otu = root / "otu.tsv"
            metadata = root / "metadata.csv"
            otu.write_text("genus\tI1\tI2\nAkkermansia\t2\t4\n", encoding="utf-8")
            metadata.write_text("sample_id,diagnosis\nI1,Control\nI3,MASLD\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Sample IDs differ"):
                extension.load_target(otu, metadata, "genus", "sample_id", "diagnosis", "MASLD", "Control")


class ModelDesignTests(unittest.TestCase):
    def test_source_sets_remain_frozen_to_original_bioprojects(self):
        counts, metadata = synthetic_data()
        with patch.object(extension, "load_data", return_value=(counts, metadata, None)) as loader:
            extension.source_data("asia")
            selected = {spec.accession for spec in loader.call_args.args[0]}
            self.assertEqual(selected, extension.ASIAN_ACCESSIONS)
            extension.source_data("all")
            selected = {spec.accession for spec in loader.call_args.args[0]}
            self.assertEqual(selected, extension.BASELINE_ACCESSIONS)

    def test_grouped_selection_and_locked_roundtrip(self):
        counts, metadata = synthetic_data()
        bundle, info = extension.select_and_fit(counts, metadata, "SVM", methods=("Differential",), dimensions=(5,))
        self.assertEqual(info["internal_cv"], "grouped_3_fold")
        target_counts, target_meta = synthetic_data(cohorts=1, prefix="IND")
        target = extension.TargetData(target_counts, target_meta, {"samples": 12})
        predictions, metrics, calibration = extension.score_model(bundle, info, target, "source_svm", 30)
        self.assertEqual(len(predictions), 12)
        self.assertEqual(metrics["comparison"], "source_svm")
        self.assertGreaterEqual(metrics["auc_roc"], 0)
        self.assertFalse(calibration.empty)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_path = root / "model.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(bundle, handle)
            lock_path = root / "model.lock.json"
            lock_path.write_text(json.dumps({"lock_version": 1, "model_file": model_path.name,
                                             "model_sha256": extension.sha256_file(model_path)}), encoding="utf-8")
            loaded, _ = extension.load_locked_model(lock_path)
            np.testing.assert_allclose(extension.predict_bundle(bundle, target_counts),
                                       extension.predict_bundle(loaded, target_counts))
        overlapping = extension.TargetData(counts.iloc[:12], metadata.iloc[:12], {"samples": 12})
        with self.assertRaisesRegex(ValueError, "sample IDs overlap"):
            extension.score_model(bundle, info, overlapping, "invalid_overlap", 5)

    def test_indonesia_only_requires_single_cohort_filter(self):
        counts, metadata = synthetic_data(cohorts=1, samples_per_cohort=18, prefix="IND")
        _, info = extension.select_and_fit(counts, metadata, "SVM", min_cohorts=1,
                                           methods=("Differential",), dimensions=(5,))
        self.assertEqual(info["internal_cv"], "stratified_3_fold")
        with self.assertRaisesRegex(ValueError, "Grouped tuning needs at least three independent cohorts"):
            extension.select_and_fit(counts, metadata, "SVM", min_cohorts=2,
                                     methods=("Differential",), dimensions=(5,))

    def test_target_comparison_uses_one_fixed_test_set(self):
        source_counts, source_meta = synthetic_data(prefix="SRC")
        target_counts, target_meta = synthetic_data(cohorts=1, samples_per_cohort=24, prefix="IND")
        target_meta["split"] = ["train" if int(index.rsplit("_", 1)[1]) < 18 else "test"
                                for index in target_meta.index]
        target = extension.TargetData(target_counts, target_meta, {"samples": 24})
        original_select = extension.select_and_fit

        def narrow_select(counts, metadata, model_name, min_cohorts=2):
            return original_select(counts, metadata, model_name, min_cohorts,
                                   methods=("Differential",), dimensions=(5,))

        with tempfile.TemporaryDirectory() as temp, patch.object(extension, "source_data", return_value=(source_counts, source_meta)), \
                patch.object(extension, "select_and_fit", side_effect=narrow_select):
            extension.compare_target(target, "split", "train", "test", "SVM", Path(temp), 20)
            predictions = pd.read_csv(Path(temp) / "target_comparison_predictions.csv")
            groups = predictions.groupby("comparison")["sample_id"].apply(set)
            self.assertEqual(groups["indonesia_only"], groups["indonesia_plus_asia"])
            self.assertEqual(len(groups["indonesia_only"]), 6)


class DescriptiveTests(unittest.TestCase):
    def test_geography_uses_source_and_target_without_changing_source(self):
        source_counts, source_meta = synthetic_data(prefix="SRC")
        source_meta["country"] = "Thailand"
        counts, metadata = synthetic_data(cohorts=1, samples_per_cohort=18, prefix="IND")
        target = extension.TargetData(counts, metadata, {"samples": 18})
        with tempfile.TemporaryDirectory() as temp, patch.object(
                extension, "source_data", return_value=(source_counts, source_meta)):
            extension.geography_target(target, Path(temp))
            coordinates = pd.read_csv(Path(temp) / "geography_pcoa.csv")
            cohorts = pd.read_csv(Path(temp) / "geography_cohorts.csv")
            self.assertEqual(len(coordinates), 66)
            self.assertEqual(int(cohorts.loc[cohorts["country"] == "Indonesia", "n"].sum()), 18)

    def test_profile_and_candidate_replication(self):
        counts, metadata = synthetic_data(cohorts=1, samples_per_cohort=18, prefix="IND")
        target = extension.TargetData(counts, metadata, {"samples": 18})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extension.profile_target(target, root, permutations=19)
            self.assertEqual(len(pd.read_csv(root / "profile_alpha.csv")), 18)
            self.assertEqual(len(pd.read_csv(root / "profile_top_genera.csv")), 20)
            self.assertEqual(len(pd.read_csv(root / "profile_beta_permanova.csv")), 1)

        source_counts, source_meta = synthetic_data(prefix="SRC")
        with tempfile.TemporaryDirectory() as temp, patch.object(extension, "source_data", return_value=(source_counts, source_meta)), \
                patch.object(extension, "CANDIDATE_FILE", Path(temp) / "frozen_taxa.csv"):
            root = Path(temp)
            pd.DataFrame([{"taxon": "Faecalibacterium", "hedges_g": -0.5, "stable": True},
                          {"taxon": "Not_present", "hedges_g": -0.3, "stable": True}]).to_csv(
                              root / "frozen_taxa.csv", index=False)
            extension.replicate_candidates(target, root)
            result = pd.read_csv(root / "candidate_genus_replication.csv")
            self.assertEqual(result.loc[result["taxon"] == "Faecalibacterium", "status"].iloc[0], "estimated")
            self.assertEqual(result.loc[result["taxon"] == "Not_present", "status"].iloc[0], "not_testable")


if __name__ == "__main__":
    unittest.main()
