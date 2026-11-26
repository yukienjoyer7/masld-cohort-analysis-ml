# Indonesian cohort integration checklist

The Indonesian cohort is in progress and is **not part of the current six-cohort results**. This checklist identifies work that can be prepared before its data arrive. It does not prescribe undocumented wet-lab or QIIME 2 parameters.

## Before data transfer

1. Confirm data-sharing permission, ethics/consent requirements, de-identification, and whether raw reads, metadata, or only derived abundance data may be released publicly. Keep any restricted data out of this public repository.
2. Define a sample-level data dictionary: study-specific ID, sequencing run ID when available, diagnostic label and its ascertainment criteria, collection site and date, age, sex, BMI, diabetes status, medication exposures, sequencing platform, library/amplicon information, and relevant technical batches. Record missingness explicitly.
3. Freeze a cohort inclusion/exclusion log and a mapping from source labels to the analysis labels. Do not assume historical NAFLD/MAFLD labels meet the 2023 MASLD definition without supporting clinical variables.
4. Collect the upstream processing record, including QIIME 2/DADA2/SILVA versions, commands, parameters, reference assets, and per-sample read attrition, if available. Preserve provenance without filling gaps by inference.

## When data arrive

1. Validate file integrity and unique sample IDs; reconcile raw-read, metadata, and abundance-table IDs.
2. Apply the documented upstream workflow and harmonize taxon names at genus level. Record sample exclusions and zero-assigned-abundance handling.
3. Produce an Indonesian-only descriptive profile and compare preprocessing/measurement characteristics with existing cohorts before any cross-cohort interpretation.
4. Run the existing leakage-controlled analysis with the Indonesian cohort added as a **held-out external cohort**. Report its AUROC, AUPRC, calibration, uncertainty, and class prevalence alongside per-cohort metrics, not only a pooled score.
5. Compare Indonesian-only training/evaluation and combinations with existing Asian cohorts only if sample counts and class balance support those designs. Specify the splits before inspecting test outcomes.
6. Recompute all cohort-dependent statistics, tables, and figures after integration. Do not carry forward the current six-cohort numerical results into a seven-cohort claim.

## Publication boundary

The manuscript may discuss the Indonesian cohort only after the data, consent status, quality checks, and analyses support it. Until then, the public experiment and results describe six non-Indonesian BioProjects.
