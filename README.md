# MASLD multicohort microbiome analysis

This public repository contains the **experiment**, not the manuscript or input abundance tables. It includes downstream analysis code, numerical outputs, and analytical figures. The local `paper/` and `OTU-table/` directories are intentionally excluded from Git.

## Data provenance and scope

Raw 16S rRNA reads and metadata originated from NCBI SRA BioProjects PRJNA1089365, PRJNA1152299, PRJNA1370285, PRJNA720274, PRJNA851946, and PRJNA962007. The study processed reads using QIIME 2 and DADA2, assigned taxonomy with SILVA, and produced abundance-table intermediates locally. The Python pipeline starts with genus-level harmonization of these intermediates. Input tables and metadata are **not distributed in this repository**. Complete upstream QIIME 2/DADA2/SILVA commands, artifacts, and parameter logs have not yet been added, so the upstream stage is not fully reproducible from this repository alone.

The current analytical dataset contains **390 samples** from six BioProjects in Thailand, China, and Spain. **No Indonesian cohort is included in the current results.** An Indonesian cohort is planned; its integration and validation checks are outlined in [`docs/indonesia_cohort_integration.md`](docs/indonesia_cohort_integration.md). Existing performance values must not be interpreted as results for that future cohort.

## Repository layout

- `OTU-table/`: required local input directory, excluded from Git.
- `src/run_analysis.py`: harmonization, statistical analyses, and nested leave-one-cohort-out machine learning.
- `results/`: numerical outputs, sample manifest, audit records, and analytical figures.
- `requirements.txt`: Python dependencies.

## Reproduce downstream analysis

Place the study-derived input tables and matching metadata under the local `OTU-table/` directory using the filenames specified in `src/run_analysis.py`. Without these inputs, the full pipeline and figure regeneration cannot be run from a clean clone. Then, from the repository root, with Python 3.12 and `uv` installed:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python src/run_analysis.py
```

To regenerate figures without rerunning model training (the local `OTU-table/` inputs are still required):

```bash
.venv/bin/python src/run_analysis.py --figures-only
```

Analytical figures are written to `results/figures/`. A separate manuscript overview is generated only when its local illustration asset exists under the ignored `paper/visualization/` directory. See [`src/README.md`](src/README.md) for methodology and reproducibility boundaries.

## Current analyses

The pipeline reports Bray-Curtis PCoA, cohort-adjusted PERMANOVA, random-effects taxon meta-analysis, nested leave-one-cohort-out (LOCO) evaluation of SVM, random forest, MLP, and 1D CNN classifiers, a random-split comparison, and sensitivity analysis excluding the HIV-associated cohort. All trainable preprocessing and model selection are confined to training cohorts within the nested LOCO design.

The current results are a retrospective analysis of public cohorts, not a clinically validated MASLD screening test.
