# Reproducible downstream analysis

## Data provenance

The abundance tables expected in the local `OTU-table/` directory are intermediate outputs produced by this study, not premade external tables. This input directory is Git-ignored and is not included in the public repository. Public raw 16S rRNA reads and metadata were retrieved from the NCBI Sequence Read Archive for BioProjects PRJNA1089365, PRJNA1152299, PRJNA1370285, PRJNA720274, PRJNA851946, and PRJNA962007. The reads were processed with QIIME 2, using DADA2 for quality control, denoising, and chimera filtering. Taxonomy was assigned with SILVA and aggregated at genus level before cross-cohort harmonization.

The upstream QIIME 2/DADA2 scripts, artifacts, commands, software/database versions, and complete parameter logs have not yet been organized in this repository snapshot. Exact trimming, denoising, classifier, amplicon-region, and read-attrition details are therefore intentionally not stated here.

## Running the downstream pipeline

After restoring the expected local `OTU-table/` files, run from the project root:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python src/run_analysis.py
```

The script loads the study-derived abundance tables, harmonizes assigned genera, applies multiplicative zero replacement and centered log-ratio transformation, and runs nested leave-one-cohort-out evaluation. Every feature filter, reducer, model choice, and decision threshold is fitted without the held-out cohort. It also produces Bray-Curtis PCoA, cohort-adjusted PERMANOVA, random-effects differential meta-analysis, sensitivity analysis excluding the HIV-associated cohort, and analytical figures in `results/figures/`.

The manuscript and its dedicated graphical-overview illustration are kept locally under the Git-ignored `paper/` directory. On a clean clone, the script skips that optional manuscript-only figure while still generating all analytical figures.

The random 5-fold benchmark is included only to quantify validation optimism. It is not the primary performance estimate.
