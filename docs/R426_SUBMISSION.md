# R4.26 submission reproduction map

Canonical repository: https://github.com/DS-Gao/public-charging-tou-price-assessment
Fixed version: [vR4.26-submission](https://github.com/DS-Gao/public-charging-tou-price-assessment/releases/tag/vR4.26-submission)

This is a packaging and evidence-publication update. Scientific code, numerical input files, saved draws and prior reference values in the Git repository are unchanged. The additional evidence archive preserves its scientific files byte for byte; its internal manifest and source correspondence provide exact hashes.

## Downloads and verification

1. Download the repository source at the fixed release tag. This includes `src/`, `scripts/`, `config/`, `data/derived/`, `data/reference/` and the locked requirements.
2. Download `R426_Analysis_Evidence.zip` and `R426_Analysis_Evidence_SHA256.txt` from that same release. Compare the ZIP's SHA-256, extract it separately, then run `python verify_evidence.py` in the extracted archive.
3. In the source checkout, `python scripts/verify_release.py` checks the tracked publication files and Python syntax. These commands are integrity checks, not statistical re-estimation.

## Result correspondence and scope

| Evidence | Maintained repository / companion archive | Reproduction scope |
|---|---|---|
| Local construction, components, support, ST/TX/KWH weights | `run.py core`, `run.py distribution`; existing `data/derived/` and `data/reference/` | Recalculation from derived inputs; some uncertainty results reaggregate saved draws. |
| Cross-window E/CAL/LOCAL | `run.py core`; existing cross-window input and draws | Public entry point; shared subsequent target and 5,846 common units. |
| E+S and support-conditioned four-input results | Archive `analysis_evidence/results/A1/`, `A2/`; same frozen row-level input | Full frozen summaries and historical source. E+S is not an option of the maintained core CLI. Historical wrappers retain old context dependencies; see archive README. |
| Formal reconstruction | Public `run.py simulation`; archive `analysis_evidence/results/A3/` | Executable specified synthetic experiment plus full frozen 15 x 2,000 replicate evidence, 40 metrics and known-truth records. No new simulation was run for publication. |
| Aggregation and station sensitivities | Archive `analysis_evidence/r423_audits/task2/`, `station/` | Frozen summaries, station-cluster leaveout results, aggregate reconstruction checks and historical calculation sources. |
| UrbanEV 70/80/90 percent completeness | Archive `analysis_evidence/r423_audits/task3/` | Frozen original/common-sample summaries and monthly panels; full raw five-minute ingestion and pre-threshold intermediate environment are not bundled. |
| Other historical reliability results | Existing `data/reference/historical/`; archive `public_reference/historical_summaries/` | Preserved summaries, not a claim of complete end-to-end reproduction of every historical experiment. |

The public workflows do not rerun the entire original covariance fitting, permutation generation or masking generation. Historical scripts in the archive document computation, but some expect the original workspace or old manuscript number-mapping files. They are not advertised as standalone executables. Source and result filenames have retained their provenance labels; this does not change the scientific version of the manuscript.

## Environment and commands

Use Python 3.12 and `python -m pip install -r requirements-lock.txt` in a virtual environment. The maintained commands and their dependencies are documented in the root README and [REPRODUCTION.md](REPRODUCTION.md). Always choose a new output directory outside the source checkout; do not overwrite the frozen reference data.

For optional raw reconstruction, obtain the exact version-2 Beijing January/July files and follow [DATA_SOURCES.md](../DATA_SOURCES.md). A dataset DOI resolving successfully does not mean the full raw dataset is in this repository.

## Sources and terms

- Beijing version 2: https://doi.org/10.6084/m9.figshare.31952289.v2 (CC BY 4.0); derived panels retain attribution and transformation notices.
- UrbanEV station-level five-minute release: https://doi.org/10.5061/dryad.np5hqc04z (CC0 1.0).
- Original code/documentation: MIT. Data-specific terms remain separate. See [DATA_SOURCES.md](../DATA_SOURCES.md).

SI1 and SI2 remain two separate journal Supplementary Material PDFs. The machine-readable evidence is supplied through this repository and release, not as an additional Editorial Manager attachment.
