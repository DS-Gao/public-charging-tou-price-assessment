# Do time-of-use tariffs reach public charging prices? Component-resolved evidence from Beijing and Shenzhen

Dongsheng Gao · Junjie Hu · Jaeyoung Jay Lee

Version **v1.0**, dated **2026-11-26**. Research materials: https://github.com/DS-Gao/public-charging-tou-price-assessment/releases/tag/v1.0

The study compares electricity, service and total-price changes at adjoining time-of-use periods. Beijing transaction records and Shenzhen published prices provide separate applications. The paper evaluates direction, monetary discrepancies, historical component information and the comparisons represented under each support rule.

## Data sources and licenses

Obtain Beijing transactions and infrastructure from Figshare version 2, https://doi.org/10.6084/m9.figshare.31952289.v2 (CC BY 4.0), and UrbanEV station prices from Dryad, https://doi.org/10.5061/dryad.np5hqc04z, version released 4 February 2026 (CC0). `DATA_SOURCES.md` identifies the source files and access instructions. Complete original data archives and transaction records are not redistributed. Bundled files are derived observations, selected illustrative records, saved resamples and simulation outputs. Code and documentation use MIT, and data retain their source attribution and licenses.

## Installation and verification

Use Python 3.12 in a new virtual environment. From the package root:

```sh
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/verify_final_claims.py
python -m unittest discover -s tests -v
```

All three verification commands must exit 0. The first checks file hashes and Python syntax. The second checks 30 declared scientific and consistency conditions, including all indexed source files being present. The four unit tests cover numerical contracts for weights, missingness and reconstruction. These checks do not rerun all raw-data reconstruction, bootstrap or simulation experiments.

## Reproduction routes

Use new output directories outside the package:

```sh
python run.py core --output ../core_results --bootstrap saved
python run.py distribution --output ../distribution_results
python analysis/phase_a/scripts/run_phase_a.py --output ../paired_results
python scripts/verify_paired_outputs.py --repeat ../paired_results
python scripts/render_paper_figures.py --output ../paper_figures
```

The core route recalculates local comparisons from derived observations and reaggregates saved resamples. The distribution route reconstructs supported population summaries, masks and input comparisons. The paired route reproduces the conditional station-cluster inference using the supplied historical and evaluation inputs. The display entry point redraws Figures 3–6 from fixed data and exports the other finalized vector assets. Figure 1 is copied from the author-provided artwork. No statistical estimation is performed by this display command. Original statistical script filenames are preserved. Raw reconstruction requires separately obtained source records and the paths described in `DATA_SOURCES.md`. These routes are distinct from the three quick verification commands.

## Directory guide

| Path | Contents |
|---|---|
| `figures/` | Eleven main and nine supplementary figures |
| `data/derived/`, `data/reference/`, `data/additions/` | Derived observations, scientific diagnostics, saved estimates and resamples |
| `analysis/phase_a/` | Conditional paired-inference inputs, implementation and results |
| `src/`, `scripts/`, `config/`, `tests/` | Statistical implementation, reproduction commands, settings and tests |
| `evidence/` | Reconstruction and distributional simulation results used by the paper |
| `CLAIMS_TO_FILES.csv` | Sources and selectors for numerical claims, table rows, figures and prose |
| `MAIN_OBJECT_SOURCES.csv` | Tables 1–10, A.1 and B.1, and Figures 1–11 |
| `SUPPLEMENTARY_OBJECT_SOURCES.csv` | Tables S1–S26 and Figures S1–S9 |
| `analysis/phase_a/task1`, `task2`, `task5`, `references/` | Paired station-cluster intervals, support groups, price-operator comparisons and inputs for Section 4 and Sections S3–S4 |
| `data/additions/phase_a/` | Saved paired-inference and activity-weighting summaries for Section 4 |
| `data/additions/batch1/task1`, `task2`, `task3` | Station coverage, January recording and charger-field diagnostics for Sections 2.1, 3.3 and S1–S4 |
| `data/additions/batch1/task5` | Shenzhen noon comparisons and complete-day fixed-total classification for Section 3.2 and Section S6 |
| `data/additions/batch2/taskA` | Coverage, January price checks and population-weight decomposition for Sections 4–5 and S1, S4–S5 |
| `data/additions/batch2/taskB`, `taskC`, `taskD` | Station profiles, other-station calibrations and Shenzhen daily menus for Sections 3–4 and S2, S4, S6 |
| `data/additions/targeted/` | Price-construction and device-field diagnostics for Sections 2–3 and S3–S4 |
| `data/reference/historical/` | Saved boundary distributions, pairing references, support diagnostics and simulation summaries for Sections 3–5 and S1–S7 |
| `data/reference/prices/`, `crosswindow/`, `application/`, `external/`, `masking/` | Date/charger constructions, historical-input errors, monitoring selection, illustrative Shenzhen records and masking results |
| `evidence/analysis_evidence/r423_audits/` | Aggregation-target and grid-support sensitivities for Sections S4.5 and S6.1 |
| `evidence/public_reference/` | Supporting saved results used by the indexed comparisons |
| `config/`, `examples/`, `tests/`, `.github/` | Analysis settings, minimal usage examples, numerical-contract tests and automated verification |


Readers should start with `run.py` and `CLAIMS_TO_FILES.csv`, then follow a section, table or figure locator to its package-relative sources and selector. Directory names preserve the names assigned during processing because the frozen implementation reads those paths. Their scientific roles are given above. Continued table parts share one printed number. Selected trace values describe their stated periods, while monthly contrasts use the paired-date construction. `CITATION.cff` supplies the author, title and version metadata. Cite the original datasets separately.

The downloadable `Public_Charging_Final_Evidence_v1.0.zip` is a self-contained copy of these materials. `Public_Charging_Final_Evidence_v1.0_SHA256.txt` verifies the archive. No earlier research package is required.

Input reconciliation tables required by the frozen implementation remain available. Implementation filenames and embedded source identifiers are preserved.

## Article

These materials support Dongsheng Gao, Junjie Hu and Jaeyoung Jay Lee, “Do time-of-use tariffs reach public charging prices? Component-resolved evidence from Beijing and Shenzhen.” Manuscript under review.
