# ASTROVET

**Explainable, physics-informed TESS transit-candidate vetting and triage**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![NASA ExoMiner baseline](https://img.shields.io/badge/baseline-NASA%20ExoMiner%2B%2B-blue)](https://github.com/nasa/ExoMiner)

ASTROVET processes TESS light curves and candidate metadata to help researchers prioritize Threshold-Crossing Events (TCEs) for human review. It combines transit-period searches, physical diagnostics, self-supervised light-curve representations, optional reference-model scores, and an explainable fusion layer.

> [!IMPORTANT]
> ASTROVET is a research and candidate-triage tool. It does not confirm planets, provide statistical validation, or replace expert review. The trained SSL-plus-physics head was evaluated on a selected TOI follow-up cohort; its calibrated scores are not population-wide planet probabilities. Broader independent validation across sectors and pipelines remains necessary.

## What is in this repository

- Python package and command-line interface for TESS FITS ingestion, quality checks, BLS/TLS period searches, physics diagnostics, reports, and the local review dashboard.
- Self-supervised encoder pretraining on real unlabeled FITS files, with synthetic data available only through an explicit demo option.
- Leakage-aware training utilities for two distinct research tasks: the Astronet-Triage `E` versus `B/J` catalog baseline, and the separately trained SSL-plus-physics head for TOI dispositions.
- Tests, configuration examples, [model card](docs/model_card.md), [implementation status](docs/implementation_status.md), and [upload workflow](docs/upload_workflow.md).

The GitHub repository contains source code and documentation, not the downloaded TESS observations, private/local catalogs, trained checkpoints, or generated model bundles. Those items are ignored by Git; you must supply or download the data you need. The NASA ExoMiner source and prediction catalog are not copied into this repository. ASTROVET's adapter can query a catalog you provide and can use the supported container workflow when its dependencies are available.

## Architecture

```text
TESS FITS + candidate metadata
             │
    ┌────────┼─────────┐
    ▼        ▼         ▼
 quality   BLS/TLS   self-supervised
 checks    search   light-curve encoder
    └────────┼─────────┘
             ▼
 physics diagnostics + optional ExoMiner reference score
             ▼
 fusion / triage score + reasons + evidence report
```

Missing inputs and modalities remain explicitly unavailable; they are not silently replaced with fabricated baseline scores. BLS/TLS detects periodic transit-like behavior, which can also arise from eclipsing binaries or stellar variability.

## Install and verify

Clone the repository and install the package with its development dependencies:

```bash
git clone https://github.com/Subhra1432/ASTROVET.git
cd ASTROVET
python -m pip install -e ".[dev]"
```

On Windows, the Python launcher can be used instead:

```powershell
py -m pip install -e ".[dev]"
```

Check the installation and run the test suite:

```bash
python -m astrovet.cli doctor
python -m pytest -q
```

Use `py -m ...` in place of `python -m ...` in PowerShell if needed. Dependencies and supported Python versions are defined in [pyproject.toml](pyproject.toml).

## Use the dashboard with an uploaded FITS file

Start the local web interface:

```bash
python -m astrovet.cli ui --port 8080
```

Open <http://127.0.0.1:8080/> and upload a TESS SPOC or QLP `.fits` / `.fits.gz` light curve. The upload workflow checks the file, records a checksum and cadence audit, and runs the available vetting steps. Uploaded originals are stored under `data/uploads/`, which is ignored by Git. See [the upload guide](docs/upload_workflow.md) for accepted inputs, limits, and details.

After analysis, the dashboard can show a downloadable two-panel light-curve chart. It also looks up a matching 11 × 11 TESS pixel cutout from MAST using the candidate's TIC and sector, then displays a median view of the quality-clear pixels. Pixel cutouts are cached locally in `data/pixel_cutouts/` and are not committed to Git. The pixel view is supplemental context; the uploaded light curve is still used for transit analysis. If MAST has no cutout or is unreachable, candidate vetting still completes without the pixel image. See the [TESSCut guide](https://astroquery.readthedocs.io/en/stable/mast/mast_cut.html).

## Command-line workflows

### Fetch an observed TESS light curve from MAST

The command uses online MAST retrieval by default. Replace the example TIC and sector with the target you want:

```bash
python -m astrovet.cli data fetch \
  --tic 390651552 \
  --sector 23 \
  --output-dir data/raw/mast \
  --manifest data/manifests/mast_ingest.json
```

`data/raw/` and generated manifests are local outputs and are not committed. Offline synthetic fixtures require an explicit opt-in; do not use them as scientific observations.

### Audit and train the catalog baseline

Audit a supplied TCE catalog:

```bash
python -m astrovet.cli data audit \
  --input /path/to/tces_with_labels_v3.csv \
  --output artifacts/audit_report.json
```

Train the tabular `E` versus `B/J` classifier:

```bash
python -m astrovet.cli model train \
  --data /path/to/tces_with_labels_v3.csv \
  --artifacts-dir artifacts/models/tce_catalog_baseline
```

This task predicts the Astronet-Triage `E` label from catalog features. `E` means a periodic eclipsing signal and includes both planets and eclipsing binaries; it is **not** a planet/non-planet label. Its artifact is separate from the FITS-vetting fusion model. See the [model card](docs/model_card.md) and [label policy](docs/label_policy.md).

### Pretrain the self-supervised encoder

Pretrain from one real FITS file:

```bash
python -m astrovet.cli ssl pretrain \
  --input-fits data/raw/mast/your_lightcurve.fits \
  --epochs 5 \
  --output-checkpoint data/checkpoints/ssl_encoder.npz
```

Or use a directory of real FITS files:

```bash
python -m astrovet.cli ssl pretrain \
  --input-fits-dir data/raw/mast \
  --epochs 5 \
  --output-checkpoint data/checkpoints/ssl_encoder.npz
```

Pretraining learns a representation from unlabeled light curves; by itself it does not train a planet classifier. For scientifically controlled training, pass a host-grouped split manifest with `--train-host-split-manifest` so only training hosts contribute to pretraining. Synthetic demo views require the explicit `--synthetic-demo` option.

### Run period searches and candidate vetting

Run BLS and TLS on a real light curve, optionally checking a supplied ephemeris (duration is in hours):

```bash
python -m astrovet.cli periodogram search \
  --fits-file data/raw/mast/your_lightcurve.fits \
  --period 1.4658 \
  --epoch 1931.1684 \
  --duration 1.0
```

Run the full report workflow:

```bash
python -m astrovet.cli explain \
  --candidate-id 390651552-1-S23 \
  --fits-file data/raw/mast/your_lightcurve.fits \
  --period 1.4658 \
  --epoch 1931.1684 \
  --duration 1.0 \
  --output-dir artifacts/reports
```

To include a self-supervised representation, add `--encoder-mode ssl --ssl-checkpoint data/checkpoints/ssl_encoder.npz`. Reports are written to the selected output directory and include evidence, diagnostic reasons, and plots. A periodogram detection is not a planet confirmation.

### Train and enable the SSL-plus-physics research head

The scripts in [`scripts/`](scripts/) prepare a disposition cohort from MAST TCE statistics and NASA Exoplanet Archive TOI dispositions, train the catalog prototype, then train the SSL-plus-physics fusion head from matched real FITS files. These workflows require external downloaded data and checkpoints; they are not one-click downloads and their artifacts are not stored in GitHub. See the [implementation status](docs/implementation_status.md) and [model card](docs/model_card.md) for the cohort, results, and limitations.

To explicitly enable a locally trained compatible encoder and fusion bundle in the dashboard, set their paths before launch:

```powershell
$env:ASTROVET_SSL_CHECKPOINT = "C:\path\to\ssl_encoder.npz"
$env:ASTROVET_FUSION_MODEL = "C:\path\to\fusion_model.joblib"
py -m astrovet.cli ui --port 8080
```

The fusion head was evaluated on a selected TOI follow-up cohort and must not be interpreted as a population-calibrated probability. It remains optional and is not silently loaded as a default model.

## NASA ExoMiner reference baseline

ASTROVET records the NASA ExoMiner++ release metadata and supports lookup from an external published prediction catalog. The catalog is not bundled. Provide its local path explicitly:

```bash
python -m astrovet.cli baseline run \
  --catalog /path/to/exominer_predictions.csv \
  --candidate-id 390651552-1-S23
```

If the candidate is absent from the catalog, ASTROVET reports it as unavailable rather than inventing a score. This catalog adapter does not retrain or modify NASA's upstream model. See [NASA's ExoMiner repository](https://github.com/nasa/ExoMiner) for the upstream project.

## Repository layout

```text
configs/             Example runtime and model configuration
docs/                Model, data, workflow, and project status documents
scripts/             Cohort preparation and research training workflows
src/astrovet/        CLI and core ingestion, models, physics, reports, and UI
tests/               Automated tests
data/                Local datasets and generated manifests (not committed)
artifacts/            Local model bundles and reports (not committed)
```

## Data sources

- [MAST TESS mission data](https://archive.stsci.edu/missions-and-data/tess)
- [MAST TESS TCE bulk downloads](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html)
- [NASA Exoplanet Archive TOI table documentation](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
- [NASA ExoMiner source repository](https://github.com/nasa/ExoMiner)

## Project documentation

- [Implementation status, evaluated results, and remaining scientific limitations](docs/implementation_status.md)
- [Model card and target definitions](docs/model_card.md)
- [Disposition label policy](docs/label_policy.md)
- [Data dictionary](docs/data_dictionary.md)
- [FITS upload and vetting workflow](docs/upload_workflow.md)
- [Detailed project progress report (Markdown)](docs/ASTROVET_project_progress_report.md) · [Word document](docs/ASTROVET_project_progress_report.docx)
