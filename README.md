# ASTROVET 🔭

**Explainable, physics-informed TESS transit-candidate vetting and triage**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![NASA ExoMiner baseline](https://img.shields.io/badge/baseline-NASA%20ExoMiner%2B%2B-green.svg)](https://github.com/nasa/ExoMiner)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ASTROVET is a candidate-prioritization and research tool for NASA TESS light curves. It processes Threshold-Crossing Events (TCEs) by combining transit period searches, physics-based diagnostics, self-supervised representations, and explainable evidence reports to help researchers prioritize candidates for human review.

---

## Key Features

- **Interactive Web Dashboard**: Upload `.fits` / `.fits.gz` light curves from TESS SPOC or QLP and get an instant vetting report.
- **Dual Periodicity Search**: Automated Box Least Squares (BLS) and Transit Least Squares (TLS) to detect and cross-check periodic signals.
- **Physics-Informed Diagnostics**: Checks for odd/even transit depth consistency, secondary eclipses, transit duration, and stellar density.
- **TESS Pixel Cutouts**: Automatically fetches 11×11 pixel cutouts from MAST to check for target contamination or background eclipsing binaries.
- **Plain-Language Explanations**: Optional AI summary via Groq to rephrase diagnostic findings without altering the underlying physics score.

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Subhra1432/ASTROVET.git
cd ASTROVET
pip install -e .
```

*(On Windows, you can use `py -3.11 -m pip install -e .`)*

Verify your setup:
```bash
python -m astrovet.cli doctor
```

### 2. Start the Web Dashboard

Launch the local review interface:

```bash
python -m astrovet.cli ui --port 8080
```

Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your browser and upload any TESS SPOC or QLP `.fits` light curve.

---

## CLI Commands

ASTROVET includes a full command-line suite:

- **Run Periodicity Search (BLS & TLS)**:
  ```bash
  python -m astrovet.cli periodogram search --fits-file path/to/lightcurve.fits
  ```

- **Generate Full Vetting Evidence Report**:
  ```bash
  python -m astrovet.cli explain --candidate-id 390651552-1-S23 --fits-file path/to/lightcurve.fits --output-dir artifacts/reports
  ```

- **Fetch TESS Light Curves from MAST**:
  ```bash
  python -m astrovet.cli data fetch --tic 390651552 --sector 23 --output-dir data/raw/mast
  ```

---

## Optional: Groq AI Explanations

The dashboard can rephrase vetting metrics into plain-language summaries using Groq. Set your API key in your terminal before launching:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY = "your-groq-api-key"

# Linux / macOS
export GROQ_API_KEY="your-groq-api-key"
```

*(Note: Groq only receives metric summaries; raw FITS data and target IDs remain local).*

---

## Repository Structure

```text
src/astrovet/        Core algorithms (BLS/TLS, physics diagnostics, models, UI)
data/                Local datasets, uploads, and pixel cutout cache (ignored by Git)
artifacts/           Generated candidate vetting reports and diagnostic plots
docs/                Technical documentation and model cards
tests/               Automated test suite
```

---

## Documentation

- [Upload Workflow Guide](docs/upload_workflow.md)
- [Model Card & Target Metrics](docs/model_card.md)
- [Implementation Status & Performance](docs/implementation_status.md)
- [Disposition Label Policy](docs/label_policy.md)

---

## ⚖️ Scientific Disclaimer

ASTROVET is a candidate-prioritization and triage aid for human review. A high model score is **not** a planet confirmation or statistical validation. A low model score is **not** proof of absence. Independent follow-up and expert review remain necessary.
