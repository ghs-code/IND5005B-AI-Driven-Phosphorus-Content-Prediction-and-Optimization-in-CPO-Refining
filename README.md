# PhosphorAI Risk Workbench

PhosphorAI is a local Streamlit workbench for enterprise CPO phosphorus
feasibility review. It validates quality-table inputs, joins internal factor
tables when provided, screens factor evidence, produces a high-phosphorus manual
review ranking, and writes a CSV / JSON / HTML report package to a user-selected
folder.

The workbench does **not** replace laboratory phosphorus testing, does **not**
automate batch segregation, and does **not** optimize phosphoric acid or
bleaching earth dosing automatically.

## Install

Use Python 3.11 or newer.

```bash
python3 -m pip install -U pip
python3 -m pip install -e .
```

Optional local defaults can be configured from `.env.example`.

```bash
make init-config
```

## Launch

```bash
make ui
```

The Streamlit UI is the supported enterprise entry point. Select quality-table
Excel inputs, optional internal factor tables, mapping fields, and an output
directory in the browser.

Each run creates a folder such as:

```text
<selected-output-dir>/phosphorai_run_<timestamp>/
```

The report package includes:

- `run_manifest.json`
- `quality_validation_summary.json`
- `preprocessing_summary.json`
- `core_factor_screening.csv`
- `core_factor_screening.json`
- `internal_factor_evidence.csv`
- `internal_factor_evidence.json`
- `risk_scoring_batch_ranking.csv`
- `enterprise_summary.html`

Some files are generated only when the corresponding internal-factor or risk
scoring step is enabled and has enough data.

## Input Expectations

Quality-table Excel files must follow the current template: data rows start at
row 5 and business fields are read from columns B:U. Multiple files and years
are supported.

Internal factor tables may be CSV or Excel. A single file, a directory, or
multiple files can be supplied. Use reliable join keys such as `sample_id`,
`batch_id`, `lot_id`, or a practical composite key such as `date,feed_tank`.
Low match rates or duplicate join keys block strong factor conclusions.

## Evidence Levels

- `direct_supported`: supplied enterprise fields show stable association or
  model signal.
- `weak_or_context_dependent`: signal exists but is not stable or strong enough.
- `not_supported_current_data`: fields exist but current data does not show a
  useful signal.
- `not_assessable`: required fields or joins are missing.
- `observed_process_response`: process fields show observed downstream response
  evidence, not automatic dosing authorization.

## Local Data

Confidential raw data and generated outputs should stay local. Recommended
ignored locations:

```text
local_data/raw/          # confidential Excel inputs
local_data/internal/     # confidential internal factor tables
local_data/deliveries/   # generated enterprise report packages
local_archive/           # old local academic materials and historical runs
```

## Verify

```bash
make verify
```
