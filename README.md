# PhosphorAI Risk Workbench

PhosphorAI is an enterprise validation workbench for crude palm oil (CPO)
phosphorus analysis. It helps refinery teams check whether available quality
data and internal operational factors are associated with phosphorus outcomes,
then use the result as a manual review and data-improvement aid.

The current prototype does **not** replace laboratory phosphorus testing, does
**not** automate batch isolation, and does **not** optimize phosphoric acid or
bleaching earth dosing automatically.

## What The Workbench Does
- Validates one or more local quality-table Excel files that follow the current
  template.
- Screens core quality factors such as FFA, M&I, IV, DOBI, and carotene/PV
  against the selected target.
- Joins enterprise internal factor data to quality records and reports direct
  evidence levels for origin, storage, transport, lab, process, and chemistry
  fields when those fields are supplied.
- Runs the feed phosphorus risk-scoring prototype and exports a ranked batch
  review list.
- Provides a Streamlit browser UI for modular operation.

## Data Privacy
Confidential raw data and generated outputs should stay local. The repository
tracks only reusable code, empty placeholders, and user documentation.

Recommended local-only locations:

```text
local_data/raw/       # confidential Excel inputs
local_runs/           # archived run outputs
local_reports/        # optional local reports
```

These locations are ignored by Git.

## Install
Use Python 3.11 or newer.

```bash
python3 -m pip install -U pip
python3 -m pip install -e .
```

To run the browser workbench:

```bash
python3 -m pip install -e ".[ui]"
```

## Modular Workflows
Show resolved paths:

```bash
make print-config
```

Validate quality-table inputs:

```bash
make validate-data CPO_RAW_INPUT=local_data/raw
```

Preprocess quality tables:

```bash
make preprocess CPO_RAW_INPUT=local_data/raw CPO_YEAR=all
```

Screen core quality factors:

```bash
make core-factors CPO_MODEL_SOURCE=local_runs/final_feed_2024_2025/processed/model_source.csv
```

Validate enterprise internal factors:

```bash
make internal-factors \
  CPO_MODEL_SOURCE=local_runs/final_feed_2024_2025/processed/model_source.csv \
  CPO_INTERNAL_FACTOR_INPUT=local_data/internal/factors.csv \
  CPO_INTERNAL_JOIN_KEYS=date,feed_tank \
  CPO_INTERNAL_FACTOR_FIELDS=storage_hours,source,transport_mode
```

Run the risk-scoring prototype:

```bash
make risk-score CPO_MODEL_SOURCE=local_runs/final_feed_2024_2025/processed/model_source.csv
```

Launch the browser UI:

```bash
make ui
```

Run tests and lightweight checks:

```bash
make verify
```

## Input Expectations
Quality-table Excel files must follow the current template: data rows start at
row 5 and business fields are read from columns B:U. Multiple files and years
are supported.

Internal factor tables may be CSV or Excel. A single file, a directory, or
multiple files can be supplied; files are combined before validation. They
should include reliable join keys such as `sample_id`, `batch_id`, `lot_id`, or
a practical composite key such as `date,feed_tank`. Low match rates or duplicate
join keys block strong factor conclusions.

## Evidence Levels
- `direct_supported`: supplied enterprise fields show stable association or
  model signal.
- `weak_or_context_dependent`: signal exists but is not stable or strong enough.
- `not_supported_current_data`: fields exist but current data does not show a
  useful signal.
- `not_assessable`: required fields or joins are missing.
- `observed_process_response`: process fields show observed downstream response
  evidence, not automatic dosing authorization.

## Current Acceptance Sample
`make final-feed` remains as a local 2024/2025 acceptance workflow. It preserves
the existing feasibility baseline and is not the only enterprise workflow.
