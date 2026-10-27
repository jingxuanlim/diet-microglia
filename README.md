# diet-microglia: processing of microglia two-photon time-lapse imaging

Code accompanying:

> Yusoff, D. M.†, Chua, K. X.†, Phang, S. K. A., Chaganty, V., Tham, V. Y.-Y., Lee, T., Lim, J.-X., Cheng, R.-K., Lie, W., Koh, A., Hu, J., Ng, J. L., Ang, A. X. L., Yip, K., Klumby, E., Wei, Y. & Wee, C. L.\* Dietary choline disrupts sleep via a microbiome-dependent neuro-immune axis.

The pipeline is in [`notebooks/process-microglia.ipynb`](notebooks/process-microglia.ipynb).
It takes raw Nikon `.nd2` stacks to per-timepoint microglia masks and morphology tables:

1. Extract microglia candidates, masks and brain outlines.
2. Estimate rigid motion-correction transforms on the brain outline and apply them to the outline, microglia-mask and raw stacks.
3. Label and characterise motion-corrected microglia (per-timepoint morphology table).

The code it calls is in the `microbiome` package (`src/microbiome/`).

## Installation

Requires Python 3.11.

```bash
conda env create -f environment.yml   # exact versions used; installs microbiome in editable mode
conda activate diet-microglia
```

or, into an existing environment: `pip install -e ".[notebook]"`.

Both install a [fork of bigstream](https://github.com/jingxuanlim/bigstream) pinned to a commit.
The notebook needs the fork's changes to `bigstream.motion_correct.motion_correct`.

## Input data

Each dataset is a directory listed in the `rootdir` column of a Google Sheet:

```
<...>/<YYYYMMDD>/fishNN/        ← rootdir (absolute path, writable)
├── im_CM0/<name>.nd2           ← raw time-lapse
└── fishNN.json                 ← optional; {"baseline": "<rootdir of baseline dataset>"}
```

The sheet columns used are `rootdir`, `drug`, `slice`, `status` (only `processing` rows are run) and
`stacksize` (only `200um` rows are run).

## Running

Set these environment variables, then run the notebook:

| Variable | Meaning |
| --- | --- |
| `GSHEET_KEYFILE` | Google service-account JSON key for reading the sheet. |
| `ANALYSIS_DIR` | Directory for collection-level outputs. |
| `GSHEET_NAME`, `GSHEET_WORKSHEET` | Optional. Spreadsheet and worksheet names. Defaults: `experiments`, `diet_brain-microglia`. |
| `MICROBIOME_TEMP_DIR` | Optional. Fast local scratch disk. Default: system temp directory. |

```bash
jupyter lab notebooks/process-microglia.ipynb
```

## Output

`microglia_df.pkl`, written to `<rootdir>/processed/` for each dataset: the per-microglia,
per-timepoint morphology table.
