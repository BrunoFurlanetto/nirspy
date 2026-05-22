# Getting Started

This guide walks you through installing NIRSPY and running your first
fNIRS pipeline.

## Prerequisites

- Python 3.10 or later
- A SNIRF file with fNIRS data (`.snirf`)

## Installation

### From PyPI

```bash
pip install nirspy
```

### From source (development)

```bash
git clone https://github.com/BrunoFurlanetto/nirspy.git
cd nirspy
uv venv
uv pip install -e ".[dev]"
```

## Launch the GUI

```bash
nirspy serve
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

## Your First Pipeline

1. **Upload a SNIRF file** -- click the upload button and select your `.snirf`
2. **Add blocks** -- click blocks from the catalog on the left:
   - Load SNIRF
   - Optical Density
   - Beer-Lambert Law
   - Bandpass Filter (0.01 -- 0.5 Hz)
   - Scalp Coupling Index
   - Prune Channels
   - Block Average
3. **Configure parameters** -- click a block to edit its parameters
4. **Run** -- click the Run button
5. **Inspect results** -- view the raw data plot, QC dashboard, and HRF

## Command-Line Interface

```bash
# Run a saved pipeline
nirspy run pipeline.yml --input data.snirf --output results/

# Run with verbose logging
nirspy run pipeline.yml --input data.snirf --output results/ --verbose
```

## Convert Files

NIRSPY includes a file converter for:

- `.nirs` (HOMER/MATLAB) to `.snirf`
- `.snirf` to `.nirs`
- Oxysoft `.txt` export to `.snirf`

Use the **Convert** tab in the GUI, or the Python API:

```python
from nirspy.io.converters import nirs_to_snirf, snirf_to_nirs

nirs_to_snirf("data.nirs", "data.snirf")
snirf_to_nirs("data.snirf", "data.nirs")
```
