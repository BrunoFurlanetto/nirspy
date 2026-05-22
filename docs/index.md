# NIRSPY — NIRS Processing in Python

> GUI-first, fNIRS-focused pipeline builder in Python.
> Runs locally in your browser. Wraps [MNE-NIRS](https://mne.tools/mne-nirs/)
> so every pipeline remains reproducible and inspectable.

## What is NIRSPY?

NIRSPY is a free, open-source (BSD-3) tool for functional near-infrared
spectroscopy (fNIRS) data processing. It provides a visual pipeline builder
that runs locally in your browser -- no MATLAB license required.

Key features:

- **Modular pipeline builder** -- drag-and-drop blocks for preprocessing,
  quality control, and analysis
- **Reproducible YAML pipelines** -- save, share, and reload your exact
  processing steps
- **Built on MNE-NIRS** -- leverages the battle-tested MNE ecosystem
- **No programming required** -- designed for researchers, not developers
- **File converters** -- convert `.nirs` (HOMER) to `.snirf` and back

## Quick Install

```bash
pip install nirspy
```

## Quick Start

```bash
# Launch the GUI
nirspy serve
# Open http://127.0.0.1:8050 in your browser

# Or run a pipeline from the command line
nirspy run pipeline.yml --input data.snirf --output results/
```

## Status

Pre-alpha (v0.1.0). Core pipeline engine, preprocessing, quality control,
and HRF block-average blocks are implemented. The visual pipeline builder
and Dash GUI are functional. See the [Roadmap](roadmap.md) for planned
features.

## License

BSD-3-Clause -- same as MNE-NIRS and the scientific Python ecosystem.
