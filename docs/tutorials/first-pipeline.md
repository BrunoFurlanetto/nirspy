# Your First Pipeline

This tutorial walks you through building a complete fNIRS processing
pipeline in NIRSPY, from loading data to computing the hemodynamic
response function (HRF).

## The Standard fNIRS Pipeline

A typical fNIRS pipeline follows these steps
(Yuecel et al., 2021 -- Best Practices for fNIRS):

1. **Load** -- read the SNIRF file
2. **Optical Density** -- convert raw intensity to OD
3. **Motion Correction** -- remove motion artifacts (optional, v0.2+)
4. **Beer-Lambert Law** -- convert OD to HbO/HbR concentrations
5. **Bandpass Filter** -- remove slow drift and physiological noise
6. **Quality Control** -- compute SCI and prune bad channels
7. **Block Average** -- epoch and average the HRF per condition

## Step-by-step in the GUI

### 1. Launch NIRSPY

```bash
nirspy serve
```

### 2. Upload your SNIRF file

Click the **Upload** button in the top bar and select your `.snirf` file.

### 3. Build the pipeline

Add blocks from the catalog (left panel) in this order:

| Block | Purpose |
|-------|---------|
| Load SNIRF | Read the data file |
| Optical Density | Intensity to OD conversion |
| Beer-Lambert Law | OD to HbO/HbR |
| Bandpass Filter | Remove noise (default: 0.01--0.5 Hz) |
| Scalp Coupling Index | Measure channel quality |
| Prune Channels | Flag bad channels |
| Block Average | Compute HRF per condition |

### 4. Configure parameters

Click any block to see its parameters in the right panel.

**Bandpass Filter:**

- Low cutoff: 0.01 Hz (removes slow drift)
- High cutoff: 0.5 Hz (removes cardiac, ~1 Hz)

**Prune Channels:**

- SCI threshold: 0.7 (channels below this are flagged as bad)

**Block Average:**

- tmin: -2.0 s (baseline start)
- tmax: 18.0 s (epoch end)

### 5. Run the pipeline

Click **Run**. The progress bar shows execution status.

### 6. Inspect results

After execution, check the tabs:

- **Raw Data** -- time series of the loaded data
- **QC Dashboard** -- heatmap of SCI values per channel
- **HRF Plot** -- average HbO/HbR response per condition

### 7. Save the pipeline

Click **Save Pipeline** to download a YAML file. You can reload this
pipeline later or run it from the command line:

```bash
nirspy run my-pipeline.yml --input data.snirf --output results/
```

## References

- Yuecel, M. A., et al. (2021). Best practices for fNIRS publications.
  *Neurophotonics*, 8(1), 012101.
