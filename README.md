# mne_python_analysis

**Version:** 4.1.0  
**Lab:** Action & Cognition Laboratory, National Central University (NCU)  
**Supervisor:** Prof. Erik Chung  
**Contact:** Dillian (HE-JUN, CHEN)

---

## Overview

This repository contains a Python-based EEG analysis pipeline for the **Alternating Serial Reaction Time (ASRT)** experiment. The pipeline processes raw EEG data (Neuroscan `.cnt` / `.cdt` format) through preprocessing, ICA artifact removal, epoching, and ERSP (Event-Related Spectral Perturbation) time-frequency analysis.

The pipeline is designed to investigate neural oscillatory correlates of **motor and perceptual sequence learning**, with a focus on theta (4–8 Hz) and alpha (8–13 Hz) frequency bands.

---

## Experiment Design

- **Task:** ASRT — 8-element alternating sequence (pattern + random interleaved)
- **EEG System:** 32-channel g.tec / Neuroscan (Grael V2), downsampled to 512 Hz
- **Conditions:** Regular (High-probability triplets) vs. Random (Low-probability triplets)
- **Phases:** Learning Phase + Testing Phase (Motor / Perceptual interleaved)
- **Counterbalancing:** Group A (Motor first); Group B (Perceptual first)

---

## Analysis Parameters

### Preprocessing
| Parameter | Value |
|---|---|
| Bandpass filter | 0.5–40 Hz |
| Notch filter | 60 Hz |
| Sampling rate | 512 Hz |
| Reference | Average reference (30 scalp electrodes) |
| Artifact removal | ICA (ocular + muscle) |

### Epoch Design

| Parameter | Stimulus-locked | Response-locked |
|---|---|---|
| Time window | -0.8 to +1.0 s | -1.1 to +0.5 s |
| Time-domain baseline | -500 to -100 ms | -500 to -100 ms (pre-stimulus axis) |
| Freq-domain baseline | -500 to -100 ms | -1000 to -600 ms (fixed window, during RSI) |
| Main analysis window | +100 to +300 ms | -300 to +50 ms |
| Main frequency band | Alpha (8–13 Hz) | Theta (4–8 Hz) |
| ROI | O1, Oz, O2, P3, Pz, P4 | Fz, FCz, Cz, C3, C4 |

> **Note on Response-locked epochs:** Response-locked epochs are derived from stimulus-locked epochs via RT alignment (interpolation), not by cutting directly from raw EEG. Time-domain baseline correction must be applied *before* RT alignment, as the baseline window is defined on the stimulus time axis.

> **Note on fixed-window baseline:** The fixed window (-1000 to -600 ms) corresponds to the RSI (inter-stimulus interval), representing a pre-stimulus resting state. This approach follows Lum et al. (2023). A known limitation is that for trials with RT > 600 ms, stimulus-evoked activity may partially overlap with the baseline window; however, this effect is diluted by trial averaging.

### ERSP Computation
- **Method:** Morlet wavelet (complex)
- **Baseline mode:** logratio (dB = 10 × log₁₀ (P / baseline))
- **Zero-padding:** 1024 FFT points (for theta/alpha frequency resolution)

---

## Repository Structure

```
mne_python_analysis/
├── main.py                           # Main entry point (interactive menu)
├── data_io.py                        # Data loading and saving
├── preprocessing.py                  # Filtering, downsampling, referencing
├── ica_analysis.py                   # ICA artifact removal
├── epochs.py                         # Epoch creation
├── group_ersp_analysis.py            # Group-level ERSP analysis (v2.1)
├── asrt_response_ersp_from_epochs.py # Response-locked ERSP (Option 21)
├── extract_rt_precise.py             # RT extraction from EEG trigger channel
├── response_lock.py                  # RT alignment / interpolation
├── roi_analysis.py                   # ROI-based analysis
├── spectral_analysis.py              # Spectral utilities
├── statistical_analysis.py           # Statistical analysis
├── signal_processing.py              # Signal processing utilities
├── montage.py                        # Electrode montage configuration
├── microstate.py                     # Microstate analysis (experimental)
├── utils.py                          # General utilities
├── ui/
│   ├── menu.py                       # Interactive menu display
│   └── prompts.py                    # User input prompts
└── asrt/
    ├── ersp.py                       # ERSP computation core
    ├── ersp_plots.py                 # ERSP visualization
    └── workflows.py                  # ASRT analysis workflows
```

---

## Dependencies

- Python 3.8+
- [MNE-Python](https://mne.tools/) >= 1.0
- NumPy
- SciPy
- Matplotlib

Install dependencies:
```bash
pip install mne numpy scipy matplotlib
```

---

## Third-Party Components

This pipeline uses the **CURRY Python Reader** for loading Neuroscan `.cdt` / `.cnt` files.

- **Source:** [https://github.com/neuroscan/curry-python-reader](https://github.com/neuroscan/curry-python-reader)
- **License:** BSD 3-Clause (see below)
- **Usage:** The `curryreader.py` module is included as-is (latest version as of 2026-03).

```
BSD 3-Clause License
Copyright (c) 2021, Compumedics Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
```

---

## Key References

- Lum, J. A. G., et al. (2023). Theta oscillations and sequence learning. *Psychophysiology*, 60, e14179.
- Hamamé, C. M., et al. (2011). Alpha modulation in perceptual learning. *PLoS ONE*, 6(4), e19221.
- Simor, P., et al. (2025). ASRT probabilistic learning and EEG oscillations. *Journal of Neuroscience*, 45(19), e1421242025.

---

## Notes

- EEG data files (`.cnt`, `.fif`, `.h5`) are excluded from this repository via `.gitignore`.
- This pipeline was developed for a pilot study (N=3); results are preliminary and without statistical testing.
- Current version: **v4.1.0** (2026-03)
