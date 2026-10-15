# HG_analysis

Python code for analyzing high-gamma activity (70–160 Hz) in intracranial EEG recordings during a music memory task. The analyses compare correctly recognized old and new stimuli and examine principal component dynamics, behavioral associations, decoding, and Granger causality.

The full project code is available in [MIB_iEEG](https://github.com/barbaragrosjean/MIB_iEEG).

## Contents

- `LB0_preprocessing.py`: Preprocess recordings and save epochs.
- `LB1_computePCA.py`: Compute principal component analysis (PCA) at the group and individual subject levels.
- `LB2_behavior.py`: Examine correlations between principal component features and reaction times using resampling and permutation tests.
- `LB3_decoding.py`: Run classification analyses using principal component features.
- `LB4_granger_causality.py`: Estimate time-resolved Granger causality between principal components using surrogate testing.
- `main.ipynb`: Explore and visualize network dynamics, decoding, behavioral analyses, and Granger causality results.
- `src/`: Supporting functions and analysis configuration.

## Usage

The dataset is not included in this repository. The scripts require project-specific recordings, behavioral data, and intermediate analysis files.

Before running the analyses:

1. Set `PROJECT_PATH`, `DATA_PATH`, and `OUT_PATH` in `src/config.py`.
2. Review the data paths, participant selections, and analysis parameters in each script.
3. Prepare the required input files, then run the relevant scripts and use `main.ipynb` to inspect the results.

Some paths and settings are defined directly in the scripts and must be adapted to your local environment.
