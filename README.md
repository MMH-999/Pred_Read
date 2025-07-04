# Pred_Read

# Random Forest Readmission Risk Prediction

This repository contains the code and a minimal exemplary data used for training and evaluating a Random Forest classifier to predict hospital readmission risk.

## Overview

- Trains a random forest classifier for three patient groups (here called: `HWR`, `Surg`, `Med_all`)
- Performs stratified train-test splitting
- Applies simple feature selection using univariate F-tests
- Trains across 20 different random seeds (splits)
- Optionally removes low-importance features in the second run of each split (`ts_mode=2`)
- Saves for subsequent analysis:
  - Predicted probabilities on test data
  - Feature importances for model interpretation

- Please note: If the code is used on actual data, the categorical variables must already be encoded as dummy variables in the data before using the provided script (e.g., ICD codes)

## Repository Structure

```bash
.
├── rf_pipeline.py           # Main pipeline script
├── test_data.csv            # Minimal working example dataset
├── README.md                # This file
├── LICENSE                  # Public domain license (CC0)
├── outputs/
│   ├── predicted_probs_<disease>_<split>.csv
│   └── feature_importances_<disease>_<split>.csv
