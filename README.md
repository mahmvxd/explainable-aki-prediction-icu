# Explainable AKI Prediction in ICU Patients

An explainable machine learning pipeline for early prediction of Acute Kidney Injury (AKI) in ICU patients using the PhysioNet 2019 Challenge dataset.

## Project Overview

This project predicts the risk of **AKI within the next 24 hours** using ICU time-series data and engineered physiological features such as creatinine trends, blood pressure patterns, lactate, and BUN.

The pipeline includes:

- KDIGO-inspired AKI labeling from creatinine trajectories
- Temporal feature engineering using rolling windows and deltas
- Logistic Regression baseline model
- HistGradientBoosting classifier
- SHAP explainability for global and individual predictions

## Dataset

This project uses the **PhysioNet/Computing in Cardiology Challenge 2019** ICU dataset.

- 36,320 ICU patients
- Hourly physiological and laboratory measurements
- Publicly accessible from PhysioNet

**Note:** The raw dataset is not included in this repository.

## Prediction Task

Predict whether a patient will develop **AKI within the next 24 hours**.

## Label Definition

AKI labels were generated using a KDIGO-inspired creatinine rule:

- increase in creatinine by **>= 0.3 mg/dL** relative to rolling 48-hour minimum
- or creatinine **>= 1.5x baseline**

## Final Results

### Full dataset performance

| Model | AUROC | AUPRC | Brier |
|------|------:|------:|------:|
| Logistic Regression | 0.822 | 0.348 | 0.072 |
| HistGradientBoosting | 0.873 | 0.489 | 0.062 |

## Risk Stratification

Using the final HistGradientBoosting model:

| Risk Band | AKI Rate |
|----------|---------:|
| Low | 1.9% |
| Medium | 11.4% |
| High | 42.6% |

## Explainability

SHAP analysis showed that the most influential features included:

- Creatinine_mean_1h
- cr_above_rollmin_48h
- MAP_mean_48h
- BUN features
- SBP_min_12h
- Age
- ICU length of stay

## Repository Structure

```text
notebooks/           model development notebook
outputs/reports/     final metrics
outputs/figures/     SHAP plots
models/              saved trained models
