# Exoplanet Classification AI/ML Project

## Project Overview

Our team aims to build an AI/ML-powered system to classify exoplanets using publicly available datasets from NASA's Kepler, K2, and TESS missions. The system will leverage the **transit method**, which measures brightness dips in light curves when a planet passes in front of a star, to classify objects as confirmed exoplanets, candidates, or false positives.

The goal is to reduce manual effort in exoplanet identification, create a user-friendly interactive tool, and enable researchers and enthusiasts to:

* Upload or input new light curve/tabular data.
* Obtain automated predictions with confidence scores.
* Visualize light curves, feature importance, and model results.
* Optionally adjust hyperparameters or retrain the model with new data.

## Roadmap / Workflow

### Step 1: Data Preprocessing

* Select relevant features: orbital period, transit depth, transit duration, planet/star radius.
* Clean data by removing missing values, outliers, and duplicates.
* Normalize or scale features as needed.

### Step 2: Model Training

* Train the model on **Kepler KOI Cumulative List + Certified False Positives**.
* Algorithms: Random Forest, CNN, RNN, or Transformer (based on tabular or light curve data).

### Step 3: Testing & Fine-Tuning

* Test on **Kepler Confirmed Names** to establish baseline accuracy.
* Optional: Fine-tune on **K2 Planets and Candidates** to improve generalization.

### Step 4: Cross-Mission Validation

* Validate on **TESS Project Candidates** to evaluate model generalization.
* Use evaluation metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC curves.

### Step 5: Web Interface / Deployment

* Enable users to upload new light curves or tabular data.
* Display predictions with confidence scores.
* Visualize features, light curves, and model performance metrics.

### Step 6: Optional Enhancements

* Hyperparameter tuning via the interface.
* Incorporate additional survey datasets for future expansion (CoRoT, SuperWASP, KELT, etc.).
* Retrain model dynamically with new user-provided data.

## Dataset Explanation and Usage

### 1. Kepler Mission

| Dataset                                                                                | Purpose / Use                                               | ML Use Case                     |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------- |
| KOI (Cumulative List)                                                                  | Updated KOIs with confirmed/candidate/false positive labels | Primary training data           |
| Certified False Positives                                                              | Known negatives                                             | Training data (balance classes) |
| KOI (All Lists)                                                                        | Every historical KOI version                                | Not recommended for training    |
| Threshold-Crossing Events                                                              | Raw transit events                                          | Optional advanced ML            |
| Kepler Confirmed Names                                                                 | Only confirmed planets                                      | Validation/testing              |
| Positional Probabilities, Completeness and Reliability, Simulated Data, Kepler Stellar | Optional features or augmentation                           | Feature enrichment or testing   |

### 2. K2 Mission

| Dataset                   | Purpose / Use                                    | ML Use Case                       |
| ------------------------- | ------------------------------------------------ | --------------------------------- |
| K2 Planets and Candidates | Candidates + confirmed planets + false positives | Optional training/fine-tuning     |
| K2 Confirmed Names        | Only confirmed planets                           | Validation/testing                |
| K2 Targets                | All observed stars                               | Optional negative/background data |

### 3. TESS Mission

| Dataset                 | Purpose / Use                                 | ML Use Case                                         |
| ----------------------- | --------------------------------------------- | --------------------------------------------------- |
| TESS Project Candidates | Planetary candidates & some confirmed planets | Testing/validation for cross-mission generalization |

### 4. Other Transit Surveys (Optional)

* CoRot Astero-Seismology, CoRoT Exoplanet, SuperWASP, KELT, XO, HATNet, Cluster, TrES
* Use for future data augmentation or cross-survey validation

### Important thing 
Dataset Purposes

Kepler KOI (Cumulative List)

Purpose: Training

Includes confirmed planets, candidates, and false positives

Clean and labeled → ideal for supervised ML training

Kepler Confirmed Names

Purpose: Testing / Validation

Only contains confirmed planets

No negative examples → not suitable for training, but good for checking model accuracy on true positives

K2 Planets and Candidates

Purpose: Optional fine-tuning or validation

Includes candidates and confirmed planets → can train if you want to extend the model

K2 Confirmed Names

Purpose: Testing / Validation

Only confirmed planets → good for evaluating model performance

TESS Project Candidates

Purpose: Testing / Cross-mission validation

Includes candidates and some confirmed planets → good for evaluating generalization of model trained on Kepler
## Recommended Dataset Usage
Workflow for Three Separate Models

Model 1: Kepler KOI Cumulative List + Certified False Positives

Train the first model entirely on Kepler data.

Test on Kepler Confirmed Names.

Evaluate metrics: accuracy, precision, recall, F1-score, confusion matrix.

Model 2: K2 Planets and Candidates

Train the second model on K2 Planets and Candidates.

Test on K2 Confirmed Names.

Evaluate metrics similarly.

Model 3: TESS Project Candidates

Train the third model on TESS Project Candidates (if you have labels).

Test on a subset of confirmed planets or known candidates.

Evaluate metrics.

## Summary

* Focus on **Kepler for training**, **K2 for optional fine-tuning**, and **TESS for testing**.
* This ensures the model learns from clean, labeled data and generalizes well to new mission data.
* Other survey datasets can be integrated later for further model improvement.

---

*End of README*
