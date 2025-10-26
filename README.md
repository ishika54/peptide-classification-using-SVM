# Peptide Classification Using SVM and Pfeature

## Overview
This project implements a **peptide classification model** using **Support Vector Machine (SVM)** in Python.  
The features for the peptides are extracted using **Pfeature**, specifically **Amino Acid Composition (AAC)**.  
The model predicts peptide labels (positive/negative) based on these features.  

---

## Authors
- **Ishika Gupta (MT25180)**  
- **Kirti Singh (MT25212)**  
- First Year M.Tech, Computational Biology, IIIT Delhi  

---

## Features

- **Input:** CSV files containing peptide sequences and labels (for training) or sequences only (for testing).  
- **Feature Extraction:** Uses Pfeature to generate descriptors like AAC (Amino Acid Composition), DPC (Dipeptide Composition), etc.  
- **Modeling:** Trains an SVM classifier with an RBF kernel.  
- **Validation:** Uses stratified train-validation split for model evaluation.  
- **Performance Metrics:** Computes accuracy, classification report, and ROC AUC score.  
- **Output:** Generates predictions for test data and saves a submission CSV.  
- **Visualization:** ROC curve is plotted and saved as `roc_curve.png`.  

---

## Why Radial SVM?
- **Radial Basis Function (RBF) kernel** can handle **non-linear relationships** in data.  
- AAC features can be highly correlated and non-linearly separable; hence RBF is preferred over linear kernel.  
- Ensures better generalization for peptide classification tasks.

---

## Input Data Format

### Training CSV
- Must contain **two columns**:
  - `sequence`: Amino acid sequences of peptides  
  - `labels`: Target label (1 or -1)  

Example:

| sequence   | labels |
|------------|--------|
| ACDEFGHIK  | 1      |
| LMNPQRSTV  | -1     |

### Test CSV
- Must contain **one column**:
- `sequence`: Amino acid sequences of peptides  
- Optionally, an `# ID` column for identifying peptides in output  

---
## Requirements

- Python >= 3.8  
- Required Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
- Pfeature installed and accessible via command line (`pfeature_comp`). 

## Usage

### **Execution Command**
```bash
python3 code_final_2.py -i train.csv -t test.csv

**Arguments**

Flag	Description	Default
-i, --input	Path to training CSV file	Required
-t, --test	Path to test CSV file	Required
-g, --generate	Feature extraction method (AAC, DPC, etc.)	AAC
-o, --output	Output CSV file for predictions	output_features.csv

**Methods Implemented**

FASTA Conversion: Converts CSV sequences to FASTA format required by Pfeature

Feature Extraction:

Uses Pfeature command-line tool:

    "pfeature_comp -i input.fasta -j AAC -o output.csv"

AAC (Amino Acid composition) features calculated for each peptide

SVM Classification:

- Radial kernel (RBF)

- class_weight='balanced' to handle class imbalance

- probability=True to compute prediction probabilities and ROC AUC

Evaluation:

- Accuracy and classification report on validation set

- ROC curve plotting with AUC calculation

**Output**

output_features.CSV containing predicted probabilities for test peptides:
| # ID | Label |
| ---- | ----- |
| 1    | 0.85  |
| 2    | 0.12  |

- **ROC Curve**: Saved as `roc_curve.png`  
  ![ROC Curve of validation set](roc_curve.png)


**Model Performance:**

- Validation Accuracy: 0.7897

- Validation ROC AUC Score: 0.844

**Notes:**

- Ensure pfeature_comp is installed and added to your system PATH.
- The script automatically handles column names and sequence formatting for Pfeature.

**References:**

- Pfeature: A comprehensive tool for calculating protein/peptide descriptors GitHub.

- Scikit-learn: Machine learning in Python Documentation.

- SVM for Peptide Classification: Standard supervised learning method using amino acid composition or other descriptors.