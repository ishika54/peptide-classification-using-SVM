import argparse
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import os

def generate_fasta(df, filename, include_labels=True):
    with open(filename, "w") as f:
        for i, row in df.iterrows():
            if include_labels:
                header = f">seq{i}|{row['labels']}"
            else:
                header = f">seq{i}"
            sequence = row['sequence'] if 'sequence' in df.columns else row[' Sequence']
            f.write(f"{header}\n{sequence}\n")

def run_pfeature(input_fasta, method, output_csv):
    print(f"Running Pfeature for {method} ...")
    subprocess.run(["pfeature_comp", "-i", input_fasta, "-j", method, "-o", output_csv], check=True)
    print(f"{method} features saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Protein classification using Pfeature and SVM.")
    parser.add_argument("-i", "--input", required=True, help="Path to training CSV file (with sequence and labels).")
    parser.add_argument("-t", "--test", required=True, help="Path to test CSV file (without labels).")
    parser.add_argument("-g", "--generate", default="AAC", help="Feature generation method (AAC, DPC, etc.). Default=AAC")
    parser.add_argument("-o", "--output", default="submission.csv", help="Output CSV file for predictions.")
    args = parser.parse_args()

    train_file = args.input
    test_file = args.test
    method = args.generate
    output_file = args.output

    # Load training data
    df = pd.read_csv(train_file)
    print("\nTraining Data:")
    print(df.head())

    # Ensure columns are correctly named
    df.columns = ['sequence', 'labels']
    labels = df['labels']

    # Generate FASTA for training
    train_fasta = "train.fasta"
    generate_fasta(df, train_fasta, include_labels=True)

    # Generate training features
    train_feature_file = f"{method}.csv"
    run_pfeature(train_fasta, method, train_feature_file)

    aac_features = pd.read_csv(train_feature_file).reset_index(drop=True)
    aac_features = pd.concat([aac_features, labels], axis=1)

    # Load test data
    test_df = pd.read_csv(test_file)
    test_fasta = "test.fasta"
    generate_fasta(test_df, test_fasta, include_labels=False)

    test_feature_file = f"test_{method}.csv"
    run_pfeature(test_fasta, method, test_feature_file)

    test_aac_features = pd.read_csv(test_feature_file).reset_index(drop=True)

    # Prepare features
    X_features = aac_features.drop('labels', axis=1)
    y = aac_features['labels'].values

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_features, y, test_size=0.15, random_state=42, stratify=y
    )

    # Train SVM
    print("\nTraining SVM model...")
    clf = SVC(kernel='rbf', class_weight='balanced', random_state=0, probability=True)
    clf.fit(X_train, y_train)

    # Validation results
    y_val_pred = clf.predict(X_val)
    print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    # Predict on test
    X_test = test_aac_features
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Prepare submission
    submission_df = pd.DataFrame({
        '# ID': test_df['# ID'],
        'Label': y_prob
    })
    submission_df.to_csv(output_file, index=False)
    print(f"\nTest predictions saved to {output_file}")

    # ROC Curve
    y_val_prob = clf.predict_proba(X_val)[:, 1]
    auc_plt = roc_auc_score(y_val, y_val_prob)
    print("Validation ROC AUC Score:", auc_plt)

    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig("roc_curve.png", dpi=300)
    # plt.show()

if __name__ == "__main__":
    main()
