import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Construct the file path
file_name = "zeroshot_qwen_2017_2025_03_13_133530_2025-03-13.csv"
file_path = os.path.join(os.getcwd(), "outputs", "generated", file_name)

# Read the CSV file
df = pd.read_csv(file_path)

# Filter out instances where the predicted label is "OOD"
filtered_df = df[df["Predicted Decoded Label"] != "OOD"]

# Extract y_true and y_pred
y_true = filtered_df["True Label"]
y_pred = filtered_df["Predicted Decoded Label"]

# Compute metrics
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")
report = classification_report(y_true, y_pred)

# Print the results
print("Classification Report:")
print(report)
print("\nMetrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
