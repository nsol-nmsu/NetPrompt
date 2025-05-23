import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
import time

def main(model_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_name == "2017":
        load_model = "models/mult_cicids2017.pth"
        filepath = os.path.join(base_dir, "data", "processed", "CICIDS2017")
    elif model_name == "2019":
        load_model = "models/mult_cicddos2019.pth"
        filepath = os.path.join(base_dir, "data", "processed", "CICDDoS2019")
    
    # Load .npy files using numpy
    X_test = np.load(os.path.join(filepath, "X_test_ml.npy"))
    y_test = np.load(os.path.join(filepath, "y_test_ml.npy"))

    # Convert y_test to a DataFrame for easier handling (if it's not already)
    y_test = pd.DataFrame(y_test, columns=['label'])

    # Load the model directly
    model = torch.load(load_model)
    model.eval()  # Set the model to evaluation mode

    # Convert data to tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test['label'].values, dtype=torch.long)

    # Make predictions and measure inference time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
    end_time = time.time()

    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time / X_test_tensor.shape[0]

    # Convert tensors to numpy arrays for evaluation
    y_true = y_test_tensor.numpy()
    y_pred = predicted.numpy()

    # Evaluate the model
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print(f"\nAverage Inference Time per Sample: {avg_inference_time:.9f} seconds")
    print(f"Total Inference Time for All Samples: {total_inference_time:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run network traffic classification with selected model, prompt template, and dataset year.")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=["2017", "2019"],
        help="Model year: 2017 or 2019"
    )

    args = parser.parse_args()
    main(args.model)