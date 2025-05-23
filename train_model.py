#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import time
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from model import MLP_Mult  # Assumes you have defined your model in model.py
import logging

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def log_print(message):
    """Helper function to log a message (both to file and terminal)."""
    logging.info(message)

def data_loader(model_year):
    """
    Load the saved ML npy data for the given model year.
    """
    if model_year == "2017":
        data_path = os.path.join(os.getcwd(), "data", "processed", "CICIDS2017")
    elif model_year == "2019":
        data_path = os.path.join(os.getcwd(), "data", "processed", "CICDDoS2019")
    else:
        raise ValueError("Model year must be '2017' or '2019'.")
    
    X_train = np.load(os.path.join(data_path, "X_train_ml.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(data_path, "X_test_ml.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(data_path, "y_train_ml.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(data_path, "y_test_ml.npy"), allow_pickle=True)
    
    return X_train, X_test, y_train, y_test

def prepare_datasets(X_train, X_test, y_train, y_test, batch_size=32):
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).long().squeeze())
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).long().squeeze())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        log_print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    log_print(f"Accuracy: {accuracy:.4f}")
    log_print(f"Precision: {precision:.4f}")
    log_print(f"Recall: {recall:.4f}")
    log_print(f"F1-Score: {f1:.4f}")
    
    report = classification_report(all_labels, all_preds, zero_division=0)
    log_print("\nClassification Report:\n" + report)
    return report

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate model for CICIDS2017 or CICDDoS2019")
    parser.add_argument("-m", "--model", required=True, choices=["2017", "2019"], help="Model year: 2017 or 2019")
    args = parser.parse_args()
    
    model_year = args.model

    # Set up logging based on dataset
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    if model_year == "2017":
        log_filename = os.path.join(outputs_dir, "train_model_cicids2017.log")
    else:
        log_filename = os.path.join(outputs_dir, "train_model_cicddos2019.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    X_train, X_test, y_train, y_test = data_loader(model_year)
    
    log_print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    log_print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    log_print("Unique values in y_train: " + str(np.unique(y_train)))
    log_print("Unique values in y_test: " + str(np.unique(y_test)))
    
    batch_size = 32
    train_loader, test_loader = prepare_datasets(X_train, X_test, y_train, y_test, batch_size=batch_size)
    
    input_shape = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    log_print(f"Input shape: {input_shape}")
    log_print(f"Number of classes: {num_classes}")
    
    # Hyperparameters
    num_epochs = 25
    first_layer = 64
    second_layer = 128
    third_layer = 64
    learning_rate = 0.01
    
    model_instance = MLP_Mult(input_shape=input_shape, first_hidden=first_layer, second_hidden=second_layer,
                              num_units=third_layer, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_instance.parameters(), lr=learning_rate)
    
    log_print("\nTraining model...")
    start_time = time.time()
    train_model(model_instance, train_loader, criterion, optimizer, num_epochs=num_epochs)
    training_time = time.time() - start_time
    log_print(f"Training time: {training_time:.2f} seconds")
    
    log_print("\nEvaluating model...")
    _ = test_model(model_instance, test_loader)
    
    # Save model only
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    if model_year == "2017":
        model_path = os.path.join(models_dir, "mult_cicids2017.pth")
    else:
        model_path = os.path.join(models_dir, "mult_cicddos2019.pth")
    
    torch.save(model_instance, model_path)
    log_print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
