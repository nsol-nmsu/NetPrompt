import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def data_loader(year):
    """
    Load the dataset based on the provided year.
    """
    if year == 2019:
        file_path = os.path.join(os.getcwd(), "data", "original", "CICDDoS2019", "CICDDoS2019.csv")
    elif year == 2017:
        file_path = os.path.join(os.getcwd(), "data", "original", "CICIDS2017", "CICIDS2017.csv")
    else:
        raise ValueError("Year must be 2017 or 2019.")
    data = pd.read_csv(file_path)
    
    # Clean column names and drop unwanted columns
    data.columns = [col.strip() for col in data.columns]
    data = data.drop(columns='Flow ID')
    if year == 2019:
        # data = data.drop(columns=['Unnamed: 0', 'Fwd Header Length.1', 'Destination Port'])
        data = data.drop(columns=['Unnamed: 0'])

    print(f"Feature names for {os.path.basename(file_path)}:")
    print(data.columns.tolist())
    print("Data shape:", data.shape)
    return data

def get_feature_descriptions():
    """
    Returns a dictionary mapping feature names to their descriptions.
    
    (Note: Instead of focusing solely on common features, you can inspect unique features and their dtypes separately.)
    """
    return {
        # Extra columns to keep:
        "Source IP": "Source IP address",
        "Source Port": "Source port number",
        "Destination IP": "Destination IP address",
        "Protocol": "Protocol number",
        "Timestamp": "Timestamp of the flow",
        
        # Unique features for each dataset:
        "Destination Port": "Destination port number on the target host for the flow",
        "External IP": "External IP address involved in the connection (indicates communication outside the local network)",
        "SimillarHTTP": "Indicator of similar HTTP traffic",
        "Inbound": "Indicator of inbound traffic",
        "Fwd Header Length.1": "Alternate measurement of total bytes used for headers in the forward direction",
        
        # Feature descriptions from CICFlowMeter:
        "Flow Duration": "Duration of the flow in microseconds",
        "Total Fwd Packets": "Total packets in the forward direction",
        "Total Backward Packets": "Total packets in the backward direction",
        "Total Length of Fwd Packets": "Total size of packets in the forward direction",
        "Total Length of Bwd Packets": "Total size of packets in the backward direction",
        "Fwd Packet Length Min": "Minimum size of packet in the forward direction",
        "Fwd Packet Length Max": "Maximum size of packet in the forward direction",
        "Fwd Packet Length Mean": "Mean size of packet in the forward direction",
        "Fwd Packet Length Std": "Standard deviation size of packet in the forward direction",
        "Bwd Packet Length Min": "Minimum size of packet in the backward direction",
        "Bwd Packet Length Max": "Maximum size of packet in the backward direction",
        "Bwd Packet Length Mean": "Mean size of packet in the backward direction",
        "Bwd Packet Length Std": "Standard deviation size of packet in the backward direction",
        "Flow Bytes/s": "Number of flow bytes per second",
        "Flow Packets/s": "Number of flow packets per second",
        "Flow IAT Mean": "Mean time between two packets sent in the flow",
        "Flow IAT Std": "Standard deviation time between two packets sent in the flow",
        "Flow IAT Max": "Maximum time between two packets sent in the flow",
        "Flow IAT Min": "Minimum time between two packets sent in the flow",
        "Fwd IAT Min": "Minimum time between two packets sent in the forward direction",
        "Fwd IAT Max": "Maximum time between two packets sent in the forward direction",
        "Fwd IAT Mean": "Mean time between two packets sent in the forward direction",
        "Fwd IAT Std": "Standard deviation time between two packets sent in the forward direction",
        "Fwd IAT Total": "Total time between two packets sent in the forward direction",
        "Bwd IAT Min": "Minimum time between two packets sent in the backward direction",
        "Bwd IAT Max": "Maximum time between two packets sent in the backward direction",
        "Bwd IAT Mean": "Mean time between two packets sent in the backward direction",
        "Bwd IAT Std": "Standard deviation time between two packets sent in the backward direction",
        "Bwd IAT Total": "Total time between two packets sent in the backward direction",
        "Fwd PSH Flags": "Number of times the PSH flag was set in packets traveling in the forward direction (0 for UDP)",
        "Bwd PSH Flags": "Number of times the PSH flag was set in packets traveling in the backward direction (0 for UDP)",
        "Fwd URG Flags": "Number of times the URG flag was set in packets traveling in the forward direction (0 for UDP)",
        "Bwd URG Flags": "Number of times the URG flag was set in packets traveling in the backward direction (0 for UDP)",
        "Fwd Header Length": "Total bytes used for headers in the forward direction",
        "Bwd Header Length": "Total bytes used for headers in the backward direction",
        "Fwd Packets/s": "Number of forward packets per second",
        "Bwd Packets/s": "Number of backward packets per second",
        "Min Packet Length": "Minimum length of a packet",
        "Max Packet Length": "Maximum length of a packet",
        "Packet Length Mean": "Mean length of a packet",
        "Packet Length Std": "Standard deviation length of a packet",
        "Packet Length Variance": "Variance length of a packet",
        "FIN Flag Count": "Number of packets with FIN",
        "SYN Flag Count": "Number of packets with SYN",
        "RST Flag Count": "Number of packets with RST",
        "PSH Flag Count": "Number of packets with PUSH",
        "ACK Flag Count": "Number of packets with ACK",
        "URG Flag Count": "Number of packets with URG",
        "CWE Flag Count": "Number of packets with CWR",
        "ECE Flag Count": "Number of packets with ECE",
        "Down/Up Ratio": "Download and upload ratio",
        "Average Packet Size": "Average size of packet",
        "Avg Fwd Segment Size": "Average size observed in the forward direction",
        "Avg Bwd Segment Size": "Average size observed in the backward direction",
        "Fwd Avg Bytes/Bulk": "Average number of bytes bulk rate in the forward direction",
        "Fwd Avg Packets/Bulk": "Average number of packets bulk rate in the forward direction",
        "Fwd Avg Bulk Rate": "Average number of bulk rate in the forward direction",
        "Bwd Avg Bytes/Bulk": "Average number of bytes bulk rate in the backward direction",
        "Bwd Avg Packets/Bulk": "Average number of packets bulk rate in the backward direction",
        "Bwd Avg Bulk Rate": "Average number of bulk rate in the backward direction",
        "Subflow Fwd Packets": "The average number of packets in a sub flow in the forward direction",
        "Subflow Fwd Bytes": "The average number of bytes in a sub flow in the forward direction",
        "Subflow Bwd Packets": "The average number of packets in a sub flow in the backward direction",
        "Subflow Bwd Bytes": "The average number of bytes in a sub flow in the backward direction",
        "Init_Win_bytes_forward": "The total number of bytes sent in initial window in the forward direction",
        "Init_Win_bytes_backward": "The total number of bytes sent in initial window in the backward direction",
        "act_data_pkt_fwd": "Count of packets with at least 1 byte of TCP data payload in the forward direction",
        "min_seg_size_forward": "Minimum segment size observed in the forward direction",
        "Active Mean": "Mean time a flow was active before becoming idle",
        "Active Std": "Standard deviation time a flow was active before becoming idle",
        "Active Max": "Maximum time a flow was active before becoming idle",
        "Active Min": "Minimum time a flow was active before becoming idle",
        "Idle Mean": "Mean time a flow was idle before becoming active",
        "Idle Std": "Standard deviation time a flow was idle before becoming active",
        "Idle Max": "Maximum time a flow was idle before becoming active",
        "Idle Min": "Minimum time a flow was idle before becoming active",
    }

def compute_exemplars_context(X, y, num_exemplars_per_class):
    """
    Computes exemplars for each class based on the Euclidean distance to the class mean.
    
    Args:
        X (DataFrame): Numeric feature matrix.
        y (DataFrame/Series): Corresponding labels (with column "Label").
        num_exemplars_per_class (int): Number of exemplars to select per class.
    
    Returns:
        DataFrame: Exemplar instances with a "rank" column (1 = best representative).
    """
    df = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name="Label")
    df["Label"] = y["Label"]
    
    exemplar_list = []
    for class_label in df["Label"].unique():
        df_label = df[df["Label"] == class_label].copy()
        # Compute mean vector (from numeric columns only)
        mean_vector = df_label.drop(columns=["Label"]).mean()
        # Compute Euclidean distance for each row
        distances = df_label.drop(columns=["Label"]).apply(lambda row: np.linalg.norm(row - mean_vector), axis=1)
        df_label["distance"] = distances
        # Sort by distance (closest = more representative)
        df_label_sorted = df_label.sort_values("distance")
        # Select top exemplars for this class
        df_exemplars = df_label_sorted.head(num_exemplars_per_class).copy()
        df_exemplars["rank"] = range(1, len(df_exemplars) + 1)
        exemplar_list.append(df_exemplars)
    df_exemplars_all = pd.concat(exemplar_list)
    df_exemplars_all = df_exemplars_all.drop(columns=["distance"])
    return df_exemplars_all

def process_data(year):
    # Load dataset
    data = data_loader(year)

    print("\nUnique features in this dataset and their data types:")
    print(data.dtypes)
    
    # Clean data and split
    print(f"\nShape of dataset ({year}):", data.shape)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    X = data.drop(columns=['Label'])
    y = data['Label']
    print(f"Shape after dropping NaN values: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Combine X_test and y_test, then balance by sampling per label
    test = X_test.copy()
    test['Label'] = y_test
    balanced_test = test.groupby('Label').apply(
        lambda group: group.sample(n=min(1000, len(group)), random_state=42)
    ).reset_index(drop=True)
    
    # Split back into X_test and y_test ensuring alignment
    X_test = balanced_test.drop(columns=['Label'])
    y_test = balanced_test['Label']
    
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame(name='Label')
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame(name='Label')
    
    print("\nUnique label counts in the balanced test set:")
    print(y_test['Label'].value_counts())
    print("Balanced X_test shape:", X_test.shape)
    print("Balanced y_test shape:", y_test.shape)
    
    # -------------------------
    # Label Encoding for ML Data:
    if year == 2017:
        label_mapping = {
            'BENIGN': 0,
            'Bot': 1, 
            'DDoS': 2,
            'DoS GoldenEye': 3,
            'DoS Hulk': 4,
            'DoS Slowhttptest': 5,
            'DoS slowloris': 6,
            'FTP-Patator': 7,
            'Heartbleed': 8,
            'Infiltration': 9,
            'PortScan': 10,
            'SSH-Patator': 11
        }
    else:
        label_mapping = {
            'BENIGN': 0,
            'UDP-lag': 1,
            'DrDoS_SSDP': 2,
            'DrDoS_DNS': 3,
            'DrDoS_MSSQL': 4,
            'DrDoS_NetBIOS': 5,
            'DrDoS_LDAP': 6,
            'DrDoS_NTP': 7,
            'DrDoS_UDP': 8,
            'Syn': 9,
            'DrDoS_SNMP': 10
        }
    
    # Create ML versions of y_train and y_test with label encoding
    y_train_ml = y_train.copy()
    y_test_ml = y_test.copy()
    y_train_ml["Label"] = y_train_ml["Label"].map(label_mapping)
    y_test_ml["Label"] = y_test_ml["Label"].map(label_mapping)
    
    # -------------------------
    # Save Data for ML:
    # For ML, select only numeric columns (drop object features)
    X_train_ml = X_train.select_dtypes(include=[np.number])
    X_test_ml = X_test.select_dtypes(include=[np.number])
    
    scaler = MinMaxScaler()
    X_train_ml = scaler.fit_transform(X_train_ml)
    X_test_ml = scaler.transform(X_test_ml)
    
    print("\nML Data Shapes:")
    print("  X_train_ml shape:", X_train_ml.shape)
    print("  X_test_ml shape:", X_test_ml.shape)
    
    if year == 2017:
        output_dir = os.path.join("data", "processed", "CICIDS2017")
    else:
        output_dir = os.path.join("data", "processed", "CICDDoS2019")
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X_train_ml.npy"), X_train_ml)
    np.save(os.path.join(output_dir, "y_train_ml.npy"), y_train_ml)
    np.save(os.path.join(output_dir, "X_test_ml.npy"), X_test_ml)
    np.save(os.path.join(output_dir, "y_test_ml.npy"), y_test_ml)
    
    print(f"\nML data saved to {output_dir}")
    
    # --- Print saved ML data shapes, unique instance counts, and unique classes ---
    print("\nSaved ML Data Shapes:")
    print(f"  X_train_ml shape: {X_train_ml.shape}")
    print(f"  y_train_ml shape: {y_train_ml.shape}")
    print(f"  X_test_ml shape: {X_test_ml.shape}")
    print(f"  y_test_ml shape: {y_test_ml.shape}")
    unique_ml_instances = np.unique(X_test_ml, axis=0).shape[0]
    print(f"Unique ML test instances: {unique_ml_instances}")
    print("\nUnique classes and their instance counts in ML data:")
    print(y_test_ml['Label'].value_counts())
    
    # -------------------------
    # Helper function to create the contextual DataFrame:
    def create_contextual_df(df, y_labels):
        # Drop any existing "prompt" column, then create a new one.
        df = df.copy().drop(columns=["prompt"], errors="ignore")
        df["prompt"] = df.apply(lambda row: ", ".join([f"{col} is {row[col]}" for col in df.columns]), axis=1)
        return pd.DataFrame({
            "X": df["prompt"],
            "y": y_labels["Label"]  # Original string labels used for contextual CSV
        })
    
    # Save contextual data using original feature names (without descriptions)
    contextual_df_orig = create_contextual_df(X_test, y_test)
    if year == 2017:
        output_csv_path_orig = os.path.join("data", "processed", "CICIDS2017", "original", "X_test_contextual.csv")
    else:
        output_csv_path_orig = os.path.join("data", "processed", "CICDDoS2019", "original", "X_test_contextual.csv")
    
    os.makedirs(os.path.dirname(output_csv_path_orig), exist_ok=True)
    contextual_df_orig.to_csv(output_csv_path_orig, index=False)
    print(f"\nContextual CSV without feature description saved to: {output_csv_path_orig}")
    print(f"  Shape: {contextual_df_orig.shape}, Unique instances: {contextual_df_orig['X'].nunique()}")
    print("\nUnique classes and their instance counts in CSV (without feature description):")
    print(contextual_df_orig["y"].value_counts())
    
    # Save contextual data using feature descriptions.
    feature_descriptions = get_feature_descriptions()
    renamed_columns = {col: feature_descriptions[col] for col in X_test.columns.intersection(feature_descriptions.keys())}
    X_test_with_desc = X_test.rename(columns=renamed_columns)
    
    contextual_df_desc = create_contextual_df(X_test_with_desc, y_test)
    if year == 2017:
        output_csv_path_desc = os.path.join("data", "processed", "CICIDS2017", "description", "X_test_contextual.csv")
    else:
        output_csv_path_desc = os.path.join("data", "processed", "CICDDoS2019", "description", "X_test_contextual.csv")
    
    os.makedirs(os.path.dirname(output_csv_path_desc), exist_ok=True)
    contextual_df_desc.to_csv(output_csv_path_desc, index=False)
    print(f"\nContextual CSV with feature description saved to: {output_csv_path_desc}")
    print(f"  Shape: {contextual_df_desc.shape}, Unique instances: {contextual_df_desc['X'].nunique()}")
    print("\nUnique classes and their instance counts in CSV (with feature description):")
    print(contextual_df_desc["y"].value_counts())
    
    # -------------------------
    # Compute and save exemplar contextual examples for few-shot prompting
    # using the numeric features from X_test.
    num_exemplars_per_class = 5  # Desired exemplars per class

    # # ----- Exempler Examples -----
    # X_test_numeric = X_test.select_dtypes(include=[np.number])
    # exemplars_df = compute_exemplars_context(X_test_numeric, y_test, num_exemplars_per_class)
    # -------------------------------

    # ----- Random Examples -----
    random_examples = X_test.copy()
    random_examples['Label'] = y_test['Label']

    exemplars_df = random_examples.groupby('Label').apply(
        lambda group: group.sample(n=min(num_exemplars_per_class, len(group)), random_state=42)
    ).reset_index(drop=True)
    exemplars_df["rank"] = exemplars_df.groupby("Label").cumcount() + 1
    # -------------------------------

    # -------------------------
    # Exemplar contextual examples using feature descriptions
    exemplars_contextual_with = exemplars_df.merge(contextual_df_desc[['X']], left_index=True, right_index=True, how='left')
    exemplars_contextual_with_final = exemplars_contextual_with[['X', 'Label', 'rank']].rename(columns={'Label': 'y'})
    examples_csv_path_with = os.path.join(os.path.dirname(output_csv_path_desc), "X_test_context_examples.csv")
    exemplars_contextual_with_final.to_csv(examples_csv_path_with, index=False)
    print(f"\nContextual exemplars CSV (with feature description) saved to: {examples_csv_path_with}")
    print("\nExemplar Contextual Data Details (with feature description):")
    print(f"  Exemplar contextual data shape: {exemplars_contextual_with_final.shape}")
    print("\nUnique classes and their exemplar instance counts (with feature description):")
    print(exemplars_contextual_with_final["y"].value_counts())
    
    # Exemplar contextual examples using original feature names (without descriptions)
    exemplars_contextual_without = exemplars_df.merge(contextual_df_orig[['X']], left_index=True, right_index=True, how='left')
    exemplars_contextual_without_final = exemplars_contextual_without[['X', 'Label', 'rank']].rename(columns={'Label': 'y'})
    examples_csv_path_without = os.path.join(os.path.dirname(output_csv_path_orig), "X_test_context_examples.csv")
    exemplars_contextual_without_final.to_csv(examples_csv_path_without, index=False)
    print(f"\nContextual exemplars CSV (without feature description) saved to: {examples_csv_path_without}")
    print("\nExemplar Contextual Data Details (without feature description):")
    print(f"  Exemplar contextual data shape: {exemplars_contextual_without_final.shape}")
    print("\nUnique classes and their exemplar instance counts (without feature description):")
    print(exemplars_contextual_without_final["y"].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Process dataset for ML and LLM contextualization")
    parser.add_argument("-d", "--data", required=True, choices=["2017", "2019"], help="Dataset year: 2017 or 2019")
    args = parser.parse_args()
    
    year = int(args.data)
    process_data(year)

if __name__ == "__main__":
    main()
