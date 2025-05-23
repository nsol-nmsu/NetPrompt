import os
import gc
import torch
import csv
import argparse
import pandas as pd
import time
from datetime import datetime
from llm.mistral import MistralModel
from llm.gpt_neo import GPTNeoModel
from llm.llama import LLaMAModel
from llm.deepseek import DeepSeekModel
from llm.qwen import QwenModel
from llm.gemini import GeminiModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_data(year):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if year == 2019:
        filepath = os.path.join(base_dir, "data", "processed", "CICDDoS2019", "original", "X_test_contextual_EXEMPLER.csv")
    elif year == 2017:
        filepath = os.path.join(base_dir, "data", "processed", "CICIDS2017", "original", "X_test_contextual_EXEMPLER.csv")
    else:
        raise ValueError("Year must be 2017 or 2019.")
    return pd.read_csv(filepath)

def main(model_name, prompt_name, data_year, example=None):
    print(f"Running {model_name} with {prompt_name} prompt on {data_year} dataset.")
    df = load_data(data_year)
    prompt_path = os.path.join(os.getcwd(), "prompts", f"prompt_{data_year}_exp1.txt")
    with open(prompt_path, "r", encoding="latin1") as file:
        prompt_df = file.read()

    if data_year == 2019:
        example_path = os.path.join("data", "processed", "CICDDoS2019", "original", "X_test_context_examples_EXEMPLER.csv")
    else:
        example_path = os.path.join("data", "processed", "CICIDS2017", "original", "X_test_context_examples_EXEMPLER.csv")
    
    if prompt_name == "zeroshot":
        print("Running in zeroshot mode.")
        ranks = []
    elif prompt_name in ["fewshot", "chain-of-thought"]:
        print(f"Using {example} examples for few-shot prompting.")
        if example == 1:
            ranks = [1]
        elif example == 2:
            ranks = [1, 2]
        elif example == 3:
            ranks = [1, 2, 3]

    if ranks:
        example_df = pd.read_csv(example_path)
        filtered_examples = example_df[example_df["rank"].isin(ranks)]
        examples = [
            f"Label: {row['y']}; Network flow features: {row['X']}.\n"
            for _, row in filtered_examples.iterrows()
        ]
        prompt_df += "\n\nHere are examples for each label:\n" + "\n".join(examples)
    
    # Define the valid label set based on dataset year.
    if data_year == 2017:
        valid_labels = {"BENIGN", "BOT", "DDOS", "DOS GOLDENEYE", "DOS HULK", 
                        "DOS SLOWHTTPTEST", "DOS SLOWLORIS", "FTP-PATATOR", 
                        "HEARTBLEED", "INFILTRATION", "PORTSCAN", "SSH-PATATOR"}
    else:
        valid_labels = {"BENIGN", "UDP-LAG", "DRDOS_SSDP", "DRDOS_DNS", "DRDOS_MSSQL", 
                        "DRDOS_NETBIOS", "DRDOS_LDAP", "DRDOS_NTP", "DRDOS_UDP", 
                        "SYN", "DRDOS_SNMP"}
    
    # Initialize the model based on the provided name.
    if model_name == "mistral":
        model = MistralModel()
    elif model_name == "llama":
        model = LLaMAModel()
    elif model_name in ["deepseek", "deepseek-ai/DeepSeek-V3"]:
        model = DeepSeekModel()
    elif model_name == "qwen":
        model = QwenModel()
    elif model_name == "gpt3":
        model = GPTNeoModel()
    elif model_name == "gemini":
        model = GeminiModel(api_key="*****") # API key for Gemini model
    else:
        raise ValueError("Unsupported model. Choose from: mistral, gpt3, llama, deepseek, qwen")

    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y:%m:%d_%H%M%S")
    date = datetime.now().strftime("%Y-%m-%d")

    if example is None:
        output_filename = os.path.join(output_folder, "generated", f"{prompt_name}_{model_name}_{data_year}_{timestamp}_{date}.csv")
        metrics_filename = os.path.join(output_folder, "metrics", f"{prompt_name}_{model_name}_{data_year}_{timestamp}_{date}_attack_nofeature_description.txt")
    else:
        output_filename = os.path.join(output_folder, "generated", f"{prompt_name}_{example}_{model_name}_{data_year}_{timestamp}_{date}.csv")
        metrics_filename = os.path.join(output_folder, "metrics", f"{prompt_name}_{example}_{model_name}_{data_year}_metrics_{timestamp}_{date}_attack_nofeature_description.txt")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    total_instances = 0
    invalid_prediction_count = 0  # Count of instances where prediction is not in valid set.
    # For classification metrics, we use only instances with valid predictions.
    filtered_true = []
    filtered_pred = []

    total_input_tokens = 0
    total_output_tokens = 0

    num_prompt_tokens = 0
    num_generated_tokens = 0

    total_inference_time = 0 

    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Generated Text", "Predicted Label", "True Label", "Input Prompt Tokens", "Generated Text Tokens", "Inference Time (s)"])
        
        start_time = time.time()
        
        for idx, row in df.iterrows():
            total_instances += 1

            x_data = row["X"]
            true_label = str(row["y"]).strip().upper()

            # Prompt generation
            if model_name == "mistral":
                prompt = f"<s>[INST] <<SYS>>\n{prompt_df}\n<</SYS>>\n\nNetwork Traffic: {x_data}[/INST]"
            elif model_name in ["qwen", "deepseek", "gpt3", "llama"]:
                messages = [
                    {"role": "system", "content": prompt_df},
                    {"role": "user", "content": f"Here is the network flow features: {x_data}. Please classify it."}
                ]
                prompt = model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif model_name == "gemini":
                prompt = f"{prompt_df}\nHere is the network flow features: {x_data}. Please classify it."
                
            if model_name in ["mistral", "qwen", "deepseek", "gpt3", "llama"]:
                tokenized_prompt = model.tokenizer(prompt, return_tensors="pt")
                num_prompt_tokens = tokenized_prompt.input_ids.shape[1]
                total_input_tokens += num_prompt_tokens

                start_infer = time.time()
                generated_text = model.generate(prompt, max_new_tokens=256)
                end_infer = time.time()
                inference_time = end_infer - start_infer
                total_inference_time += inference_time

                tokenized_generated = model.tokenizer(generated_text, return_tensors="pt")
                num_generated_tokens = tokenized_generated.input_ids.shape[1]
                total_output_tokens += num_generated_tokens
            else:
                start_infer = time.time()
                generated_text = model.generate(prompt, max_new_tokens=256)
                end_infer = time.time()
                inference_time = end_infer - start_infer
                total_inference_time += inference_time

            predicted_label = generated_text.strip().upper()
            
            if predicted_label not in valid_labels:
                invalid_prediction_count += 1
            else:
                filtered_true.append(true_label)
                filtered_pred.append(predicted_label)
            
            print(f"ID: {idx} | Generated: {predicted_label} | True: {true_label} | Inference Time: {inference_time:.4f} s")
            print("=" * 100)

            writer.writerow([idx, generated_text, predicted_label, true_label, num_prompt_tokens, num_generated_tokens, round(inference_time, 4)])

            del x_data, prompt, generated_text, true_label
            gc.collect()
            torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        avg_inference_time = total_inference_time / total_instances if total_instances > 0 else 0
        print(f"\nTotal inference time for all instances: {total_inference_time:.2f} seconds")
        print(f"Average Inference Time per Sample: {avg_inference_time:.4f} seconds")
        print(f"Total runtime (including all overhead): {total_time:.2f} seconds")

    # Compute instruction-following error rate.
    instruction_error_rate = invalid_prediction_count / total_instances if total_instances > 0 else 0

    avg_inference_time = total_inference_time / total_instances if total_instances > 0 else 0
    print(f"Average Inference Time per Sample: {avg_inference_time:.4f} seconds")

    # Compute classification metrics only on instances with valid predictions.
    valid_instance_count = len(filtered_true)
    if valid_instance_count > 0:
        classification_accuracy = accuracy_score(filtered_true, filtered_pred)
        classification_precision = precision_score(filtered_true, filtered_pred, average='macro', zero_division=0)
        classification_recall = recall_score(filtered_true, filtered_pred, average='macro', zero_division=0)
        classification_f1 = f1_score(filtered_true, filtered_pred, average='macro', zero_division=0)
        report = classification_report(filtered_true, filtered_pred, zero_division=0)
    else:
        classification_accuracy = classification_precision = classification_recall = classification_f1 = 0
        report = "No valid predictions to report."
    
    print("\n" + "="*40)
    print(" Evaluation Metrics ".center(40, "="))
    print("="*40)
    
    print(f"\nTotal Instances: {total_instances}")
    print(f"Invalid Prediction Count (Instruction Errors): {invalid_prediction_count}")
    print(f"Instruction Error Rate: {instruction_error_rate:.4f}")
    print(f"Instances with Valid Predictions: {valid_instance_count}")
    print(f"Classification Accuracy (Valid Predictions): {classification_accuracy:.4f}")
    print(f"Classification Precision (Valid Predictions): {classification_precision:.4f}")
    print(f"Classification Recall (Valid Predictions): {classification_recall:.4f}")
    print(f"Classification F1 Score (Valid Predictions): {classification_f1:.4f}")
    print("\nClassification Report:\n", report)

    if model_name in ["mistral", "qwen", "deepseek", "gpt3", "llama"]:
        print(f"Total Input Tokens: {total_input_tokens}")
        print(f"Total Output Tokens: {total_output_tokens}")

    with open(metrics_filename, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompt Template: {prompt_name}\n")
        f.write(f"Dataset Year: {data_year}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Date: {date}\n")
        f.write("\n" + "="*40 + "\n")
        f.write(f"Total Instances: {total_instances}\n")
        f.write(f"Invalid Prediction Count (Instruction Errors): {invalid_prediction_count}\n")
        f.write(f"Instruction Error Rate: {instruction_error_rate:.4f}\n")
        f.write(f"Instances with Valid Predictions: {valid_instance_count}\n")
        f.write(f"Classification Accuracy (Valid Predictions): {classification_accuracy:.4f}\n")
        f.write(f"Classification Precision (Valid Predictions): {classification_precision:.4f}\n")
        f.write(f"Classification Recall (Valid Predictions): {classification_recall:.4f}\n")
        f.write(f"Classification F1 Score (Valid Predictions): {classification_f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write(f"Average Inference Time per Sample: {avg_inference_time:.4f} seconds\n")
        f.write(f"Total Inference Time for All Samples: {total_inference_time:.2f} seconds\n")
        f.write(f"Total Runtime (Including All Overhead): {total_time:.2f} seconds\n")

        if model_name in ["mistral", "qwen", "deepseek", "gpt3", "llama"]:
            f.write(f"\nTotal Input Tokens: {total_input_tokens}\n")
            f.write(f"Total Output Tokens: {total_output_tokens}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run network traffic classification with selected model, prompt template, and dataset year.")
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        choices=["mistral", "gpt3", "llama", "deepseek", "qwen", "gemini"],
        help="Name of the model to use (mistral, gpt3, llama, deepseek, qwen)"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        required=False,
        default="zeroshot",
        choices=["zeroshot", "fewshot", "chain-of-thought"],
        help="Prompt template to use: zeroshot, fewshot, chain-of-thought"
    )
    parser.add_argument(
        "-d", "--data",
        type=int,
        required=True,
        choices=[2017, 2019],
        help="Dataset year: 2017 or 2019"
    )
    parser.add_argument(
        "-e", "--examples",
        type=int,
        required=False,
        choices=[1, 2, 3],
        help="Number of examples to use for few-shot prompting (1, 2, or 3)"
    )
    args = parser.parse_args()

    if (args.prompt in ["fewshot", "chain-of-thought"]) and args.examples is None:
        parser.error("The --examples argument is required when using fewshot prompt.")

    main(args.model, args.prompt, args.data, args.examples)
