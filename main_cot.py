import re
import os
import pandas as pd 
import prompts.cot_prompt as cot_prompt 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from datetime import datetime
import csv
import sys 
import gc
import torch
import platform

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run LLM evaluation on network traffic data')
    
    parser.add_argument('--model_name', type=str, default="qwen",
                       choices=["qwen", "llama"],
                       help='Model to use (qwen or llama)')
    parser.add_argument('--data_year', type=int, default=2017,
                       choices=[2017, 2019],
                       help='Dataset year (2017 or 2019)')
    parser.add_argument('--prompt_name', type=str, default="cot",
                       help='Name of the prompt strategy to use')
    parser.add_argument('--attack_desc_enable', action='store_true',
                       help='Enable attack descriptions in prompt')
    parser.add_argument('--num_examples', type=int, default=1,
                       help='Number of examples to include in prompt')
    parser.add_argument('--feature_type', type=str, default="description")
    
    return parser.parse_args()


hostname = platform.node()
if hostname in ["auco", "qgong-pc"]:
    base_folder = "/data/qgong"
    if hostname == 'auco':
        gpu_memory_utilization = 0.5
    else:
        gpu_memory_utilization = 0.5
elif hostname == "epscor23" :
    base_folder = "/home/epscor23nobackup/qgong"
    gpu_memory_utilization = 0.3
elif hostname=="simurgh":
    base_folder = "/home/epscor23nobackup/qgong"
    gpu_memory_utilization = 0.3


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class LLMModel:
    def __init__(self, model_name):
        if model_name=="qwen":
            self.model_name = f"{base_folder}/models/Qwen2.5-7B-Instruct-1M"
        elif model_name == "llama":
            self.model_name = f"{base_folder}/models/llama-31-8B-I"

        
        # Initialize vLLM engine
        self.llm = LLM(model=self.model_name,
            tensor_parallel_size=1,
            max_model_len=40000,
            enable_chunked_prefill=True,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        # Load tokenizer separately for prompt formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        
        # Configure default sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=2048,
            skip_special_tokens=True,
            top_p=0.9,
            temperature=0.7
        )

    def generate(self, prompt):
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        for output in outputs:
            generated_text = output.outputs[0].text
        
        return generated_text.strip()

def extract_prediction(text):
    # Regex pattern to match content between "@@@@ Prediction: " and " @@@@"
    pattern = r'@+\s*Prediction:\s*(.*?)\s*@+'
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None
    
def load_data(year,feature_type):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if year == 2019:
        filepath = os.path.join(base_dir, "data", "processed", "CICDDoS2019", feature_type, "X_test_contextual.csv")
    elif year == 2017:
        filepath = os.path.join(base_dir, "data", "processed", "CICIDS2017", feature_type, "X_test_contextual.csv")
    else:
        raise ValueError("Year must be 2017 or 2019.")
    print(f"Read data from {filepath}")
    return pd.read_csv(filepath)

def main():
    args = parse_arguments()

    model_name = args.model_name
    data_year = args.data_year
    prompt_name = args.prompt_name
    attack_desc_enable = args.attack_desc_enable 
    num_examples = args.num_examples
    feature_type = args.feature_type
    print(args)


    print(f"Running {model_name} with {prompt_name} prompt on {data_year} dataset.")
    df = load_data(data_year, feature_type)
    print(df.head())
    print("/data/qgong/code/Milcom2025/prompts/cot_prompt_2017_exp1.txt")
    if data_year == 2019:
        example_path = os.path.join("data", "processed", "CICDDoS2019", feature_type, "X_test_context_examples.csv")
    else:
        example_path = os.path.join("data", "processed", "CICIDS2017", feature_type, "X_test_context_examples.csv")
    print(example_path)
    if data_year == 2017:
        valid_labels = {"BENIGN", "BOT", "DDOS", "DOS GOLDENEYE", "DOS HULK", 
                        "DOS SLOWHTTPTEST", "DOS SLOWLORIS", "FTP-PATATOR", 
                        "HEARTBLEED", "INFILTRATION", "PORTSCAN", "SSH-PATATOR"}
    else:
        valid_labels = {"BENIGN", "UDP-LAG", "DRDOS_SSDP", "DRDOS_DNS", "DRDOS_MSSQL", 
                        "DRDOS_NETBIOS", "DRDOS_LDAP", "DRDOS_NTP", "DRDOS_UDP", 
                        "SYN", "DRDOS_SNMP"}

    model = LLMModel(model_name)

    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y:%m:%d_%H%M%S")
    date = datetime.now().strftime("%Y-%m-%d")

    example_str=""
    ranks = []

    if num_examples == 1:
            ranks = [1]
    elif num_examples == 2:
            ranks = [1, 2]
    elif num_examples == 3:
            ranks = [1, 2, 3]



    if len(ranks)!=0:
        example_df = pd.read_csv(example_path)
        filtered_examples = example_df[example_df["rank"].isin(ranks)]
        examples = [
            f"Label: {row['y']}; Network flow features: {row['X']}.\n"
            for _, row in filtered_examples.iterrows()
        ]
        example_str += "\n\nHere are examples for each label:\n" + "\n".join(examples)


    if num_examples is None or num_examples==0:
        output_filename = os.path.join(output_folder, "generated", f"{prompt_name}_{model_name}_{data_year}_{timestamp}_{date}.csv")
        metrics_filename = os.path.join(output_folder, "metrics", f"{prompt_name}_{model_name}_{data_year}_{timestamp}_{date}_attack_nofeature_description.txt")
    else:
        output_filename = os.path.join(output_folder, "generated", f"{prompt_name}_{num_examples}_{model_name}_{data_year}_{timestamp}_{date}.csv")
        metrics_filename = os.path.join(output_folder, "metrics", f"{prompt_name}_{num_examples}_{model_name}_{data_year}_metrics_{timestamp}_{date}_attack_nofeature_description.txt")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    print("output_file : ", output_filename)
    print("metrics_file : ", metrics_filename)

    
    
    # df = df.sample(n=10)
    total_instances = 0
    invalid_prediction_count = 0  # Count of instances where prediction is not in valid set.
    filtered_true = []
    filtered_pred = []
    total_input_tokens = 0
    total_output_tokens = 0

    print(df.shape)

    if data_year == 2017:
        att_desc = cot_prompt.attack_desc_2017
        output = cot_prompt.output_2017
    elif data_year == 2019:
        att_desc = cot_prompt.attack_desc_2019
        output = cot_prompt.output_2019

    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Generated Text", "Predicted Label", "True Label", "Input Prompt Tokens", "Generated Text Tokens"])
        
        start_time = time.time()
        
        for idx, row in df.iterrows():        
            total_instances += 1

            prompt_str=cot_prompt.sys_inst+"\n"

            if attack_desc_enable == True:
                prompt_str += "Here are the types of attacks you may encounter, along with their characteristics:\n"+att_desc+"\n\n"

            if num_examples!=0 and num_examples!=None:
                prompt_str += "Here are examples you can refer:\n"+example_str+"\n"
            
            x_data = row["X"]
            true_label = str(row["y"]).strip().upper()
            # print(x_data)
            # print(true_label)

            user_str = "Here is one network traffic:"+x_data+"\n"

            user_str += output+"\n"

            messages = [
                        # {"role": "system", "content": ""},
                        {"role": "user", "content": prompt_str+user_str}
                    ]
            prompt = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)
            
            # print(prompt)
            tokenized_prompt = model.tokenizer(prompt, return_tensors="pt")
            num_prompt_tokens = tokenized_prompt.input_ids.shape[1]
            total_input_tokens += num_prompt_tokens

            g_time_start = time.time()
            generated_text = model.generate(prompt)
            tokenized_generated = model.tokenizer(generated_text, return_tensors="pt")
            num_generated_tokens = tokenized_generated.input_ids.shape[1]
            total_output_tokens += num_generated_tokens

            # print(prompt)
            # print(generated_text)
            # print("prediction: ", extract_prediction(generated_text), ", True Label: ", true_label)
            predicted_label = extract_prediction(generated_text)
            # print("===="*20)
            if predicted_label not in valid_labels:
                invalid_prediction_count += 1
            else:
                filtered_true.append(true_label)
                filtered_pred.append(predicted_label)
            print(f"{total_instances}, ID: {idx} | Generated: {predicted_label} | True: {true_label}, {time.time()- g_time_start}")
            print("=" * 100)

            writer.writerow([idx, generated_text, predicted_label, true_label, num_prompt_tokens, num_generated_tokens])
            del x_data, prompt, generated_text, true_label
            gc.collect()
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"\nTotal time taken for {total_instances} instances: {total_time:.2f} seconds")


    instruction_error_rate = invalid_prediction_count / total_instances if total_instances > 0 else 0

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
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print("="*40)
    print(f"model_name: {model_name}")
    print(f"data_year: {data_year}")
    print(f"prompt_name: {prompt_name}")
    print(f"attack_desc_enable: {attack_desc_enable}")
    print(f"num_examples: {num_examples}")

    with open(metrics_filename, "w") as f:
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
        f.write(f"\nTotal Input Tokens: {total_input_tokens}\n")
        f.write(f"Total Output Tokens: {total_output_tokens}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"data_year: {data_year}\n")
        f.write(f"prompt_name: {prompt_name}\n")
        f.write(f"attack_desc_enable: {attack_desc_enable}\n")
        f.write(f"num_examples: {num_examples}\n")
        f.write(f"feature_type: {feature_type}\n")


if __name__ == "__main__":
    main()

