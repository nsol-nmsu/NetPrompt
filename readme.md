<h1 align="center"> NetPrompt: Evaluation of LLMs as Network Intrusion Detection System </h1>

<p align="center">
Pratyay Kumar, Abu Saleh Md Tayeen, Qixu Gong, Jiefei Liu, Satyajayant Misra,<br>
Huiping Cao, Jayashree Harikumar
</p>

<p align="center">
<strong>MILCOM 2025</strong> (Accepted and Presented)
</p>

<h2> Guide to use LLMs as NIDS </h2>

Welcome to the NetPrompt repository for Evaluating Large Language Models (LLMs) as Network Intrusion Detection System (NIDS). This repository contains the [Processed Datasets](./data/processed) and [Prompts](./prompts): Zer-Shot (ZS), Few-Shot-*p* (FS) where *p* ∈ {1, 2, 3}, and Chain-of-Thought (CoT).

## Repository Structure:

- [data](./data): Processed version of CICIDS2017 and CICDDoS2019 datasets.
- [llm](./llm): Wrapper classes for the LLM models used in our experiments.
- [prompt](./prompts): All prompt templates used for evaluation, including ZS, FS, and CoT formats.
- [scripts](./scripts): Shell scripts to reproduce all experiments.
- [models](./models): Pretrained baseline MLP models used for comparative evaluation.
- [tables](./tables): Contains additional results with **Precision** and **Recall** (not included in the main body of the paper).

## Getting Started:

### Installation

1. Create and activate conda environment:
   ```bash
   conda create -n netprompt python=3.9.21
   conda activate netprompt
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - **HuggingFace Token** (for LLaMA): Edit `llm/llama.py` at line 9, replace `"******"` with your token
   - **Gemini API Key**: Edit `main.py` at line 127, replace `"*****"` with your API key

### Quick Start

To reproduce results from Table V:
```bash
cd scripts
./llama.sh
./qwen.sh
./gemini.sh
```

For individual experiments:
- Run `main.py` for Zero-Shot (ZS) and Few-Shot (FS) experiments
- Run `main_cot.py` for Chain-of-Thought (CoT) experiments

## Usage:
General format for Zero-Shot & Few-Shot prompting:
```
python main.py -m <model_name> -d <data_year> -p <prompt_type> [-e <fewshot_example_number>]
```

- `-m`/`--model_name`: `gemini`, `llama` or `qwen`
- `-d`/`--data_year`: `2017` or `2019`
- `-p`/`--prompt_type`: `zeroshot` or `fewshot`
- `-e`/`--num_examples`: Requires for `fewshot` prompting only

General format for Chain-of-Thought prompting:
```
python main_cot.py --help
usage: main_cot.py [-h] [--model_name {qwen,llama}] [--data_year {2017,2019}] [--prompt_name PROMPT_NAME] [--attack_desc_enable] [--num_examples NUM_EXAMPLES] [--feature_type FEATURE_TYPE]
```

- `-h`/`--help`: show help message and exit
- `--model_name`: Model to use - `llama` or `qwen`
- `-d`/`--data_year`: `2017` or `2019`
- `--prompt_name`: PROMPT_NAME - Name of the prompt strategy to use
- `--attack_desc_enable`: Enable attack descriptions in prompt
- `--num_examples`: NUM_EXAMPLES - Number of examples to include in prompt
- `feature_type`: FEATURE_TYPE

To train MLP model:
`python3 train_model.py -m {2017 or 2019}`

## Extend NetPrompt

To add a new LLM:
- Define a new class under `llm/`
- Add the model in `main.py`

To add new prompts:
- Add your prompt under `prompt/`
- Reference it in `main.py` or `main_cot.py`

## Citation

This work has been accepted and presented at MILCOM 2025. The BibTeX citation will be added here once the proceedings are published.
