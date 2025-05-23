<h1 align="center"> NetPrompt </h1>

<h2> Guide to use LLMs as NIDS </h2>

Welcome to the NetPrompt repository for Evaluating LLMs as Network Intrusion Detection System. This repository contains the [Processed Datasets](./data/processed), [Prompts](./prompts): Zer-Shot (ZS), Few-Shot-*p* (FS) where *p* ∈ {1, 2, 3}, and Chain-of-Thought (CoT).

## Repository Structure:

- [data](./data): Processed version of CICIDS2017 and CICDDoS2019 datasets.
- [llm](./llm): Wrapper classes for the LLM models used in our experiments.
- [prompt](./prompt): All prompt templates used for evaluation, including ZS, FS, and CoT formats.
- [scripts](./scripts): Shell scripts to reproduce all experiments.
- [models](./models): Pretrained baseline MLP models used for comparative evaluation.

## Getting Started:
To reproduce our results:

- Install dependencies listed in `requirements.txt`
- Add your HuggingFace and Gemini API keys.
- Run `main.py` for Zero-Shot (ZS) and Few-Shot (FS) experiments.
- Run `main_cot.py` for Chain-of-Thought (CoT) experiments

## Usage:
General format for Zero-Shot & Few-Shot prompting:
`python main.py -m <model_name> -d <data_year> -p <prompt_type> [-e <fewshot_example_number>]`

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

## Extend Netprompt

To add a new LLM:
- Define a new class under `llm/`
- Add the model in `main.py`

To add new prompts:
- Add your prompt under `prompt/`
- Reference it in `main.py` or `main_cot.py`

## Citation

This work is currently under review. A BibTeX citation entry will be provided upon acceptance.