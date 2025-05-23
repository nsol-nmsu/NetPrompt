import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login

class LLaMAModel:
    def __init__(self):
        login(token="******") # Replace with your token
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1, 
            # max_model_len=8192, 
            trust_remote_code=True 
        )
        
        # Load tokenizer separately for prompt formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Configure default sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=256, 
            # temperature=0.6, 
            # top_p=0.9, 
            # top_k=-1, 
            stop=[],
            skip_special_tokens=True
        )

    def generate(self, prompt, max_new_tokens=256):
        self.sampling_params.max_tokens = max_new_tokens
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        return outputs[0].outputs[0].text.strip()