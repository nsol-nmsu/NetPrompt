from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class QwenModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            max_model_len=60000,
            trust_remote_code=True
        )
        
        # Load tokenizer separately for prompt formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Configure default sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=64,
            # temperature=0.0,
            # top_p=1.0,
            # top_k=-1,
            stop=[],
            skip_special_tokens=True
        )

    def generate(self, prompt, max_new_tokens=64):
        self.sampling_params.max_tokens = max_new_tokens
        
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        return outputs[0].outputs[0].text.strip()