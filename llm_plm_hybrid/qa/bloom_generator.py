# llm_plm_hybrid/qa/bloom_generator.py

from transformers import BloomForCausalLM, BloomTokenizerFast
import torch

class BloomGenerator:
    """
    Wraps a BLOOM causal‐LM for free‐form answer generation.
    """
    def __init__(self,
                 model_name: str = "bigscience/bloom-560m",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BloomTokenizerFast.from_pretrained(model_name)
        self.model = BloomForCausalLM.from_pretrained(model_name).to(self.device)

    def generate(self,
                 prompt: str,
                 max_length: int = 256,
                 **kwargs) -> str:
        """
        Generates a human‐legible response given a markdown‐style prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
