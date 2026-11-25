import os
from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "llama_pte", "tokenizer"))
model = ExecuTorchModelForCausalLM.from_pretrained(os.path.join(ROOT, "llama_pte"))
print(model.text_generation(tokenizer=tok, prompt="Say hello in one short sentence.", max_seq_len=64))