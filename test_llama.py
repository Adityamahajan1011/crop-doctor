from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("llama_pte/tokenizer")
model = ExecuTorchModelForCausalLM.from_pretrained("llama_pte")
print(model.text_generation(tokenizer=tok, prompt="Say hello in one short sentence.", max_seq_len=64))