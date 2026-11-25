import json, os, platform, sys, time, statistics
import numpy as np, torch, psutil
from PIL import Image
from torchvision import transforms
from executorch.runtime import Runtime

proc = psutil.Process()
def rss(): return round(proc.memory_info().rss / 1e6)
def peak():
    try:
        import resource; return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3)   # Linux, MB
    except Exception: return None

R = {"host": platform.node(), "machine": platform.machine(), "python": sys.version.split()[0],
     "cpus": os.cpu_count(), "total_ram_mb": round(psutil.virtual_memory().total / 1e6)}

# ---- CNN (.pte) ----
img_path = sys.argv[1] if len(sys.argv) > 1 else None
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0) if img_path else torch.randn(1, 3, 224, 224)
t = time.time()
m = Runtime.get().load_program("mobilenetv3_field_v1_fp32.pte").load_method("forward")
R["cnn_load_s"] = round(time.time() - t, 3)
for _ in range(3): m.execute([x])
times = []
for _ in range(30):
    t = time.time(); m.execute([x]); times.append((time.time() - t) * 1000)
times.sort()
R["cnn_ms_mean"] = round(statistics.mean(times), 1); R["cnn_ms_p95"] = round(times[int(0.95 * len(times)) - 1], 1)
R["cnn_file_mb"] = round(os.path.getsize("mobilenetv3_field_v1_fp32.pte") / 1e6, 1)
R["rss_after_cnn_mb"] = rss()
print("CNN:", R["cnn_ms_mean"], "ms mean")

# ---- Llama (.pte) ----
from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer
KB = json.load(open("kb.json", encoding="utf-8"))
e = KB["Corn_(maize)___Common_rust_"]
tok = AutoTokenizer.from_pretrained("llama_pte/tokenizer")
t = time.time(); llm = ExecuTorchModelForCausalLM.from_pretrained("llama_pte"); R["llama_load_s"] = round(time.time() - t, 1)
R["llama_file_mb"] = round(os.path.getsize("llama_pte/model.pte") / 1e6)
msgs = [{"role": "system", "content": "You rewrite one plant-disease description in very simple English for a farmer. Use only the information given. Write one or two short sentences."},
        {"role": "user", "content": f"{e['crop']} - {e['disease']}.\nDescription: {e['symptoms']}\n\nRewrite in simple words."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
runs = []
for _ in range(3):
    t = time.time(); out = llm.text_generation(tokenizer=tok, prompt=prompt, max_seq_len=300); dt = time.time() - t
    n_new = max(1, len(tok(out.split("assistant")[-1]).input_ids))
    runs.append({"total_s": round(dt, 1), "new_tokens": n_new, "overall_tok_per_s": round(n_new / dt, 2)})
R["llama_runs"] = runs
R["rss_after_llama_mb"] = rss(); R["peak_rss_mb"] = peak()

fn = f"benchmark_{platform.node()}.json"
json.dump(R, open(fn, "w"), indent=1)
print(json.dumps(R, indent=1)); print("saved", fn)