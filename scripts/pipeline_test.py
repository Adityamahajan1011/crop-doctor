import json, re, sys, time, os
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from executorch.runtime import Runtime
from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

IMG, CROP = sys.argv[1], sys.argv[2]
names = torch.load("models/mobilenetv3_field_v1.pth", map_location="cpu", weights_only=True)["class_names"]
KB = json.load(open("app/kb.json", encoding="utf-8"))

cnn = Runtime.get().load_program("models/mobilenetv3_field_v1_fp32.pte").load_method("forward")
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
img = ImageOps.exif_transpose(Image.open(IMG)).convert("RGB")
t = time.time()
p = cnn.execute([tf(img).unsqueeze(0)])[0].softmax(1)[0]
print(f"CNN time: {time.time()-t:.3f}s")

idx = [i for i, n in enumerate(names) if n.split("___")[0] == CROP]
mass = float(p[idx].sum())
pf = torch.zeros_like(p); pf[idx] = p[idx]; pf = pf / pf.sum()
best = int(pf.argmax()); key = names[best]
print(f"\nCrop-match {mass:.0%} | prediction: {key} ({float(pf[best])*100:.1f}%)")

if key.endswith("healthy"):
    print("Leaf looks healthy."); sys.exit()
e = KB.get(key)
if not e:
    print("No KB entry for this disease."); sys.exit()
print("\n--- Verified knowledge base ---")
print("Symptoms:", e["symptoms"]); print("Prevention:", "; ".join(e["prevention"]))
print("Management:", e["management"]); print("Source:", e["source"])

BLOCK = ["spray","fungicide","pesticide","chemical","insecticide","dose","dosage","apply","mancozeb",
         "copper","litre","liter","water","irrigat","fertili","nutrient"]
COLORS = ["yellow","black","brown","white","gray","grey","red","reddish","orange","green","tan","purple","pink"]

def clean_body(text):
    lines = [l.strip(" *-\t") for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if not (len(l.split()) <= 5 and not l.endswith("."))]
    return " ".join(re.split(r"(?<=[.!?])\s+", " ".join(lines))[:2])

def ok_sym(text, e):
    t = text.lower(); src = (e["symptoms"] + " " + e["disease"] + " " + e["crop"]).lower()
    if any(x in t for x in ["i can't", "i cannot", "sorry"]): return False
    if set(re.findall(r"\d+", t)) - set(re.findall(r"\d+", src)): return False
    if any(b in t and b not in src for b in BLOCK): return False
    if any(c in t and c not in src for c in COLORS): return False
    return 6 <= len(t.split()) <= 60

tok = AutoTokenizer.from_pretrained("llama_pte/tokenizer")
llm = ExecuTorchModelForCausalLM.from_pretrained("llama_pte")
msgs = [{"role": "system", "content": "You rewrite one plant-disease description in very simple English for a farmer. "
         "Use only the information given. Do not add advice, causes or treatments. Write one or two short sentences."},
        {"role": "user", "content": f"{e['crop']} - {e['disease']}.\nDescription: {e['symptoms']}\n\nRewrite in simple words."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
t = time.time()
out = llm.text_generation(tokenizer=tok, prompt=prompt, max_seq_len=300)
text = clean_body(out.split("assistant")[-1].strip())
print(f"\n--- AI simple explanation ({time.time()-t:.1f}s) ---")
print(text if ok_sym(text, e) else "[guard rejected the AI text, showing verified text only]")