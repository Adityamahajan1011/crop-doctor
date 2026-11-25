import os, re, json, time, hashlib, html, platform
import numpy as np
import torch, torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageOps
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
REG = json.load(open(os.path.join(HERE, "registry.json"), encoding="utf-8"))
KB = json.load(open(os.path.join(HERE, "kb.json"), encoding="utf-8"))
CROPS = REG["crops"]
MODELS = {m["id"]: m for m in REG["models"]}
CROP_TO_MODEL = {}
for m in REG["models"]:
    for c in m["enabled_crops"]:
        CROP_TO_MODEL.setdefault(c, m["id"])
BLUR_MIN, DARK_MIN = 25.0, 45.0

CONFUSABLE = [
    {"Potato___Early_blight", "Potato___Late_blight"},
    {"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Northern_Leaf_Blight"},
    {"Tomato___Early_blight", "Tomato___Late_blight"},
    {"Tomato___Early_blight", "Tomato___Septoria_leaf_spot", "Tomato___Target_Spot", "Tomato___Bacterial_spot"},
]
MARGIN = 0.25     # if the top two are within 25 points and confusable, don't pick one

def confusable_with(top, min_pair=0.6):
    """Top two answers are a known look-alike pair, close in score, and together confident."""
    (n1, p1), (n2, p2) = top[0], top[1]
    for g in CONFUSABLE:
        if n1 in g and n2 in g and (p1 - p2) < MARGIN and (p1 + p2) >= min_pair:
            return n2
    return None

def make_tf(size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(list(mean), list(std))])

# ---------------- vision backends ----------------
class TorchMNV3:
    def __init__(self, cfg):
        ck = torch.load(os.path.join(HERE, cfg["path"]), map_location="cpu", weights_only=True)
        self.class_names = ck["class_names"]
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, ck["num_classes"])
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        self.m, self.tf = m, make_tf()
    def probs(self, img):
        with torch.no_grad():
            return self.m(self.tf(img).unsqueeze(0)).softmax(1)[0].numpy()

class ExecuTorchCNN:
    def __init__(self, cfg):
        from executorch.runtime import Runtime
        ck = torch.load(os.path.join(HERE, cfg["classes_from"]), map_location="cpu", weights_only=True)
        self.class_names = ck["class_names"]
        self.method = Runtime.get().load_program(os.path.join(HERE, cfg["path"])).load_method("forward")
        self.tf = make_tf()
    def probs(self, img):
        out = self.method.execute([self.tf(img).unsqueeze(0)])[0]
        return out.softmax(1)[0].numpy()

BACKENDS = {"torch_mnv3": TorchMNV3, "executorch_pte": ExecuTorchCNN}

@st.cache_resource
def load_backend(model_id):
    b = BACKENDS[MODELS[model_id]["backend"]](MODELS[model_id])
    crop_idx = {}
    for i, n in enumerate(b.class_names):
        crop_idx.setdefault(n.split("___")[0], []).append(i)
    return b, crop_idx

def predict(img, crop):
    mid = CROP_TO_MODEL[crop]
    b, crop_idx = load_backend(mid)
    t = time.time(); p = b.probs(img); secs = time.time() - t
    mass = float(p[crop_idx[crop]].sum())
    mask = np.zeros_like(p); mask[crop_idx[crop]] = 1
    pf = p * mask; pf = pf / pf.sum()
    top = [(b.class_names[i], float(pf[i])) for i in np.argsort(pf)[::-1][:3]]
    raw = [(b.class_names[i], float(p[i])) for i in np.argsort(p)[::-1][:3]]
    return {"model": mid, "backend": MODELS[mid]["backend"], "top": top, "raw": raw,
            "crop_mass": mass, "secs": secs}

# ---------------- Llama (ExecuTorch), loaded only when the button is pressed ----------------
@st.cache_resource
def load_llm():
    from optimum.executorch import ExecuTorchModelForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "llama_pte", "tokenizer"))
    llm = ExecuTorchModelForCausalLM.from_pretrained(os.path.join(HERE, "llama_pte"))
    return llm, tok

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

def ai_explain(e):
    llm, tok = load_llm()
    msgs = [{"role": "system", "content": "You rewrite one plant-disease description in very simple English "
             "for a farmer. Use only the information given. Do not add advice, causes or treatments. "
             "Write one or two short sentences."},
            {"role": "user", "content": f"{e['crop']} - {e['disease']}.\nDescription: {e['symptoms']}\n\nRewrite in simple words."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    t = time.time()
    out = llm.text_generation(tokenizer=tok, prompt=prompt, max_seq_len=300)
    text = clean_body(out.split("assistant")[-1].strip())
    return (text if ok_sym(text, e) else None), time.time() - t

# ---------------- helpers ----------------
def photo_quality(img):
    g = np.asarray(img.resize((384, 384)).convert("L"), dtype=np.float32)
    lap = 4*g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var()), float(g.mean())

def severity(img):
    hsv = np.asarray(img.resize((256, 256)).convert("HSV")).astype(np.int32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    leaf = (h >= 14) & (h <= 135) & (s >= 40) & (v >= 20)
    green = (h >= 50) & (h <= 128) & (s >= 40) & (v >= 30)
    pct = 100 * (leaf & ~green).sum() / max(leaf.sum(), 1)
    return pct, ("mild" if pct < 10 else "moderate" if pct < 30 else "severe")

def pretty(cls): return cls.split("___")[1].replace("_", " ").strip()

# =====================================================================
#  UI  (styling only: the model, knowledge-base and safety logic are above)
# =====================================================================
E = html.escape

CSS = """
<style>
.stApp { background:#f6faf7; }
.block-container { max-width:760px; padding-top:1rem; padding-bottom:3rem; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
header[data-testid="stHeader"] { background:transparent; height:0; }
html, body, [class*="css"] { font-family:"Segoe UI","Noto Sans","Noto Sans Devanagari","Noto Sans Gurmukhi",system-ui,-apple-system,sans-serif; }
.hero { background:linear-gradient(135deg,#1f7a4d,#2fa36b); color:#fff; padding:20px 22px; border-radius:18px; margin-bottom:12px; }
.hero .ht { font-size:1.7rem; font-weight:800; line-height:1.2; }
.hero .hs { margin-top:4px; opacity:.93; font-size:1rem; }
.hero .pill { display:inline-block; background:rgba(255,255,255,.2); border-radius:999px; padding:3px 11px; font-size:.78rem; margin:10px 6px 0 0; }
.stepper { display:flex; gap:8px; margin:8px 0 14px; }
.step { flex:1; text-align:center; padding:8px 4px; border-radius:12px; background:#e8f3ec; color:#5b7a69; font-size:.92rem; font-weight:600; }
.step .dot { display:inline-block; width:22px; height:22px; line-height:22px; border-radius:50%; background:#cfe3d6; color:#1b2a22; margin-right:6px; font-size:.8rem; }
.step.now { background:#1f7a4d; color:#fff; } .step.now .dot { background:#fff; color:#1f7a4d; }
.step.done { background:#d6ecde; color:#1f7a4d; } .step.done .dot { background:#1f7a4d; color:#fff; }
.sect { font-size:1.15rem; font-weight:700; color:#1b2a22; margin:18px 0 6px; }
.sub { color:#5b7a69; font-size:.92rem; margin:2px 0 6px; }
.tips { background:#fff8e6; border:1px solid #f0dca0; border-radius:14px; padding:10px 14px; color:#5c4a12; font-size:.95rem; margin:6px 0 10px; }
.tips span { display:inline-block; margin-right:16px; }
.banner { display:flex; gap:14px; align-items:center; padding:16px 18px; border-radius:16px; margin:10px 0 6px; }
.banner .ico { font-size:2rem; line-height:1; }
.banner .bt { font-size:1.3rem; font-weight:700; line-height:1.25; }
.banner .bs { font-size:.95rem; opacity:.9; margin-top:3px; }
.banner.ok { background:#e6f6ec; color:#14532d; border:1px solid #b7e2c6; }
.banner.warn { background:#fff4dc; color:#6b4a00; border:1px solid #f1d899; }
.chip { display:inline-block; background:#eef2ee; color:#35523f; border-radius:999px; padding:2px 10px; font-size:.8rem; font-weight:600; margin:2px 0 8px; }
.card { background:#fff; border:1px solid #dfe9e2; border-radius:14px; padding:12px 16px; margin:10px 0; box-shadow:0 1px 2px rgba(0,0,0,.04); color:#1b2a22; }
.card .ct { font-weight:700; color:#1f7a4d; margin-bottom:4px; }
.card p, .card li { font-size:1.02rem; line-height:1.5; margin:0; }
.card ul { margin:0; padding-left:20px; }
.note { background:#eef3fb; border:1px solid #cbd9f0; color:#27406b; border-radius:12px; padding:8px 12px; margin:8px 0; font-size:.9rem; }
.src { color:#6a7a70; font-size:.82rem; margin:8px 0; word-break:break-word; }
.src a { color:#2f6f9f; }
.confirm { background:#fdecea; border:1px solid #f5c2bd; color:#7a1c14; border-radius:12px; padding:10px 14px; margin:12px 0; font-size:.95rem; }
.ai { background:#f1ecfb; border:1px solid #d6c9f2; color:#3b2a6b; border-radius:14px; padding:12px 16px; margin:10px 0; }
.ai p { margin:6px 0 4px; font-size:1.02rem; line-height:1.5; }
.ai .aisub { font-size:.8rem; opacity:.8; }
.foot { text-align:center; color:#7a8a80; font-size:.8rem; margin-top:26px; line-height:1.5; }
.stButton > button { border-radius:12px; min-height:3rem; font-size:1rem; border:1px solid #cfe3d6; background:#fff; color:#1b2a22; }
.stButton > button:hover { border-color:#1f7a4d; color:#1f7a4d; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] { background:#1f7a4d; color:#fff; border-color:#1f7a4d; }
[data-testid="stImage"] img { border-radius:14px; }
[data-testid="stExpander"] { border-radius:12px; background:#fff; }
</style>
"""

UI = {
 "en": {"title": "Crop Doctor", "subtitle": "Spot crop leaf diseases and get advice, on your device",
        "pills": ["Works offline", "English · हिन्दी · ਪੰਜਾਬੀ", "Runs on Arm"],
        "st_photo": "Photo", "st_crop": "Crop", "st_result": "Result",
        "step1": "1. Take or upload a leaf photo", "step2": "2. Which crop is this?", "result": "Result",
        "src_upload": "Upload photo", "src_camera": "Use camera", "choose": "Choose a photo",
        "camera_label": "Take a photo of one leaf",
        "tips": ["One leaf, close up", "Daylight, no shadow", "Plain background (hand or paper)"],
        "common": "Common in Punjab / North India", "other": "Other crops (reference demo)",
        "blur": "Photo looks blurry. A sharper photo gives a better answer.",
        "dark": "Photo looks dark. Try daylight.",
        "conf_high": "Confidence: High", "conf_med": "Confidence: Medium",
        "symptoms": "Symptoms", "prevention": "Prevention",
        "treatment": "Treatment guidance (read the product label, ask your agriculture officer)",
        "source": "Source",
        "confirm": "Always confirm with your local agriculture officer before treating.",
        "unsure_t": "Couldn't identify clearly",
        "unsure": "Please retake a close photo of one leaf on a plain background, in good light. If it still fails, show the plant to your local agriculture officer.",
        "healthy": "{crop} leaf looks healthy.",
        "similar_t": "Could be two similar diseases",
        "similar": "Could be {a} or {b}. These look very similar, so please confirm with your local agriculture officer before spraying anything.",
        "partial": "Some advice is not yet available in your language. English is shown where it is missing.",
        "noadvice": "Advisory for this disease is not available yet.",
        "footer": "Prototype for the Bharat AI-SoC Challenge. Runs fully offline with ExecuTorch on Arm hardware. It supports your agriculture officer and does not replace them."},
 "hi": {"title": "फ़सल डॉक्टर", "subtitle": "पत्ती की बीमारी पहचानें और सलाह पाएँ, सीधे आपके डिवाइस पर",
        "pills": ["बिना इंटरनेट", "English · हिन्दी · ਪੰਜਾਬੀ", "Arm पर चलता है"],
        "st_photo": "फ़ोटो", "st_crop": "फ़सल", "st_result": "नतीजा",
        "step1": "1. पत्ती की फ़ोटो लें या अपलोड करें", "step2": "2. यह कौन सी फ़सल है?", "result": "नतीजा",
        "src_upload": "फ़ोटो अपलोड करें", "src_camera": "कैमरा", "choose": "फ़ोटो चुनें",
        "camera_label": "एक पत्ती की फ़ोटो लें",
        "tips": ["एक पत्ती, नज़दीक से", "दिन की रोशनी, बिना परछाईं", "सादा पीछे का रंग (हाथ या कागज़)"],
        "common": "पंजाब / उत्तर भारत में आम फ़सलें", "other": "अन्य फ़सलें (डेमो)",
        "blur": "फ़ोटो धुंधली है। साफ़ फ़ोटो से बेहतर जवाब मिलेगा।",
        "dark": "फ़ोटो अँधेरी है। दिन की रोशनी में लें।",
        "conf_high": "पहचान का भरोसा: ऊँचा", "conf_med": "पहचान का भरोसा: मध्यम",
        "symptoms": "लक्षण", "prevention": "रोकथाम",
        "treatment": "उपचार संबंधी जानकारी (दवा का लेबल पढ़ें और अपने कृषि अधिकारी से पूछें)",
        "source": "स्रोत",
        "confirm": "इलाज करने से पहले हमेशा अपने स्थानीय कृषि अधिकारी से पुष्टि करें।",
        "unsure_t": "साफ़ पहचान नहीं हो सकी",
        "unsure": "कृपया एक पत्ती की नज़दीक से, सादे पीछे के रंग पर, अच्छी रोशनी में दोबारा फ़ोटो लें। फिर भी न हो तो पौधा अपने स्थानीय कृषि अधिकारी को दिखाएँ।",
        "healthy": "{crop} की पत्ती स्वस्थ दिखती है।",
        "similar_t": "दो मिलती-जुलती बीमारियाँ हो सकती हैं",
        "similar": "यह {a} या {b} हो सकता है। दोनों बहुत मिलते-जुलते हैं, इसलिए कुछ भी छिड़कने से पहले अपने स्थानीय कृषि अधिकारी से पुष्टि करें।",
        "partial": "कुछ सलाह अभी आपकी भाषा में उपलब्ध नहीं है। जहाँ नहीं है वहाँ अंग्रेज़ी दिखाई जा रही है।",
        "noadvice": "इस रोग की सलाह अभी उपलब्ध नहीं है।",
        "footer": "भारत AI-SoC चैलेंज के लिए बना प्रोटोटाइप। Arm हार्डवेयर पर ExecuTorch के साथ पूरी तरह ऑफ़लाइन चलता है। यह आपके कृषि अधिकारी की जगह नहीं लेता।"},
 "pa": {"title": "ਫ਼ਸਲ ਡਾਕਟਰ", "subtitle": "ਪੱਤੇ ਦੀ ਬਿਮਾਰੀ ਪਛਾਣੋ ਅਤੇ ਸਲਾਹ ਲਓ, ਸਿੱਧਾ ਤੁਹਾਡੇ ਡਿਵਾਈਸ ਉੱਤੇ",
        "pills": ["ਬਿਨਾਂ ਇੰਟਰਨੈੱਟ", "English · हिन्दी · ਪੰਜਾਬੀ", "Arm ਉੱਤੇ ਚੱਲਦਾ ਹੈ"],
        "st_photo": "ਫੋਟੋ", "st_crop": "ਫ਼ਸਲ", "st_result": "ਨਤੀਜਾ",
        "step1": "1. ਪੱਤੇ ਦੀ ਫੋਟੋ ਖਿੱਚੋ ਜਾਂ ਅੱਪਲੋਡ ਕਰੋ", "step2": "2. ਇਹ ਕਿਹੜੀ ਫ਼ਸਲ ਹੈ?", "result": "ਨਤੀਜਾ",
        "src_upload": "ਫੋਟੋ ਅੱਪਲੋਡ ਕਰੋ", "src_camera": "ਕੈਮਰਾ", "choose": "ਫੋਟੋ ਚੁਣੋ",
        "camera_label": "ਇੱਕ ਪੱਤੇ ਦੀ ਫੋਟੋ ਖਿੱਚੋ",
        "tips": ["ਇੱਕ ਪੱਤਾ, ਨੇੜਿਓਂ", "ਦਿਨ ਦੀ ਰੋਸ਼ਨੀ, ਬਿਨਾਂ ਪਰਛਾਵੇਂ", "ਸਾਦਾ ਪਿਛੋਕੜ (ਹੱਥ ਜਾਂ ਕਾਗਜ਼)"],
        "common": "ਪੰਜਾਬ / ਉੱਤਰੀ ਭਾਰਤ ਦੀਆਂ ਆਮ ਫ਼ਸਲਾਂ", "other": "ਹੋਰ ਫ਼ਸਲਾਂ (ਡੈਮੋ)",
        "blur": "ਫੋਟੋ ਧੁੰਦਲੀ ਹੈ। ਸਾਫ਼ ਫੋਟੋ ਨਾਲ ਬਿਹਤਰ ਜਵਾਬ ਮਿਲੇਗਾ।",
        "dark": "ਫੋਟੋ ਹਨੇਰੀ ਹੈ। ਦਿਨ ਦੀ ਰੋਸ਼ਨੀ ਵਿੱਚ ਖਿੱਚੋ।",
        "conf_high": "ਪਛਾਣ ਦਾ ਭਰੋਸਾ: ਉੱਚਾ", "conf_med": "ਪਛਾਣ ਦਾ ਭਰੋਸਾ: ਦਰਮਿਆਨਾ",
        "symptoms": "ਲੱਛਣ", "prevention": "ਰੋਕਥਾਮ",
        "treatment": "ਇਲਾਜ ਸੰਬੰਧੀ ਜਾਣਕਾਰੀ (ਦਵਾਈ ਦਾ ਲੇਬਲ ਪੜ੍ਹੋ ਅਤੇ ਆਪਣੇ ਖੇਤੀਬਾੜੀ ਅਫ਼ਸਰ ਤੋਂ ਪੁੱਛੋ)",
        "source": "ਸਰੋਤ",
        "confirm": "ਇਲਾਜ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ ਹਮੇਸ਼ਾ ਆਪਣੇ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਅਫ਼ਸਰ ਤੋਂ ਪੁਸ਼ਟੀ ਕਰੋ।",
        "unsure_t": "ਸਾਫ਼ ਪਛਾਣ ਨਹੀਂ ਹੋ ਸਕੀ",
        "unsure": "ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ ਪੱਤੇ ਦੀ ਨੇੜਿਓਂ, ਸਾਦੇ ਪਿਛੋਕੜ ਉੱਤੇ, ਚੰਗੀ ਰੋਸ਼ਨੀ ਵਿੱਚ ਦੁਬਾਰਾ ਫੋਟੋ ਖਿੱਚੋ। ਫਿਰ ਵੀ ਨਾ ਹੋਵੇ ਤਾਂ ਪੌਦਾ ਆਪਣੇ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਅਫ਼ਸਰ ਨੂੰ ਦਿਖਾਓ।",
        "healthy": "{crop} ਦਾ ਪੱਤਾ ਤੰਦਰੁਸਤ ਲੱਗਦਾ ਹੈ।",
        "similar_t": "ਦੋ ਮਿਲਦੀਆਂ-ਜੁਲਦੀਆਂ ਬਿਮਾਰੀਆਂ ਹੋ ਸਕਦੀਆਂ ਹਨ",
        "similar": "ਇਹ {a} ਜਾਂ {b} ਹੋ ਸਕਦਾ ਹੈ। ਦੋਵੇਂ ਬਹੁਤ ਮਿਲਦੇ-ਜੁਲਦੇ ਹਨ, ਇਸ ਲਈ ਕੁਝ ਵੀ ਛਿੜਕਣ ਤੋਂ ਪਹਿਲਾਂ ਆਪਣੇ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਅਫ਼ਸਰ ਤੋਂ ਪੁਸ਼ਟੀ ਕਰੋ।",
        "partial": "ਕੁਝ ਸਲਾਹ ਹਾਲੇ ਤੁਹਾਡੀ ਭਾਸ਼ਾ ਵਿੱਚ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਜਿੱਥੇ ਨਹੀਂ ਹੈ ਉੱਥੇ ਅੰਗਰੇਜ਼ੀ ਦਿਖਾਈ ਜਾ ਰਹੀ ਹੈ।",
        "noadvice": "ਇਸ ਰੋਗ ਬਾਰੇ ਸਲਾਹ ਹਾਲੇ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
        "footer": "ਭਾਰਤ AI-SoC ਚੈਲੰਜ ਲਈ ਬਣਿਆ ਪ੍ਰੋਟੋਟਾਈਪ। Arm ਹਾਰਡਵੇਅਰ ਉੱਤੇ ExecuTorch ਨਾਲ ਪੂਰੀ ਤਰ੍ਹਾਂ ਆਫ਼ਲਾਈਨ ਚੱਲਦਾ ਹੈ। ਇਹ ਤੁਹਾਡੇ ਖੇਤੀਬਾੜੀ ਅਫ਼ਸਰ ਦੀ ਥਾਂ ਨਹੀਂ ਲੈਂਦਾ।"},
}
CROP_ICON = {"Potato": "🥔", "Tomato": "🍅", "Pepper,_bell": "🫑", "Corn_(maize)": "🌽", "Grape": "🍇",
             "Apple": "🍎", "Cherry_(including_sour)": "🍒", "Peach": "🍑", "Strawberry": "🍓"}

st.set_page_config(page_title="Crop Doctor", page_icon="🌱", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

hero = st.empty()
lang = st.radio("Language", ["English", "हिन्दी", "ਪੰਜਾਬੀ"], horizontal=True, label_visibility="collapsed")
LK = {"English": "en", "हिन्दी": "hi", "ਪੰਜਾਬੀ": "pa"}[lang]
stepper = st.empty()

def T(k, **kw):
    v = UI[LK][k]
    return v.format(**kw) if kw else v

pills = "".join(f'<span class="pill">{E(p)}</span>' for p in UI[LK]["pills"])
hero.markdown(f'<div class="hero"><div class="ht">🌱 {E(T("title"))}</div>'
              f'<div class="hs">{E(T("subtitle"))}</div>{pills}</div>', unsafe_allow_html=True)

def set_step(n):
    labels = [T("st_photo"), T("st_crop"), T("st_result")]
    items = ""
    for i, l in enumerate(labels, 1):
        cls = "done" if i < n else ("now" if i == n else "")
        dot = "✓" if i < n else str(i)
        items += f'<div class="step {cls}"><span class="dot">{dot}</span>{E(l)}</div>'
    stepper.markdown(f'<div class="stepper">{items}</div>', unsafe_allow_html=True)

def crop_name(c): return CROPS[c].get(LK) or CROPS[c]["en"]

def loc(e, field):
    if LK != "en" and e.get(LK, {}).get(field):
        return e[LK][field], True
    return e[field], LK == "en"

def disease_name(e, key):
    if e and LK != "en" and e.get(LK, {}).get("name"): return e[LK]["name"]
    return pretty(key)

def as_list(x): return [x] if isinstance(x, str) else list(x)

def banner(kind, icon, title, sub=""):
    s = f'<div class="bs">{E(sub)}</div>' if sub else ""
    st.markdown(f'<div class="banner {kind}"><div class="ico">{icon}</div>'
                f'<div><div class="bt">{E(title)}</div>{s}</div></div>', unsafe_allow_html=True)

def card(title, inner):
    st.markdown(f'<div class="card"><div class="ct">{E(title)}</div>{inner}</div>', unsafe_allow_html=True)

def ul(items): return "<ul>" + "".join(f"<li>{E(x)}</li>" for x in as_list(items)) + "</ul>"

def linkify(s):
    return re.sub(r"(https?://[^\s|<]+)", r'<a href="\1" target="_blank">\1</a>', E(s))

def confirm_box():
    st.markdown(f'<div class="confirm">⚠️ {E(T("confirm"))}</div>', unsafe_allow_html=True)

def render_entry(e):
    sym, a = loc(e, "symptoms"); prev, b = loc(e, "prevention"); man, c = loc(e, "management")
    if not (a and b and c):
        st.markdown(f'<div class="note">{E(T("partial"))}</div>', unsafe_allow_html=True)
    card(T("symptoms"), f"<p>{E(sym)}</p>")
    card(T("prevention"), ul(prev))
    with st.expander(T("treatment")):
        st.write(man)
    st.markdown(f'<div class="src">{E(T("source"))}: {linkify(e["source"])}</div>', unsafe_allow_html=True)

def crop_buttons(keys, ncols, prefix):
    cols = st.columns(ncols)
    for i, c in enumerate(keys):
        meta = CROPS.get(c, {"en": c, "hi": "", "pa": ""})
        label = f"{CROP_ICON.get(c, '🌿')} {meta['en']} • {meta['hi']} • {meta['pa']}"
        picked = st.session_state.get("crop") == c
        if cols[i % ncols].button(label, key=f"{prefix}_{c}", use_container_width=True,
                                  type="primary" if picked else "secondary"):
            st.session_state["crop"] = c
            st.rerun()

def footer():
    st.markdown(f'<div class="foot">{E(T("footer"))}</div>', unsafe_allow_html=True)

# ---------------- Step 1: photo ----------------
set_step(1)
st.markdown(f'<div class="sect">{E(T("step1"))}</div>', unsafe_allow_html=True)
tips = "".join(f"<span>✔ {E(t)}</span>" for t in UI[LK]["tips"])
st.markdown(f'<div class="tips">{tips}</div>', unsafe_allow_html=True)
mode = st.radio("Photo source", ["upload", "camera"], horizontal=True, label_visibility="collapsed",
                format_func=lambda m: T("src_upload") if m == "upload" else T("src_camera"))
file = (st.file_uploader(T("choose"), type=["jpg", "jpeg", "png"]) if mode == "upload"
        else st.camera_input(T("camera_label")))
if file is None:
    footer(); st.stop()

img_id = hashlib.md5(file.getvalue()).hexdigest()
if st.session_state.get("img_id") != img_id:
    st.session_state["img_id"], st.session_state["crop"] = img_id, None
img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
st.image(img, width=300)
sharp, bright = photo_quality(img)
if sharp < BLUR_MIN: st.warning(T("blur"))
if bright < DARK_MIN: st.warning(T("dark"))

# ---------------- Step 2: crop ----------------
set_step(2)
st.markdown(f'<div class="sect">{E(T("step2"))}</div>', unsafe_allow_html=True)
available = [c for c in CROPS if c in CROP_TO_MODEL]
punjab = [c for c in available if CROPS[c].get("tier") == "punjab"]
other = [c for c in available if CROPS[c].get("tier") != "punjab"]
st.markdown(f'<div class="sub">{E(T("common"))}</div>', unsafe_allow_html=True)
crop_buttons(punjab, 2, "p")
with st.expander(T("other")):
    crop_buttons(other, 2, "o")
crop = st.session_state.get("crop")
if crop is None:
    footer(); st.stop()

# ---------------- Step 3: result ----------------
set_step(3)
st.markdown(f'<div class="sect">{E(T("result"))}</div>', unsafe_allow_html=True)
cfg = MODELS[CROP_TO_MODEL[crop]]
res = predict(img, crop)
best, p1 = res["top"][0]

alt = None
if res["crop_mass"] >= cfg["crop_mass_min"]:
    alt = confusable_with(res["top"], cfg["conf_threshold"])

if res["crop_mass"] < cfg["crop_mass_min"] or (alt is None and p1 < cfg["conf_threshold"]):
    banner("warn", "📷", T("unsure_t"), T("unsure"))
elif alt:
    e, ea = KB.get(best), KB.get(alt)
    banner("warn", "🔎", T("similar_t"), T("similar", a=disease_name(e, best), b=disease_name(ea, alt)))
    cols = st.columns(2)
    for col, (k, ee) in zip(cols, ((best, e), (alt, ea))):
        if ee:
            s_, _ = loc(ee, "symptoms"); p_, _ = loc(ee, "prevention")
            with col:
                card(disease_name(ee, k), f"<p><b>{E(T('symptoms'))}:</b> {E(s_)}</p>"
                     f"<p style='margin-top:6px'><b>{E(T('prevention'))}:</b></p>{ul(p_)}")
    confirm_box()
elif best.endswith("healthy"):
    banner("ok", "✅", T("healthy", crop=crop_name(crop)))
else:
    e = KB.get(best)
    if not e:
        banner("ok", "🦠", f"{crop_name(crop)}: {pretty(best)}", T("noadvice"))
    else:
        banner("ok", "🦠", f"{crop_name(crop)}: {disease_name(e, best)}")
        st.markdown(f'<span class="chip">{E(T("conf_high") if p1 >= 0.85 else T("conf_med"))}</span>',
                    unsafe_allow_html=True)
        render_entry(e)
        if LK == "en":
            key = "ai_" + best
            if st.button("✨ Explain in simple words (AI, runs offline)"):
                with st.spinner("Llama 3.2 1B is writing (the first time it loads the model)..."):
                    st.session_state[key] = ai_explain(e)
            if key in st.session_state:
                txt, secs = st.session_state[key]
                if txt:
                    st.markdown(f'<div class="ai"><span class="chip">AI · Llama 3.2 1B, on-device</span>'
                                f'<p>{E(txt)}</p><div class="aisub">Checked against the verified text above · '
                                f'{secs:.0f}s</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="note">The AI text did not pass the safety check, so only the '
                                'verified text is shown.</div>', unsafe_allow_html=True)
    confirm_box()

with st.expander("Technical details"):
    st.write(f"Machine: {platform.machine()} | Backend: {res['backend']} | CNN time: {res['secs']*1000:.0f} ms | "
             f"crop-match: {res['crop_mass']:.0%} | threshold: {cfg['conf_threshold']:.0%}")
    st.write("After crop filter:")
    for n, p in res["top"]: st.write(f"{p*100:5.1f}%  {n}")
    st.write("Before crop filter:")
    for n, p in res["raw"]: st.write(f"{p*100:5.1f}%  {n}")
footer()