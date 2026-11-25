import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = json.load(open(os.path.join(ROOT, "app", "kb.json"), encoding="utf-8"))
need = ["Potato___Early_blight", "Potato___Late_blight", "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Tomato___Early_blight", "Tomato___Late_blight"]
print(len(KB), "entries loaded")
for k in need:
    e = KB.get(k, {})
    print(("OK      " if e.get("hi") and e.get("pa") else "MISSING ") + k)     