import json
KB = json.load(open("kb.json", encoding="utf-8"))      # an error here shows the exact line to fix
need = ["Potato___Early_blight", "Potato___Late_blight", "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Tomato___Early_blight", "Tomato___Late_blight"]
print(len(KB), "entries loaded")
for k in need:
    e = KB.get(k, {})
    print(("OK      " if e.get("simple_hi") and e.get("simple_pa") else "MISSING ") + k)