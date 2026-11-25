# Crop Doctor: offline crop disease assistant on Raspberry Pi 5

Bharat AI-SoC Challenge 2026-27, Problem Statement 1.
Student: Aditya Mahajan, B.Tech ECE, NIT Jalandhar (Roll No. 2410405).

A farmer photographs one crop leaf, picks the crop, and gets the disease name, symptoms, prevention and source, in English, Hindi or Punjabi. An optional on-device language model (Llama 3.2 1B through ExecuTorch) rewrites the symptoms in simpler words. The whole system runs offline on a Raspberry Pi 5 (Arm Cortex-A76, 4 GB RAM).

## Results (summary)
- Vision model on real photos (PlantDoc test, 236 images): 65.3% top-1, 87.7% top-3; with the crop selected: 76.7% top-1, 94.9% top-3.
- Lab test set (PlantVillage, 5,431 images): 99.2% top-1. This is optimistic because of near-duplicate photos.
- Informal test on 51 web photos (17 diseases, 3 each), crop selected: 39.5/51 (77.5%).
- Raspberry Pi 5, 4 GB: vision model 10 ms per photo; Llama first word 1.1 s and a 39-token answer in 3.7 s (warm); peak memory about 2.5 GB.

See `docs/` for the full report, benchmark output and screenshots.

## Repository contents
| File | Purpose |
|---|---|
| `app.py` | Streamlit application (interface, safety rules, model backends) |
| `registry.json` | Lists models and crops; add a model here without changing code |
| `kb.json` | Knowledge base: 24 diseases, sources, English / Hindi / Punjabi text |
| `mobilenetv3_field_v1.pth` | Vision model weights, also supply the class names |
| `mobilenetv3_field_v1_fp32.pte` | Vision model as an ExecuTorch file (used by the app) |
| `benchmark.py` | Measures latency and memory of both models |
| `pipeline_test.py` | Runs the full pipeline once from the command line |
| `check_kb.py` | Checks that the knowledge base loads and has the translations |
| `requirements.txt` | Python dependencies |
| `docs/` | Report, benchmark output, screenshots |

**Not in the repository:** the Llama model folder `llama_pte/` (828 MB, too large for GitHub).
Download it from: **https://drive.google.com/drive/folders/1aqNYn61axUo-xbuPneHEVLSAfG91ePgx?usp=drive_link**
Place it in the project root so that it contains `llama_pte/model.pte` and `llama_pte/tokenizer/`.
The app still works without it; only the "Explain in simple words" button needs it.

## Setup on a Windows PC
Requires Python 3.13 and about 5 GB of free disk space.

```
git clone https://github.com/Adityamahajan1011/crop-doctor.git
cd crop-doctor
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Setup on Raspberry Pi 5 (Raspberry Pi OS Lite, 64-bit)
Check first: `uname -m` must print `aarch64`.

```
sudo apt update && sudo apt install -y git python3-venv
git clone https://github.com/Adityamahajan1011/crop-doctor.git
cd crop-doctor
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```
Then copy the `llama_pte` folder from Google Drive into `crop-doctor/` (for example with `scp -r llama_pte pi@cropdoctor.local:~/crop-doctor/`).

Check the runtime:
```
python -c "from executorch.runtime import Runtime; print('runtime ok')"
```

## Run the app
Windows:
```
cd crop-doctor
venv\Scripts\activate
streamlit run app.py
```
Raspberry Pi (so phones on the same network can connect):
```
cd ~/crop-doctor
source venv/bin/activate
streamlit run app.py --server.address 0.0.0.0 --server.headless true
```
Then open `http://localhost:8501` on the PC, or `http://<pi-address>:8501` from a phone. The Pi's address is shown by `hostname -I`.

To measure performance on the Pi:
```
python benchmark.py crc.png
```
This writes `benchmark_<hostname>.json`.

## How the app works
1. The farmer uploads or takes a leaf photo. Blurry or dark photos get a warning.
2. The farmer taps the crop. Only that crop's classes are considered.
3. The vision model (MobileNetV3-Large, ExecuTorch) scores the classes.
4. Safety rules:
   - Low confidence (below 60%) or a photo that does not match the crop gives "retake the photo".
   - If the top two answers are a known look-alike pair (for example potato early and late blight), both are shown, and no spray dose is shown.
5. The disease name, symptoms, prevention, treatment guidance and source come from `kb.json`, in the chosen language.
6. Optional: Llama 3.2 1B rewrites the symptoms in simple English. A checker rejects text that adds numbers, chemicals or colours not in the verified text.

## Adding a model or crop
1. Train a model and export it. Save the class names as `Crop___Disease`.
2. Add a new entry under `models` in `registry.json` with `backend`, `path` and `enabled_crops`.
3. Add the crop under `crops`, and its diseases to `kb.json`.

## Troubleshooting
- **Windows: "Access denied" under `.streamlit`:** before starting, run `set USERPROFILE=%CD%\home` and create the folder `home`.
- **"No module named 'optimum'":** the environment is not active. Run `venv\Scripts\activate` first.
- **The AI button fails on the Pi:** check that `llama_pte` is in the project folder, and run `free -h` (about 2.5 GB is needed while Llama runs).

## Limitations
- Pest detection is not implemented; severity estimation was removed as unreliable.
- The training data has no wheat, rice, cotton or sugarcane.
- Accuracy on real photos is about 65% without and 77% with the crop selected, on a small test set. The crop-selected number assumes the farmer picks the right crop.
- Llama 3.2 1B cannot write Hindi or Punjabi reliably, so those languages use pre-written text. That text was drafted with AI assistance and has not yet been checked by an agricultural expert.

## Acknowledgements and licences
- PlantVillage and PlantDoc datasets (public research datasets; check their licences before any commercial use).
- Punjab Agricultural University, Package of Practices for Vegetables (2025), and University of Minnesota Extension (sources in `kb.json`).
- Llama 3.2 is by Meta and is used under the Llama 3.2 Community Licence. Download it yourself from Hugging Face after accepting the licence.
- Built with PyTorch, ExecuTorch, optimum-executorch, Streamlit and torchvision.
