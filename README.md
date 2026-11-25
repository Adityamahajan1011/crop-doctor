# Crop Doctor: offline crop disease assistant on Raspberry Pi 5

Bharat AI-SoC Challenge 2026-27, Problem Statement 1.
Student: Aditya Mahajan, B.Tech ECE, NIT Jalandhar (Roll No. 2410405).

A farmer photographs one crop leaf, picks the crop, and gets the disease name, symptoms, prevention and source, in English, Hindi or Punjabi. An optional on-device language model (Llama 3.2 1B through ExecuTorch) rewrites the symptoms in simpler words. The system runs offline on a Raspberry Pi 5 (Arm Cortex-A76, 4 GB RAM).

## Results (summary)
- Vision model on real photos (PlantDoc test, 236 images): 65.3% top-1, 87.7% top-3. With the crop selected: 76.7% top-1, 94.9% top-3.
- Lab test set (PlantVillage, 5,431 images): 99.2% top-1. Optimistic because of near-duplicate photos.
- Informal test on 51 internet photos (17 diseases, 3 each), crop selected: 39.5/51 (77.5%).
- Raspberry Pi 5, 4 GB: vision model 10 ms per photo. Llama first word 1.1 s and a 39-token answer in 3.7 s (warm). Peak memory about 2.5 GB.

Full report: [`docs/CropDoctor_Final_Project_Report.pdf`](docs/CropDoctor_Final_Project_Report.pdf)

## Repository layout

```
crop-doctor/
├── README.md
├── requirements.txt
├── app/                 Streamlit application
│   ├── app.py           interface, safety rules, model backends
│   ├── registry.json    models and crops (add a model here)
│   ├── kb.json          knowledge base: 24 diseases, English / Hindi / Punjabi
│   └── .streamlit/      theme and server settings
├── models/
│   ├── mobilenetv3_field_v1.pth        vision model weights and class names
│   └── mobilenetv3_field_v1_fp32.pte   vision model as an ExecuTorch file (used by the app)
├── scripts/
│   ├── check_kb.py          checks the knowledge base and translations
│   ├── pipeline_test.py     runs the full pipeline once from the command line
│   ├── benchmark.py         measures latency and memory of both models
│   └── test_llama.py        checks that the Llama model loads and answers
├── results/
│   └── benchmark_cropdoctor.json   benchmark output from the Raspberry Pi 5
├── samples/             test leaf photos (crc, plb, peb, teb, tlb)
└── docs/                final report
```

**Not in the repository:** the Llama model folder `llama_pte/` (828 MB, too large for GitHub). Download it from **[Google Drive](https://drive.google.com/drive/folders/1aqNYn61axUo-xbuPneHEVLSAfG91ePgx?usp=drive_link)** and place it in the project root, so the root contains `llama_pte/model.pte` and `llama_pte/tokenizer/`. The app works without it; only the "Explain in simple words" button needs it.

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
Check first: `uname -m` must print