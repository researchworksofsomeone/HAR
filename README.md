# 👑 PERCON-AI: ULTRA-PROMAX MASTER EXECUTION GUIDE
*(Zero Hallucination. Zero Thinking Required. 100% Deterministic Execution.)*

This document provides the absolute, step-by-step, irrefutable command sequence to execute the entire PerconAI (Agentic Framework) pipeline on your local GPU-enabled machine (RTX 3050, 16GB RAM) from absolute zero to the final IEEE PDF.

The repository has been **RUTHLESSLY PURGED**. No fake checkpoints, no mock evaluations, no logs exist. You are starting from a completely blank, mathematically pure slate. Follow these instructions exactly in order.

---

## 🛠️ PHASE 1: ENVIRONMENT & REPOSITORY SETUP

**1. Clone the Purged Repository**
Open your terminal on your PC and clone the completely wiped repository:
```bash
git clone https://github.com/researchworksofsomeone/HAR.git
cd HAR
```

**2. Create a Clean Python Environment (Recommended)**
```bash
python -m venv percon_env
source percon_env/bin/activate   # On Windows: percon_env\Scripts\activate
```

**3. Install Absolute Dependencies**
```bash
pip install -r requirements.txt
pip install huggingface_hub
```

---

## 💾 PHASE 2: ACQUIRING THE HEAVY DATASETS
Because GitHub limits files to 100MB, the massive sensor datasets (`.memmap` files) and pre-trained neural networks (`.pth`) are securely stored on HuggingFace.

**1. Authenticate with Hugging Face**
You must authenticate to download the private data. Run this command and paste your token when prompted.
```bash
huggingface-cli login --token <INSERT_YOUR_HUGGING_FACE_TOKEN_HERE>
```

**2. Download the Heavy Assets**
Run this exactly. It pulls the massive datasets into your local folder.
```bash
huggingface-cli download Satabarto/Percon2 jade_submission_package.zip --local-dir .
```

**3. Unzip into the Project**
```bash
unzip -o jade_submission_package.zip
```
*VERIFICATION:* Check your folder. You should now see heavy files inside `data/memmap/` and `.pth` files inside `checkpoints/`. If you see them, proceed to Phase 3.

---

## 🚀 PHASE 3: THE MASTER GPU EVALUATION
This is the colossal, computationally heavy matrix execution. It utilizes your RTX 3050 via CUDA. 

**1. Launch the Master Script**
Run the following command. It will read the `.memmap` data, stream it through the JADE triggers, and compute analytical MCU energy costs.
```bash
python scripts/run_full_empirical_eval.py
```

**2. The Resume Protocol (CRITICAL)**
If your PC crashes, goes to sleep, runs out of memory, or you hit `CTRL+C`: **Do not panic.**
Just run `python scripts/run_full_empirical_eval.py` again. It will read `all_results/full_evaluation.parquet`, skip everything it already finished, and instantly resume exactly where it died.

Do not move to Phase 4 until your terminal literally prints:
`✅ FULL EMPIRICAL EXECUTION COMPLETE.`

---

## 📊 PHASE 4: STATISTICAL ANALYSIS & PLOTTING
Once Phase 3 finishes, you have gigabytes of raw metrics in `all_results/full_evaluation.parquet`. You must now crunch this into p-values, SOTA tables, and charts.

**1. Generate the P-Values and SOTA Deltas**
This computes the Holm-Bonferroni corrected Wilcoxon tests and outputs `hypothesis_test_results.json`.
```bash
python scripts/analyze_results.py
```

**2. Generate the LaTeX Tables**
This dynamically calculates Table 2 (Main Results) and Table 3 (Ablations) and saves them as `.tex` files.
```bash
python scripts/generate_tables.py
```

**3. Generate the Matplotlib Figures**
This draws the Pareto frontiers and Break-even plots and saves them as `.png` files.
```bash
python scripts/generate_figures.py
```

*VERIFICATION:* Open the `results/` folder and `figures/` folder. You should see `.tex` and `.png` files.

---

## 📄 PHASE 5: COMPILING THE IEEE MANUSCRIPT
The tables and figures are ready. You just need to compile the double-blind IEEE LaTeX document.

**1. Compile the PDF**
Ensure you have a LaTeX compiler installed (e.g., TeX Live, MiKTeX, MacTeX). Then run:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**2. Claim Your Victory**
Open `paper/main.pdf`. You now possess a 100% locally-executed, mathematically perfect, fully reproducible IEEE research paper.
