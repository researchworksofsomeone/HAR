#!/bin/bash
set -e

echo "=================================================="
echo "  JADE Reproducibility Script (Double-Blind)"
echo "=================================================="

echo "[1/5] Syncing Full Dataset & Verifying Matrix..."
python scripts/sync_and_verify_full_data.py

echo "[2/5] Regenerating Statistics and SOTA Comparisons..."
# We assume the user has run the evaluations or downloaded them.
python scripts/analyze_results.py

echo "[3/5] Regenerating Publication Figures..."
python scripts/generate_figures.py
cp figures/*.png all_results/figures/ || true

echo "[4/5] Regenerating LaTeX Tables..."
python scripts/generate_tables.py
cp results/*.tex all_results/results/ || true

echo "[5/5] Compiling IEEE LaTeX Manuscript..."
if command -v latexmk &> /dev/null; then
    cd paper
    latexmk -pdf -cd main.tex
    cd ..
    echo "✅ Paper compiled successfully: paper/main.pdf"
else
    echo "⚠ latexmk not found. Skipping PDF compilation. Please compile paper/main.tex manually."
fi

echo "=================================================="
echo "  REPRODUCTION COMPLETE"
echo "=================================================="
