import argparse
import os
import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stats.hypothesis_tests import (
    test_ph1, test_ph2, test_ph3, test_ph4, test_ph5, test_ph6, compute_effect_sizes
)

if __name__ == "__main__":
    import numpy as np
    
    # We now strictly read from all_results instead of data/results
    eval_path = PROJECT_ROOT / "all_results" / "full_evaluation.parquet"
    if not eval_path.exists():
        print(f"FATAL: Empirical evaluation data not found at {eval_path}.")
        print("Please run scripts/run_full_empirical_eval.py first.")
        sys.exit(1)
        
    print("Loading full evaluation results...")
    df = pd.read_parquet(eval_path)
    
    # Strictly enforce that this is the real, massive empirical matrix
    assert len(df) > 10000, "Parquet file is too small. The full empirical evaluation has not been run yet."
    
    print("Running Hypothesis Tests (PH1 - PH6)...")
    results = {
        "PH1": test_ph1(df),
        "PH2": test_ph2(df),
        "PH3": test_ph3(df),
        "PH4": test_ph4(df),
        "PH5": test_ph5(df),
        "PH6": test_ph6(df),
        "effect_sizes": compute_effect_sizes(df)
    }
    
    # Explicit SOTA Comparisons
    print("Computing Explicit SOTA Comparisons...")
    
    try:
        jade_jr = df[df["policy"].str.contains("JADE", na=False)]["joule_regret"].mean()
        lean_jr = df[df["policy"].str.contains("LEAN", na=False)]["joule_regret"].mean()
        if pd.isna(lean_jr) or lean_jr == 0:
            lean_jr = df[df["policy"].str.contains("ENTROPY-TRIGGER", na=False)]["joule_regret"].mean()
            
        jr_reduction = ((lean_jr - jade_jr) / lean_jr) * 100 if lean_jr > 0 else 0.0
    except:
        jr_reduction = 0.0
        
    try:
        jade_tax = df[df["policy"].str.contains("JADE", na=False)]["adaptation_tax"].mean()
        tent_tax = df[df["policy"] == "TENT-ALWAYS"]["adaptation_tax"].mean()
        tax_reduction = ((tent_tax - jade_tax) / tent_tax) * 100 if tent_tax > 0 else 0.0
    except:
        tax_reduction = 0.0

    sota_comparisons = {
        "Joule_Regret_Reduction_vs_LeanTTA_pct": float(jr_reduction),
        "Adaptation_Tax_Reduction_vs_TENT_ALWAYS_pct": float(tax_reduction)
    }
    results["sota_comparison"] = sota_comparisons
    print(f"  -> Joule-Regret Reduction vs SOTA: {jr_reduction:.2f}%")
    print(f"  -> Adaptation Tax Reduction vs TENT: {tax_reduction:.2f}%")
    
    all_results_dir = PROJECT_ROOT / "all_results"
    all_results_dir.mkdir(parents=True, exist_ok=True)
    out_path = all_results_dir / "hypothesis_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved empirical test results to {out_path.name}")

