import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("Generating Empirical LaTeX Tables...")
    eval_path = PROJECT_ROOT / "all_results" / "full_evaluation.parquet"
    if not eval_path.exists():
        print(f"FATAL: Empirical evaluation results missing at {eval_path}!")
        sys.exit(1)
        
    df = pd.read_parquet(eval_path)
    
    # Strictly enforce that this is the real, massive empirical matrix
    assert len(df) > 10000, "Parquet file is too small. The full empirical evaluation has not been run yet."
    
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Tab 2: Main Results Grid
    main_cols = ['stream_acc', 'adaptation_tax', 'nag', 'joule_regret', 'ram_peak_mb']
    
    # Filter for main policies
    main_policies = ["SRC", "BN-ALWAYS", "TENT-ALWAYS", "LEAN-ALWAYS", "ENTROPY-TRIGGER", "JADE-indomain", "FLEET-JADE"]
    main_df = df[df['policy'].isin(main_policies)].groupby('policy')[main_cols].mean().round(3)
    main_df.index.name = "Policy"
        
    with open(out_dir / "main_results.tex", "w") as f:
        f.write("% Table 2: Main Results\n")
        f.write(main_df.to_latex(float_format="%.3f"))
        
    # Tab 3: Ablations
    # Calculate empirically from the dataframe for the ablation variants
    ablation_policies = ["JADE-indomain", "JADE-MLP", "JADE-LODO", "FLEET-JADE"]
    abl_df = df[df['policy'].isin(ablation_policies)].groupby('policy')[['joule_regret', 'ram_peak_mb']].mean().round(3)
    
    # Rename columns to match paper requirements
    abl_df = abl_df.rename(columns={"joule_regret": "Regret", "ram_peak_mb": "RAM (MB)"})
    abl_df.index.name = "Variant"
    
    with open(out_dir / "ablations.tex", "w") as f:
        f.write("% Table 3: Ablation Results\n")
        f.write(abl_df.to_latex(float_format="%.2f"))
        
    print(f"Empirical Tables generated in {out_dir}/")

if __name__ == "__main__":
    main()
