import glob
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("Aggregating Massive Evaluation Results...")
    
    chunks_dir = PROJECT_ROOT / "data" / "results" / "raw_eval_chunks"
    if not chunks_dir.exists():
        print(f"Directory {chunks_dir} does not exist.")
        sys.exit(1)
        
    files = glob.glob(str(chunks_dir / "result_task*.parquet"))
    print(f"Found {len(files)} chunk files.")
    
    if len(files) == 0:
        print("No files to aggregate.")
        sys.exit(0)
        
    # Read chunked to avoid memory spike
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    out_dir = PROJECT_ROOT / "data" / "results"
    out_path = out_dir / "full_evaluation.parquet"
    
    full_df.to_parquet(out_path)
    
    print("\n================ SUMMARY ================")
    print(f"Total rows (tasks): {len(full_df)}")
    print(f"Unique Datasets: {full_df['dataset'].unique().tolist()}")
    print(f"Unique Regimes: {full_df['regime'].unique().tolist()}")
    print(f"Unique Policies: {len(full_df['policy'].unique())}")
    print(f"Unique Profiles: {full_df['profile'].unique().tolist()}")
    print(f"Peak RAM stats: Mean={full_df['ram_peak_mb'].mean():.2f} MB, Max={full_df['ram_peak_mb'].max():.2f} MB")
    print(f"Saved aggregated results to {out_path.name}")
    print("=========================================\n")

if __name__ == "__main__":
    main()
