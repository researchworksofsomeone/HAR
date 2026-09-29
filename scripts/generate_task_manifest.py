import pandas as pd
from pathlib import Path
import itertools

def main():
    print("Generating Task Manifest for Full Evaluation Matrix...")
    
    datasets = ["UCI-HAR", "PAMAP2", "HHAR"]
    # Regimes R1-R6
    regimes = ["R1", "R2", "R3", "R4", "R5", "R6"]
    
    policies = [
        "SRC", "TENT-ALWAYS", "BN-ALWAYS", "EM-ALWAYS", "CASCADE", "PERIODIC-TENT", 
        "ENTROPY-TRIGGER", "SURPRISE-TRIGGER", "SHIFT-TRIGGER", "PH-DRIFT", 
        "TOAST-STYLE", "LEAN-ALWAYS", "LAME-STYLE", 
        "JADE-indomain", "JADE-LORO", "JADE-LODO", "JADE-cold", "FLEET-JADE", 
        "HMO(λ=0)", "HMO(λ=1)"
    ]
    
    seeds = [42, 43, 44]
    profiles = ["P1", "P2", "P3"]
    
    tasks = []
    task_id = 0
    
    for dataset in datasets:
        # ~8 eval subjects (skipping controller training subset)
        if dataset == "UCI-HAR":
            subjects = list(range(21, 31)) # 10 eval subjects
        elif dataset == "PAMAP2":
            subjects = list(range(7, 10)) # 3 eval subjects (since it only has 9 total)
        else:
            subjects = list(range(7, 10))
            
        for regime in regimes:
            if dataset == "HHAR" and regime == "R6":
                # Only exploratory for HHAR as per prompt, but we'll include it
                pass
            if dataset != "HHAR" and regime == "R5":
                # R5 only for HHAR
                continue
                
            for subj in subjects:
                for policy in policies:
                    for seed in seeds:
                        # We only ablate profiles for JADE policies; static policies only run on P1 for baselining 
                        # to save compute, unless specified. We will just map all to P1, and JADE to P1,P2,P3.
                        if "JADE" in policy:
                            active_profiles = profiles
                        else:
                            active_profiles = ["P1"]
                            
                        for profile in active_profiles:
                            tasks.append({
                                "task_id": task_id,
                                "dataset": dataset,
                                "regime": regime,
                                "subject_idx": subj,
                                "policy": policy,
                                "seed": seed,
                                "profile": profile
                            })
                            task_id += 1
                            
    df = pd.DataFrame(tasks)
    
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "task_manifest.csv"
    
    df.to_csv(out_path, index=False)
    print(f"Manifest generated at {out_path}")
    print(f"Total Tasks: {len(df)}")

if __name__ == "__main__":
    main()
