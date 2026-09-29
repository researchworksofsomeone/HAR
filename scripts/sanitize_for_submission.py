import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def sanitize_file(path, dry_run=False):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original = content
    
    # Replace HF Repos
    if "researcher/Percon2" in content:
        content = content.replace("researcher/Percon2", "anonymous-jade-artifact")
        
    # Replace absolute paths with generic variables
    for pat in [r"/Users/\w+/PerconAI", r"/home/\w+/PerconAI", r"/scratch/\w+/jade_eval_tmp"]:
        matches = re.findall(pat, content)
        for m in set(matches):
            if "scratch" in pat:
                content = content.replace(m, "$TMPDIR")
            else:
                content = content.replace(m, "$PROJECT_ROOT")
            
    # Generic sweep for names
    if "researcher" in content:
        content = content.replace("researcher", "AnonymousAuthor")
        
    replacements_made = 1 if content != original else 0
        
    if not dry_run and content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    return replacements_made

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check-mode only")
    args = parser.parse_args()
    
    print("Sanitizing repository for double-blind submission...")
    
    target_exts = {".py", ".yaml", ".json", ".tex", ".md", ".sh"}
    total_replacements = 0
    files_touched = 0
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if ".git" in root or ".venv" in root or "__pycache__" in root:
            continue
            
        for file in files:
            ext = Path(file).suffix
            if ext in target_exts:
                filepath = Path(root) / file
                
                # Exclude self to avoid infinitely matching the string literal "researcher"
                if filepath.name == "sanitize_for_submission.py":
                    continue
                    
                try:
                    reps = sanitize_file(filepath, dry_run=args.check)
                    if reps > 0:
                        total_replacements += reps
                        files_touched += 1
                        print(f"[{'CHECK' if args.check else 'FIX'}] Sanitized {filepath.name} ({reps} replacements)")
                except Exception as e:
                    pass
                    
    print("\n==================================================")
    print(f"Sanitization Complete. Mode: {'CHECK' if args.check else 'APPLY'}")
    print(f"Files modified: {files_touched}")
    print(f"Total replacements: {total_replacements}")
    
    if args.check and total_replacements > 0:
        print("❌ FAILED: Found identifiable strings in check mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()
