import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("==================================================")
    print("  TASK 7: Ultimate Double-Blind Sweep")
    print("==================================================")
    
    # Patterns to redact
    patterns = [
        (re.compile(r'\bAntigravity\b', re.IGNORECASE), "agentic framework"),
        (re.compile(r'\bMOLAB\b', re.IGNORECASE), "compute cluster"),
        (re.compile(r'\bsatabarto\b', re.IGNORECASE), "researcher"),
        (re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', re.IGNORECASE), "anonymous@4open.science")
    ]
    
    extensions = {'.py', '.tex', '.md', '.yaml', '.json', '.sh'}
    exclude_dirs = {'.git', '__pycache__', 'env', 'data', 'figures', 'checkpoints'}
    
    total_matches = 0
    modified_files = 0
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            ext = Path(file).suffix
            if ext in extensions:
                filepath = Path(root) / file
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    continue
                    
                original_content = content
                for pat, repl in patterns:
                    # avoid redacting the target email itself if we already injected it
                    if repl == "anonymous@4open.science" and "anonymous@4open.science" in content:
                        content = pat.sub(lambda m: "anonymous@4open.science" if m.group(0) == "anonymous@4open.science" else repl, content)
                    else:
                        content = pat.sub(repl, content)
                    
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    modified_files += 1
                    # count differences roughly
                    total_matches += 1
                    
    if modified_files > 0:
        print(f"✅ {modified_files} files sanitized. Double-blind compliance enforced.")
    else:
        print("✅ 0 identifiable strings found. Double-blind compliance verified.")

if __name__ == "__main__":
    main()
