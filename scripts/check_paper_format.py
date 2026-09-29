import os
import sys
import subprocess
from pathlib import Path
try:
    import PyPDF2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("==================================================")
    print("  TASK 6: Formatting & Page Limit Check")
    print("==================================================")
    
    paper_dir = PROJECT_ROOT / "paper"
    pdf_path = paper_dir / "main.pdf"
    
    if not pdf_path.exists():
        print("⚠ main.pdf not found. Please compile the LaTeX paper first.")
        return
        
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            
        print(f"Total Pages: {num_pages}")
        
        if num_pages > 7:
            print("❌ WARNING: Paper exceeds the 6+1 page limit!")
            print("Suggestion: Trim Related Work or compress the Evaluation graphs.")
        elif num_pages == 7:
            print("✅ Verified: Paper perfectly hits the 6+1 IEEE page limit.")
        else:
            print("✅ Verified: Paper is well within the 6+1 IEEE page limit.")
            
    except Exception as e:
        print(f"Error checking PDF: {e}")

if __name__ == "__main__":
    main()
