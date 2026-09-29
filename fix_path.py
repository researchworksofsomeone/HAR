import json

with open("notebooks/PerconAI_Master.ipynb", "r") as f:
    nb = json.load(f)

# The second cell handles the unzipping and path
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "UNZIP PERCONAI.ZIP" in "".join(cell["source"]):
        cell["source"] = [
            "# 0. UNZIP PERCONAI.ZIP AND SETUP ENVIRONMENT\n",
            "import os\n",
            "import sys\n",
            "import zipfile\n",
            "\n",
            "# Unzip the uploaded PerconAI.zip into the current directory\n",
            "if os.path.exists(\"PerconAI.zip\"):\n",
            "    print(\"Extracting PerconAI.zip...\")\n",
            "    with zipfile.ZipFile(\"PerconAI.zip\", \"r\") as zip_ref:\n",
            "        zip_ref.extractall(\".\")\n",
            "\n",
            "# Fix Path in case you are running this from inside the 'notebooks' directory\n",
            "current_dir = os.getcwd()\n",
            "project_root = os.path.abspath(os.path.join(current_dir, '..')) if 'notebooks' in current_dir else current_dir\n",
            "if project_root not in sys.path:\n",
            "    sys.path.insert(0, project_root)\n",
            "\n",
            "import torch\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import psutil"
        ]

with open("notebooks/PerconAI_Master.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Fixed notebook path logic")
