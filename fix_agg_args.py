import json

with open("notebooks/PerconAI_Master.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "aggregate_corpus.py" in "".join(cell["source"]):
        cell["source"] = [
            "from scripts.aggregate_corpus import main as agg_main\n",
            "import sys\n",
            "import os\n",
            "sys.argv = [\n",
            "    \"aggregate_corpus.py\",\n",
            "    \"--input_dir\", \"/marimo/jade_data/corpus\",\n",
            "    \"--output_dir\", \"data/processed\",\n",
            "    \"--hf_repo\", HF_REPO,\n",
            "    \"--hf_token\", os.environ[\"HF_TOKEN\"]\n",
            "]\n",
            "agg_main()"
        ]

with open("notebooks/PerconAI_Master.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

