import json
with open("notebooks/PerconAI_Master.ipynb", "r") as f:
    nb = json.load(f)

# Find where "GENERATE META-TRAINING CORPUS" is
gen_idx = -1
for i, cell in enumerate(nb["cells"]):
    if "GENERATE META-TRAINING CORPUS" in "".join(cell["source"]):
        gen_idx = i
        break

if gen_idx != -1:
    new_cells = [
        {
          "cell_type": "markdown",
          "metadata": {},
          "source": [
            "# 10.5 PHASE 4: AGGREGATE CORPUS\n",
            "Combine chunks into the frozen meta-corpus."
          ]
        },
        {
          "cell_type": "code",
          "execution_count": None,
          "metadata": {},
          "outputs": [],
          "source": [
            "from scripts.aggregate_corpus import main as agg_main\n",
            "import sys\n",
            "sys.argv = [\n",
            "    \"aggregate_corpus.py\",\n",
            "    \"--input_dir\", \"/marimo/jade_data/corpus\",\n",
            "    \"--output_dir\", \"data/processed\"\n",
            "]\n",
            "agg_main()"
          ]
        }
    ]
    nb["cells"] = nb["cells"][:gen_idx+1] + new_cells + nb["cells"][gen_idx+1:]
    
with open("notebooks/PerconAI_Master.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
print("Updated notebook!")
