import json

with open("notebooks/PerconAI_Master.ipynb", "r") as f:
    nb = json.load(f)

new_cells = [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 11. DAY 13-15: JADE PREDICTOR TRAINING\n",
        "Train the meta-controller on the generated corpus."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from scripts.train_predictor import main as train_pred_main\n",
        "from scripts.train_fleet_jade import main as train_fleet_main\n",
        "import sys\n",
        "\n",
        "sys.argv = [\n",
        "    \"train_predictor.py\",\n",
        "    \"--split\", \"indomain\",\n",
        "    \"--model_type\", \"logistic\"\n",
        "]\n",
        "train_pred_main()\n",
        "\n",
        "sys.argv = [\"train_fleet_jade.py\"]\n",
        "train_fleet_main()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 12. DAY 16: FAST EVALUATION SANITY CHECK\n",
        "Test the pipeline end-to-end on a fast subset."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from scripts.eval_fast import main as eval_fast_main\n",
        "import pandas as pd\n",
        "import sys\n",
        "\n",
        "sys.argv = [\n",
        "    \"eval_fast.py\",\n",
        "    \"--n_subjects\", \"2\",\n",
        "    \"--n_seeds\", \"1\",\n",
        "    \"--output_dir\", \"tmp_eval_fast\"\n",
        "]\n",
        "eval_fast_main()\n",
        "\n",
        "fast_results = pd.read_parquet(\"tmp_eval_fast/results_fast.parquet\")\n",
        "display(fast_results.groupby(\"policy\")[[\"stream_acc\", \"adaptation_tax\", \"joule_regret\"]].mean())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 13. DAY 17: FULL EVALUATION (MOLAB ARRAY)\n",
        "Generate manifest and run full eval."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from scripts.generate_task_manifest import main as manifest_main\n",
        "from scripts.eval_worker import main as eval_worker_main\n",
        "from scripts.aggregate_eval_results import main as agg_eval_main\n",
        "from pathlib import Path\n",
        "import sys\n",
        "\n",
        "sys.argv = [\"generate_task_manifest.py\"]\n",
        "manifest_main()\n",
        "\n",
        "# Note: Running all tasks linearly takes hours! Running just 2 for testing.\n",
        "Path(\"data/results/raw_eval_chunks\").mkdir(parents=True, exist_ok=True)\n",
        "for task_id in range(2):\n",
        "    sys.argv = [\n",
        "        \"eval_worker.py\",\n",
        "        \"--task_id\", str(task_id),\n",
        "        \"--manifest_path\", \"data/results/task_manifest.csv\",\n",
        "        \"--output_dir\", \"data/results/raw_eval_chunks\"\n",
        "    ]\n",
        "    eval_worker_main()\n",
        "\n",
        "sys.argv = [\"aggregate_eval_results.py\"]\n",
        "agg_eval_main()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 14. DAY 18-19: STATS & VISUALIZATION\n",
        "Generate figures and tables natively."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from scripts.analyze_results import main as analyze_main\n",
        "from scripts.generate_figures import main as figures_main\n",
        "from scripts.generate_tables import main as tables_main\n",
        "import sys\n",
        "\n",
        "sys.argv = [\"analyze_results.py\"]\n",
        "analyze_main()\n",
        "\n",
        "sys.argv = [\"generate_figures.py\"]\n",
        "figures_main()\n",
        "\n",
        "sys.argv = [\"generate_tables.py\"]\n",
        "tables_main()\n",
        "\n",
        "from IPython.display import Image, display\n",
        "display(Image(filename='figures/fig3_pareto.png'))\n",
        "display(Image(filename='figures/fig5_breakeven.png'))"
      ]
    }
]

nb["cells"].extend(new_cells)

with open("notebooks/PerconAI_Master.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Updated PerconAI_Master.ipynb")
