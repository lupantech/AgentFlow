## Benchmarking Guide
Here we provide a detailed benchmarking guide to reproduce the paper’s results across ten benchmarks.

### Serving Models with VLLM

We provide an automated VLLM serving script in [`scripts/serve_vllm.sh`](../../scripts/serve_vllm.sh). 

```bash
bash scripts/serve_vllm.sh
```

**Configuration**

You can configure the following parameters in [`scripts/serve_vllm.sh`](../../scripts/serve_vllm.sh):

| Parameter | Description | Default                            |
|-----------|-------------|------------------------------------|
| MODEL | Model path to serve (HuggingFace or local) | `"AgentFlow/agentflow-planner-7b"` |
| GPU | GPU device ID(s) to use | `"0"`                              |
| PORT | VLLM serving port | `8000`                             |
| TP | Tensor-parallel-size | `1`                                |


### Running Benchmark Experiments
We provide an one-click script to run all benchmarks at once. It executes our agentic system on these benchmarks, saves the outputs, and automatically invokes the LLM for evaluation:
```bash
cd test
bash exp/run_all_models_all_datasets.sh
```

**Configuration**

You can configure benchmark settings in [`test/exp/run_all_models_all_datasets.sh`](../../test/exp/run_all_models_all_datasets.sh).

Partial Content of [`test/exp/run_all_models_all_datasets.sh`](../../test/exp/run_all_models_all_datasets.sh):
```bash
#!/bin/bash
# Usage: bash exp/run_all_models_all_datasets.sh
# Model format: "port:modelname,label,enabled_tools,tool_engines"
# Example: "8000:vllm-IPF/model1,label1,Tool1|Tool2,engine1|engine2"

############ Configuration ############
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set project directory to test/ folder (parent of exp/)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
THREADS=20

# Define all tasks to run
TASKS=(
    # "gameof24"
    "aime24"
    # "amc23"
    # "bamboogle"
    # "2wiki"
    # "gaia"
    # "musique"
    # "hotpotqa"
    # "medqa"
    # "gpqa"
)

# Define models with their tool configurations in format:
# "port:modelname,label,enabled_tools,tool_engines"
# - enabled_tools: use | as separator (will be converted to comma)
# - tool_engines: use | as separator (will be converted to comma)
# Example: "8000:vllm-IPF/model,label,Tool1|Tool2|Tool3,engine1|engine2|Default"
MODELS=(
    "8000:vllm-AgentFlow/agentflow-planner-7b,AgentFlow-7B,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
)
```

**Step:**

**1. Set Parallelism**
Set the data parallelism for inference (too high a value may exceed API thresholds).
```bash
THREADS=20  # Number of parallel workers
```

**2. Select Tasks**

Enable or disable benchmarks by commenting/uncommenting:
```bash
TASKS=(
    "aime24"
    "gameof24"
    "bamboogle"
    # "gpqa"
)
```

**3. Define Models**

Specify models with their configurations:
```bash
MODELS=(
      "8000:vllm-AgentFlow/agentflow-planner-7b,AgentFlow-7B,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
)
```

**Format:** `"port:model_path,label,Tool1|Tool2|Tool3,engine1|engine2|Default"`
- **port**: VLLM serving port
- **planner_model_path**: HuggingFace model path or local path (please add `vllm-` prefix if you serve planner model through vllm. If you want to use other model, please refer to [llm_engine.md](llm_engine.md))
- **label**: Display name for results
- **tools**: Pipe-separated tool list with engine (e.g., `Tool1|Tool2`)
- **tool_engine**: Pipe-separated tool engine for tools (If you want to use other tool_engine, please refer to [llm_engine.md](llm_engine.md))

**Note**: For all agents except the `planner`, we use [a fixed LLM engine (Qwen-2.5-7B-Instruct)](https://github.com/lupantech/AgentFlow/blob/d557fbf49f2c88aafb3d06c9b155cf3266218629/agentflow/agentflow/models/planner.py#L19).



### Results Organization

After benchmark completion, results are organized in the following structure:

```
test/
└── {TASK_NAME}/              # e.g., aime24, bamboogle
    ├── logs/
    │   └── {MODEL_LABEL}/     # e.g., AgentFlow-7B
    │       ├── 0.log          # Per-problem execution logs
    │       ├── 1.log
    │       └── ...
    ├── results/
    │   └── {MODEL_LABEL}/
    │       ├── final_results_direct_output.json    # Per-problem analysis
    │       ├── final_scores_direct_output.json    # Aggregate metrics
    │       ├── final_score_direct_output.log       # Scoring process log
    │       ├── output_0.json                      # Individual outputs
    │       ├── output_1.json
    │       └── ...
    └── cache/                 # Cached intermediate results
```

### Key Result Files

| File | Description |
|------|-------------|
| `final_scores_direct_output.json` | Aggregate metrics: accuracy, correct/wrong counts, tool usage statistics |
| `final_results_direct_output.json` | Detailed per-problem results with verification and analysis |
| `output_{i}.json` | Complete execution trace: query, response, memory, tool calls |
| `final_score_direct_output.log` | Detailed scoring process log |
