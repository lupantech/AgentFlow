## Benchmarking Guide
Here we provide a detailed benchmarking guide to reproduce the paper’s results across ten benchmarks.

### Serving Models with VLLM

We provide an automated VLLM serving script that launches multiple models in parallel using tmux sessions. 

```bash
bash scripts/serve_vllm.sh
```

**Configuration**

You can configure the following parameters in [`scripts/serve_vllm.sh`](../../scripts/serve_vllm.sh):

| Parameter | Description | Default |
|-----------|-------------|---------|
| **MODEL** | Model path to serve (HuggingFace or local) | `"agentflow/AgentFlow-Planner-7B"` |
| **GPU** | GPU device ID(s) to use | `"0"` |
| **PORT** | VLLM serving port | `8000` |
| **TP** | Tensor-parallel-size | `1` |


### Running Benchmark Experiments
We provide a one-click script to run all benchmarks at once. It executes AgentFlow on these benchmarks, saves the outputs, and automatically invokes the LLM for evaluation:
```python
cd test
bash exp/run_all_models_all_datasets.sh
```

**Configuration**

You can also configure benchmark settings in [`test/exp/run_all_models_all_datasets.sh`](../../test/exp/run_all_models_all_datasets.sh).

Content of [`test/exp/run_all_models_all_datasets.sh`](../../test/exp/run_all_models_all_datasets.sh):
```python
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
    "8000:vllm-agentflow/AgentFlow-Planner-7B,AgentFlow-7B,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
)

DATA_FILE_NAME="data.json"

############################################

cd $PROJECT_DIR

# Loop through all models
for MODEL_SPEC in "${MODELS[@]}"; do
    # Parse model specification: port:modelname,label,enabled_tools,tool_engines
    PORT=$(echo "$MODEL_SPEC" | cut -d':' -f1)
    REST=$(echo "$MODEL_SPEC" | cut -d':' -f2-)

    # Split by comma to get individual fields
    IFS=',' read -r LLM LABEL ENABLED_TOOLS_RAW TOOL_ENGINE_RAW <<< "$REST"

    # Convert | separators to commas for tool configuration
    ENABLED_TOOLS=$(echo "$ENABLED_TOOLS_RAW" | tr '|' ',')
    TOOL_ENGINE=$(echo "$TOOL_ENGINE_RAW" | tr '|' ',')

    BASE_URL="http://localhost:${PORT}/v1"

    echo "========================================"
    echo "MODEL: $LLM"
    echo "LABEL: $LABEL"
    echo "BASE_URL: $BASE_URL"
    echo "ENABLED_TOOLS: $ENABLED_TOOLS"
    echo "TOOL_ENGINE: $TOOL_ENGINE"
    echo "========================================"

    # Loop through all tasks
    for TASK in "${TASKS[@]}"; do
        echo "============================="
        echo "Starting task: $TASK with model: $LABEL"
        echo "============================="

        DATA_FILE="$TASK/data/$DATA_FILE_NAME"
        LOG_DIR="$TASK/logs/$LABEL"
        OUT_DIR="$TASK/results/$LABEL"
        CACHE_DIR="$TASK/cache"

        mkdir -p "$LOG_DIR"
        mkdir -p "$OUT_DIR"

        # Define indices based on task
        case $TASK in
            bamboogle)
                indices=($(seq 0 124))   # 0~124
                ;;
            gaia)
                indices=($(seq 0 126))   # 0~126
                ;;
            aime24)
                indices=($(seq 0 29))
                ;;
            2wiki|gameof24|amc23|nq|musique|hotpotqa|popqa|medqa|gpqa)
                indices=($(seq 0 99))
                ;;
            *)
                indices=(0)  # dafault
                ;;
        esac

        # Skip already completed indices
        new_indices=()
        for i in "${indices[@]}"; do
            if [ ! -f "$OUT_DIR/output_$i.json" ]; then
                new_indices+=($i)
            else
                echo "Output file already exists: $OUT_DIR/output_$i.json"
            fi
        done
        indices=("${new_indices[@]}")
        echo "Final indices: ${indices[@]}"

        if [ ${#indices[@]} -eq 0 ]; then
            echo "All subtasks completed for $TASK with $LABEL."
        else
            # Define single task function
            echo "Using model: $LLM"
            run_task() {
                local i=$1
                echo "Running $TASK for index $i"
                python solve.py \
                --index $i \
                --task $TASK \
                --data_file $DATA_FILE \
                --llm_engine_name $LLM \
                --root_cache_dir $CACHE_DIR \
                --output_json_dir $OUT_DIR \
                --output_types direct \
                --enabled_tools "$ENABLED_TOOLS" \
                --tool_engine "$TOOL_ENGINE" \
                --max_time 300 \
                --max_steps 10 \
                --temperature 0.7 \
                --base_url "$BASE_URL" \
                2>&1 | tee "$LOG_DIR/$i.log"
                echo "Completed $TASK for index $i"
                echo "------------------------"
            }
            export -f run_task
            export TASK DATA_FILE LOG_DIR OUT_DIR CACHE_DIR LLM ENABLED_TOOLS TOOL_ENGINE BASE_URL

            echo "Starting parallel execution for $TASK..."
            parallel -j $THREADS run_task ::: "${indices[@]}"
            echo "All subtasks completed for $TASK with $LABEL."
        fi

        ############ Calculate Scores ############
        RESPONSE_TYPE="direct_output"
        python calculate_score_unified.py \
        --task_name $TASK \
        --data_file $DATA_FILE \
        --result_dir $OUT_DIR \
        --response_type $RESPONSE_TYPE \
        --output_file "finalresults_$RESPONSE_TYPE.json" \
        | tee "$OUT_DIR/finalscore_$RESPONSE_TYPE.log"

    done

    echo "========================================"
    echo "Completed all tasks for model: $LABEL"
    echo "========================================"
done

echo "============================="
echo "All models and tasks finished!"
echo "============================="
```

**Step:**

**1. Set Parallelism**
Set the data parallelism for inference (too high a value may exceed API thresholds).
```bash
THREADS=20  # Number of parallel workers
```

**2. Select Tasks**

Enable or disable tasks by commenting/uncommenting:
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
      "8000:vllm-agentflow/AgentFlow-Planner-7B,AgentFlow-7B,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
)
```

**Format:** `"port:model_path,label,Tool1|Tool2|Tool3,engine1|engine2|Default"`
- **port**: VLLM serving port
- **planner_model_path**: HuggingFace model path or local path (please add `vllm-` prefix if you serve planner model through vllm)
- **label**: Display name for results
- **tools**: Pipe-separated tool list with engine (e.g., `Tool1|Tool2`)
- **tool_engine**: Pipe-separated tool engine for tools

**Note**: For all agents except the `planner`, we use [a fixed LLM engine (Qwen-2.5-7B-Instruct)](https://github.com/lupantech/AgentFlow/blob/d557fbf49f2c88aafb3d06c9b155cf3266218629/agentflow/agentflow/models/planner.py#L19).



### Results Organization

After benchmark completion, results are organized in the following structure:

```
test/
└── {TASK_NAME}/              # e.g., aime24, gameof24, bamboogle
    ├── logs/
    │   └── {MODEL_LABEL}/     # e.g., AgentFlow-7B
    │       ├── 0.log          # Per-problem execution logs
    │       ├── 1.log
    │       └── ...
    ├── results/
    │   └── {MODEL_LABEL}/
    │       ├── final_results_direct_output.json    # Detailed per-problem analysis
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
