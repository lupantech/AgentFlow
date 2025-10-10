<a name="readme-top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/img/logo.png">
    <img alt="AgentFlow" src="assets/img/logo.png" width=31%>
  </picture>
</p>

<h3 align="center">
AgentFlow: In-the-Flow Agentic System Optimization for Effective Planning and Tool Use
</h3>


<!--- BADGES: START --->
<p align="center">
    <a href="https://arxiv.org/abs/2510.05592"><img src="https://img.shields.io/badge/arXiv-2510.05592-B31B1B.svg?logo=arxiv" alt="Arxiv"></a>
    <a href="https://huggingface.co/spaces/AgentFlow/agentflow"><img src="https://img.shields.io/badge/Gradio-Demo-F97316.svg?logo=gradio" alt="Gradio Demo"></a>
    <a href="https://huggingface.co/papers/2510.05592"><img src="https://img.shields.io/badge/Huggingface-Paper-FFD21E.svg?logo=huggingface" alt="Huggingface Paper"></a>
    <a href="https://huggingface.co/AgentFlow"><img src="https://img.shields.io/badge/Huggingface-Model-FFD21E.svg?logo=huggingface" alt="Huggingface Model"></a>
    <a href="https://agentflow.stanford.edu/"><img src="https://img.shields.io/badge/Website-AgentFlow-E5426E?logo=kashflow" alt="Website"></a>
    <a href="https://x.com/lupantech/status/1976016000345919803"><img src="https://img.shields.io/badge/Coverage-AgentFlow-2176BC.svg?logo=x" alt="X"></a>
    <a href="https://join.slack.com/t/agentflow-co/shared_invite/zt-3f712xngl-LfxS4gmftAeKvcxR3nSkWQ"><img src="https://img.shields.io/badge/Slack-AgentFlow-D41544.svg?logo=slack" alt="Slack"></a>
    <a href="https://github.com/lupantech/AgentFlow/blob/main/assets/img/wechat_group.jpg">
  <img src="https://img.shields.io/badge/Wechat-AgentFlow-07C160.svg?logo=wechat" alt="Wechat AgentFlow">
</a>
  
  </p>
<!--- BADGES: END --->


## 🌟 Why AgentFlow?
AgentFlow is a **trainable, tool-integrated agentic framework** designed to overcome the **scalability** and **generalization limits** of today’s tool-augmented reasoning approaches.  

Unlike prevailing approaches such as [Search-R1](https://github.com/PeterGriffinJin/Search-R1) which train a **single LLM** to interleave reasoning steps with tool calls, **AgentFlow** introduces a **modular agentic system** with four specialized modules: 🧭 **Planner**, 🛠 **Executor**, ✅ **Verifier**, and ✍️ **Generator**.

![framework_overall](assets/img/framework.png)

For effective planning and tool use, the framework directly **optimizes planner agent within the system** in an **online fashion** using **Flow-based Group Refined Policy Optimization (Flow-GRPO)**, achieving superior performance across diverse domains with improved tool-calling reliability and long-horizon reasoning capabilities.

![flow_grpo](assets/img/flow_grpo.png)

## 🚀 Key Features

- 🧩 **Modular Agentic System** – Four specialized agent modules (**Planner**, **Executor**, **Verifier**, **Generator**) that coordinate via evolving memory and integrated tools across multiple turns.  
- 🔗 **Multi-Tool Integration** – Seamlessly connect with diverse tool ecosystems, including `base_generator`, `python_coder`, `google_search`, `wikipedia_search`, `web_search`, and more.  
- 🎯 **Flow-GRPO Algorithm** – Enables **in-the-flow agent optimization** for **long-horizon reasoning tasks** with sparse rewards.
- 📈 **Proven Results** – **AgentFlow (7B Backbone)** beats top baselines on 10 benchmarks, with **+14.9% search**, **+14.0% agentic**, **+14.5% math**, **+4.1% science**, even outperforming ~200B-parameter **GPT-4o**.

## 🏆 Experiments

### 📊 Main Results
**AgentFlow (Qwen-2.5-7B-Instruct Backbone)** outperforms top baselines on 10 benchmarks:  
- **+14.9%** on search  
- **+14.0%** on agentic reasoning  
- **+14.5%** on math  
- **+4.1%** on science  

💡 Even surpasses larger proprietary models like **GPT-4o (~200B)**.

![main_table1](assets/img/maintable1.png)
![main_table2](assets/img/maintable2.png)

### 🔍 In-Depth Analysis
- Improved planning and decision-making  
- Enhanced tool-calling reliability  
- Positive scaling trends with model size & reasoning turns  

Explore more in our [paper](https://arxiv.org/abs/2510.05592) or [project page](https://agentflow.stanford.edu/).

![tool_call](assets/img/tool_call.png)

---

## 📑 Table of Contents
- [⚙️ Setup](#️-setup)
  - [Installation](#installation)
  - [Setup Environment Variables](#setup-environment-variables)
- [⚡ Quick Start on AgentFlow Inference](#-quick-start-on-agentflow-inference)
- [💥 Quick Start on AgentFlow Flow-GRPO Training](#-quick-start-on-agentflow-flow-grpo-training)
  - [(Optional) Test Your Environment](#optional-test-your-environment)
  - [Dataset Preparation](#dataset-preparation)
  - [Flow-GRPO Training](#flow-grpo-training)
- [🎯 AgentFlow Benchmark](#-agentflow-benchmark)
- [🧩 Use Your Own Model in AgentFlow](#-use-your-own-model-in-agentflow)
- [🤝 Core Contributors](#-core-contributors)
- [🎓 Advisors](#-advisors)
- [🙏 Acknowledgements](#-acknowledgements)
- [🚀 Contributing](#-contributing)

## ⚙️ Setup
### Installation
```bash
bash setup.sh
source .venv/bin/activate
# (Optional) Install `parallel` for running benchmark experiments in parallel:
sudo apt-get update
sudo apt-get install parallel
```

### Setup Environment Variables
Copy the `.env.template` file from `agentflow/.env.template` and rename it to `.env`, then place it in the `agentflow/` folder. Update the following variables with your own API keys:
- `OPENAI_API_KEY` (for judging reasponse)
- `GOOGLE_API_KEY` (for Google Search tool)
- `DASHSCOPE_API_KEY` (for calling Qwen-2.5-7B-Instruct as engine for agents and tools)
- `TOGETHER_API_KEY` (alternative for calling Qwen-2.5-7B-Instruct as engine for agents and tools - recommended for international users)
- More ways: serve Qwen2.5-7B-instruct model with vLLM (details refer to [`serve_vllm_local.md`](assets/doc/serve_vllm_local.md)).

Please check [API Key Setup Guide](assets/doc/api_key.md) for detailed instructions on how to obtain these keys.

```bash
cp agentflow/.env.template agentflow/.env
# Then edit agentflow/.env with your API keys
```

## ⚡ Quick Start on AgentFlow Inference 
AgentFlow provides a modular agentic system with **four specialized modules** (planner, executor, verifier, generator) that coordinate through **evolving memory** and a **toolkit** over **multiple turns** to solve complex reasoning tasks. 

To quickly experience the system in action, run the command below (don’t forget to set up your API key):
```python 
python quick_start.py
```
Here is the content of `quick_start.py`:
```python
# Import the solver
from agentflow.agentflow.solver import construct_solver

# Set the LLM engine name
llm_engine_name = "dashscope"

# Construct the solver
solver = construct_solver(llm_engine_name=llm_engine_name)

# Solve the user query
output = solver.solve("What is the capital of France?")
print(output["direct_output"])
```

## 💥 Quick Start on AgentFlow Flow-GRPO Training 
For effective planning and tool use, the framework directly **optimizes the planner agent within the system in an online fashion using Flow-GRPO**. Below is a quick start for training.

### (Optional) Test Your Environment
Before diving in, we recommend verifying that AgentFlow's tools, LLM engines, and network configuration are properly set up. See [test_env.md](assets/doc/test_env.md) for detailed testing instructions.


### Dataset Preparation
We mix two datasets for training: [NQ (Natural Questions)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) for agentic search and [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) for mathematical reasoning.

```bash
# train data
python data/get_train_data.py
# validation data
python data/aime24_data.py
```

After that, data dir should be:
```
data/
├── train/
│   └── combined_train.parquet (182,190 samples)
├── val/
│   └── aime24.parquet (30 samples)
├── aime24_data.py
└── get_train_data.py
```

### Flow-GRPO Training 
Start agentflow training using Flow-GRPO with tmux:
```bash
# Create tmux session and start agentflow service (Window 0)
tmux new-session -s agentflow
bash train/serve_with_logs.sh

# Create new window (Ctrl+B then C) and start training (Window 1)
bash train/train_with_logs.sh
```

**Configuration:**
All training hyperparameters are in [`train/config.yaml`](train/config.yaml) (model settings, tools, RL parameters, resources, etc.)

**Logging:**
We provide a comprehensive logging to monitor training. See [logs.md](assets/doc/logs.md) for more details.



## 🎯 AgentFlow Benchmark 
Serve the trained planner model with VLLM (here we deploy our [7B Flow-GRPO planner model](https://huggingface.co/AgentFlow/agentflow-planner-7b)):
```bash
bash scripts/serve_vllm.sh
```

Run inference on benchmark tasks:
```bash
cd test
bash exp/run_all_models_all_datasets.sh
```

You can find more benchmarking details in [benchmark.md](assets/doc/benchmark.md). 

## 🧩 Use Your Own Model in AgentFlow

AgentFlow supports different LLM engines for each agent module. See [llm_engine.md](assets/doc/llm_engine.md) for supported models and [`factory.py`](agentflow/agentflow/engine/factory.py) for the corresponding `model_string` configuration:

**Planner Agent:**
- Modify the `llm_engine_name` parameter in [`test/exp/run_all_models_all_datasets.sh`](test/exp/run_all_models_all_datasets.sh)

**Other Agents (Executor, Verifier, Generator):**
- By default, these agents use a fixed LLM engine (Qwen-2.5-7B-Instruct via DashScope)
- To use your own model, modify `self.llm_engine_fixed` in [`agentflow/agentflow/models/planner.py:19`](agentflow/agentflow/models/planner.py#L19):
```python
self.llm_engine_fixed = create_llm_engine(model_string="your-engine", is_multimodal=False, temperature=temperature)
```
and

- Modify the `llm_engine_name` parameter in the Executor instantiation from [`agentflow/agentflow/solver.py:232`](agentflow/agentflow/solver.py#L232):
```python
# Instantiate Executor
executor = Executor(
    # llm_engine_name=llm_engine_name,
    llm_engine_name="dashscope",
    root_cache_dir=root_cache_dir,
    verbose=verbose,
    # base_url=base_url,
    temperature=temperature
)
```
- For detailed information on supported engines and `model_string` formats, see [`llm_engine.md`](assets/doc/llm_engine.md)



## 🤝 Core Contributors

<table>
<tr>
    <td align="center">
        <a href="https://zhuofeng-li.github.io/">
            <img src="https://github.com/Zhuofeng-Li.png" width="75px;" alt="Zhuofeng Li"/>
            <br />
            <sub><b>Zhuofeng Li</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://isaacghx.github.io/about/">
            <img src="https://github.com/IsaacGHX.png" width="75px;" alt="Haoxiang Zhang"/>
            <br />
            <sub><b>Haoxiang Zhang</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://lupantech.github.io/">
            <img src="https://github.com/lupantech.png" width="75px;" alt="Pan Lu"/>
            <br />
            <sub><b>Pan Lu</b></sub>
        </a>
    </td>
</tr>
</table>

## 🎓 Advisors

<table>
<tr>
    <td align="center">
        <a href="https://www.james-zou.com/">
            <img src="https://static.wixstatic.com/media/0f3e8f_cfa7e327b97745ddb8c4a66454b5eb3e~mv2.jpg/v1/fill/w_398,h_557,al_c,q_80,usm_0.66_1.00_0.01,enc_avif,quality_auto/46824428A5822_ForWeb.jpg" width="65px;" alt="James Zou"/>
            <br />
            <sub><b>James Zou</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://yejinc.github.io/">
            <img src="https://yejinc.github.io/profile-uw-2022.jpeg" width="75px;" alt="Yejin Choi"/>
            <br />
            <sub><b>Yejin Choi</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://yuzhimanhua.github.io/">
            <img src="https://yuzhimanhua.github.io/profile_pic.jpg" width="90px;" alt="Yu Zhang"/>
            <br />
            <sub><b>Yu Zhang</b></sub>
        </a>
    </td>
</tr>
</table>

## 🙏 Acknowledgements

We thank the following open-source projects:
- [verl](https://github.com/volcengine/verl) for the excellent RL framework design.
- [VLLM](https://github.com/vllm-project/vllm) for fast LLM inference support.
- [Ver-Tool](https://github.com/TIGER-AI-Lab/verl-tool) and [agent-lightning](https://github.com/microsoft/agent-lightning) for their early-stage exploration in agentic RL Training. 

We thank [Lambda](https://lambda.ai/careers) for GPU support!

## 🚀 Contributing

We are truly looking forward to open-source contributions to AgentFlow!  If you’re interested in contributing, collaborating, or reporting issues, please feel free to open an issue or submit a pull request (PR).  You can also reach us at [zhuofengli12345@gmail.com](mailto:zhuofengli12345@gmail.com), [isaacpfino@gmail.com](mailto:isaacpfino@gmail.com), [lupantech@gmail.com](mailto:lupantech@gmail.com) or join our Slack community: [AgentFlow](https://join.slack.com/t/agentflow-co/shared_invite/zt-3f712xngl-LfxS4gmftAeKvcxR3nSkWQ).


We are also looking forward to your feedback and suggestions!

## 📚 Citation
```bibtex
@article{li2025intheflow,
    title = {In-the-Flow Agentic System Optimization for Effective Planning and Tool Use},
    author = {Li, Zhuofeng and Zhang, Haoxiang and Han, Seungju and Liu, Sheng and Xie, Jianwen and Zhang, Yu and Choi, Yejin and Zou, James and Lu, Pan},
    journal = {arXiv preprint arXiv:2510.05592},
    year = {2025}
}
```

<p align="right" style="font-size: 14px; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
