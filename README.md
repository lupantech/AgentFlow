# AgentFlow 

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/imgs/logo.png">
    <img alt="VerlTool" src="assets/imgs/logo.png" width=20%>
  </picture>
</p>

<h3 align="center">
AgentFlow: In-The-Flow Agentic System Optimization for Effective Planning and Tool Use.
</h3>

<p align="center">
| 
<a href=""><b>Website</b></a> |
<a href=""><b>Paper</b></a> |
<a href="https://huggingface.co/agentflow"><b>Huggingface</b></a> |
<a href=""><b>Twitter</b></a> |
  <a href="https://deepwiki.com/lupantech/AgentFlow"><b>DeepWiki</b></a> |
  <a href=""><b>WeChat Group</b></a> |
  <a href="https://join.slack.com/t/agentflowco/shared_invite/zt-3f1bmai74-1CaZfpgkhRU061lYaH4zqQ"><b>Slack</b></a>
|
</p>

AgentFlow is a trainable, tool-integrated agentic framework that addresses the scalability and generalization limitations of current tool-integrated reasoning approaches. 

Unlike prevailing methods, like Search-R1 that train a single LLM interleaving thoughts and tool calls, AgentFlow provides a modular agentic system with four specialized modules (planner, executor, verifier, generator) that coordinate through evolving memory and a toolkit over multiple turns to solve complex reasoning tasks.

![framework_overall](assets/img/framework.png)

For effective planning and tool use, the framework directly optimizes the planner agent within the system in an online fashion using Flow-based Group Refined Policy Optimization (Flow-GRPO), achieving superior performance across diverse domains with improved tool-calling reliability and long-horizon reasoning capabilities.

![flow_grpo](assets/img/flow_grpo.png)

## Key Features
+ **Modular Agentic System**: Powerful Agentic System with four specialized modules (planner, executor, verifier, generator) augmented with tools that coordinate through evolving memory across multiple turns.
+ **Multi-Tool Integration**: Seamless integration with diverse tool ecosystems, including base_generator, python_coder, google_search, wikipedia_search, and web_search.
+ **Flow-GRPO Algorithm**: Novel Flow-based Group Refined Policy Optimization that enables in-the-flow optimization within agentic systems under long-horizon reasoning tasks with sparse reward.

## Experiments
### Main results
Through comprehensive experiments on ten benchmarks, AgentFlow with a 7B-scale backbone (Qwen-2.5-7B-Instruct) outperforms top-performing baselines with average accuracy gains of 14.9% on search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks. Notably, our 7B-backbone system even surpasses the ∼200B-parameter GPT-4o. 

![main_table](assets/img/main_table.png)

### In-depth analysis
Further analyses confirm the benefits of in-the-flow optimization,
demonstrating improved planning, enhanced tool-calling reliability, and positive
scaling trends with model size and reasoning turns. Please explore more findings at our paper or the project page.

![tool_call](assets/img/tool_call.png)


## Setup
### Installation
```bash
bash setup.sh
source .venv/bin/activate
# (Optional) Install `parallel` for running benchmark experiments in parallel:
sudo apt-get update
sudo apt-get install parallel
```

### Setup Environment Variables
Duplicate the `.env.template` file and rename it to `.env`. Next, update the variables (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CX`, `DASHSCOPE_API_KEY`) with your own keys.  
```
cp .env_template .env
```

## Quick Start
### (Optional) Test Your Environment
Before diving in, we recommend verifying that AgentFlow's tools, LLM engines, and network configuration are properly set up. See [test_env.md](assets/doc/test_env.md) for detailed testing instructions.

### Dataset Preparation
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
### AgentFlow Train
Start agentflow training with tmux:
```bash
# Create tmux session and start agentflow service (Window 0)
tmux new-session -s agentflow
bash train/serve_with_logs.sh

# Create new window (Ctrl+B then C) and start training (Window 1)
bash train/train_with_logs.sh
```
**Configuration:**
All training hyperparameters are in [`train/config.yaml`](train/config.yaml) (model settings, tools, RL parameters, resources, etc.)

### AgentFlow Inference
Serve the trained planner model with VLLM (here we deploy our 7B Flow-GRPO planner model):
```bash
bash scripts/serve_vllm.sh
```

Run inference on benchmark tasks:
```bash
cd test
bash exp/run_all_models_all_datasets.sh
```

## Use your own model in AgentFlow 
+ Planner 
+ 其他 model 应该怎么修改 


## Core Contributors

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

## Advisors

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

## Acknowledgements

We thank the following open-source projects:
- [verl](https://github.com/volcengine/verl) for the excellent RL framework design.
- [VLLM](https://github.com/vllm-project/vllm) for fast LLM inference support.
- [agent-lightning](https://github.com/microsoft/agent-lightning) for early-stage exploration in agentic RL Training. 

We thank [Lambda](https://lambda.ai/careers) for GPU support!

## Contributors

We are truly looking forward to open-source contributions to AgentFlow!  If you’re interested in contributing, collaborating, or reporting issues, please feel free to open an issue or submit a pull request (PR).  You can also reach us at [zhuofengli12345@gmail.com](mailto:zhuofengli12345@gmail.com) or join our Slack community: [AgentFlow](https://join.slack.com/t/agentflowco/shared_invite/zt-3f1bmai74-1CaZfpgkhRU061lYaH4zqQ).


We are also looking forward to your feedback and suggestions!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lupantech/AgentFlow&type=Date)](https://www.star-history.com/#lupantech/AgentFlow&Date)

<p align="right" style="font-size: 14px; color: #2176bc; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
























