import math
import os
import string
import re
from typing import Any, Optional

import sympy

from autogen_ext.tools.mcp import StdioServerParams
from agentflow import Trainer, LitAgent, NamedResources, LLM, reward, configure_logger, DevTaskLoader

from agentflow.solver import construct_solver
from datetime import datetime
import uuid, json
from filelock import FileLock
import asyncio

from utils import compute_score

configure_logger()


@reward
async def eval(question: str, groundtruth: any, answer_extracted: any, val: bool = False) -> float:
    """
    Evaluates if the extracted answer is correct by calling an LLM judge (gpt-4o).
    It strip(), and matches the final answer.
    """
    question_str = str(question)
    groundtruth_str = str(groundtruth)
    answer_extracted_str = str(answer_extracted)

    is_correct = compute_score(question_str, groundtruth_str, answer_extracted_str)
    
    return 1.0 if is_correct else 0.0

class AgentFlowRollout:
    def __init__(
        self,
        resources: NamedResources,
        llm_engine_name: str = "gpt-4o",
        enabled_tools: list[str] = ["Base_Generator_Tool"],
        tool_engine: list[str] = ["Default"],
        model_engine: list[str] = ["trainable", "gpt-4o", "gpt-4o"],  # [planner_main, planner_fixed, executor]
        output_types: str = "final,direct",
        max_steps: int = 3,
        max_time: int = 500,
        max_tokens: int = 2048,
        base_url="http://localhost:8888",
        verbose: bool = True,
        temperature: float = 0.0,
    ):
        assert len(tool_engine)==len(enabled_tools)
        print(f"********MODEL {llm_engine_name} SERVED AT {base_url}***********")
        self.resources = resources
        self.llm_engine = llm_engine_name
        prefix = "" if "gpt" in llm_engine_name else "vllm-"
        self.solver = construct_solver(
            llm_engine_name=prefix + llm_engine_name,
            enabled_tools=enabled_tools,
            tool_engine=tool_engine,
            model_engine=model_engine,
            output_types=output_types,
            max_steps=max_steps,
            max_time=max_time,
            max_tokens=max_tokens,
            base_url=base_url,
            verbose=verbose,
            temperature = temperature
        )
        self.verbose = verbose

    def solve(self, question: str, image_path: Optional[str] = None) -> dict:
        result = self.solver.solve(question, image_path)
        if self.verbose:
            print(f"\n==> 📝 Solver Result:")
            print(f"""
            *******************************
            RESULT
            {result}
            RESULT
            *******************************
            """)

        return result


def get_agent(
    model: str,
    openai_base_url: str,
    temperature: float,
    resources,
    tools: list[str],
    max_steps: int,
    tool_engine: str,
    model_engine: list[str],
    max_tokens: int,
    output_type: str,
    timeout: int,
):
    llm_engine_name = model
    if openai_base_url and openai_base_url != "https://api.openai.com/v1":
        vllm_base_url = openai_base_url
    else:
        vllm_base_url = None

    # Note: `output_types`, `max_time`, `verbose` are set to constant values here.
    # If these need to be dynamic, you would also need to add them to the function parameters.
    agent = AgentFlowRollout(
        resources=resources,
        llm_engine_name=llm_engine_name,
        enabled_tools=tools,
        tool_engine=tool_engine,
        model_engine=model_engine,
        max_steps=max_steps,
        max_tokens=max_tokens,
        base_url=vllm_base_url,
        verbose=True,
        output_types=output_type,
        max_time=timeout,
        temperature=temperature
    )
    return agent


class Rollout(LitAgent):

    def __init__(self,
    server_public_ip: str = "Default",
    exp_name: str = "agent_flow_exp",
    rollout_n: int = 8,
    batch_size: int = 16,
    enabled_tools: list[str] =["Base_Generator_Tool","Python_Coder_Tool","Google_Search_Tool","Wikipedia_Search_Tool"],
    tool_engine: list[str] = ["gpt-4o","gpt-4o","Default","Default"],
    model_engine: list[str] = ["trainable", "gpt-4o", "gpt-4o"],  # [planner_main, planner_fixed, executor]
    max_steps: int = 3,
    max_tokens: int = 2048,
    train_temperature: float = 0.7,
    test_temperature: float = 0.0,
    output_type: str = "direct",
    timeout: int = 300,
    ):
        super().__init__()
        self.server_public_ip=server_public_ip
        # Agents will be initialized on the first call to their respective rollouts.
        self.training_agent = None
        self.validation_agent = None
        self.val_step_n = None

        self.output_type=output_type
        self.timeout=timeout

        self.rollout_dir = None
        self.train_rollout_dir = None
        self.val_rollout_dir = None
        self.train_lock_file = None
        self.val_lock_file = None

        self.train_temperature=train_temperature
        self.test_temperature=test_temperature

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_rollout_dir = f"./rollout_data/{self.server_public_ip}/{exp_name}_{timestamp}"
        self.tools = enabled_tools
        self.tool_engine = tool_engine
        self.model_engine = model_engine
        self._solve_call_count = 0

        self.run_info_file = os.path.join(self.base_rollout_dir, ".run_info")
        self.init_lock_file = os.path.join(self.base_rollout_dir, ".init.lock")

        # Added locks and state variables for async-safe step management.
        self.train_batch_size = batch_size # As defined in the original code logic
        self.rollout_num = rollout_n # As defined in the original code logic
        self.max_steps = max_steps
        self.max_tokens = max_tokens

    async def _solve_and_evaluate(self, rollout: AgentFlowRollout, task: Any, step_n: int, val: bool = False):
        """A helper function to run the agent, parse the result, and evaluate it."""
        result = {}
        try:
            output_format = "When ready, output the final answer enclosed in <answer> and </answer> tags. Do not generate any content after the </answer> tag."
            prompt = task["question"] + " " + output_format
            # prompt = task["question"]
            result = rollout.solve(question=prompt)
            
            # Safely check for and extract the final answer
            if "direct_output" in result and result["direct_output"]:
                final_output = result["direct_output"]
                all_matches = re.findall(r"<answer>(.*?)</answer>", final_output, re.DOTALL)
                if all_matches:
                    answer = all_matches[-1].strip()
                else:
                    answer = final_output
            else:
                print("Warning: Result has no direct_output or direct_output is empty.")
                answer = "None"
        except Exception as e:
            print(f"Failure during agent execution: {str(e)}. Defaulting to 'None'.")
            answer = "None"

        # Evaluate the answer against the ground truth
        reward_value = await eval(task["question"], str(task["result"]), answer, val)  # reward is tracked with the decorator
        print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward_value))

        idx = task.get("extra_info", {}).get("idx", "unknown_idx")

        rollout_data = {
            "step": task.get("step", ""), # TODO: check whether it can be solved
            "idx": idx,
            "id": task.get("id", ""),
            "prompt": task["question"],
            "model":rollout.llm_engine,
            "tools":self.tools,
            "groundtruth": task.get("extra_info", {}).get("groundtruth", task["result"]),
            "answer_extracted": answer,
            "reward": reward_value,
            "total_result":result,
            "timestamp": datetime.now().isoformat(),
        }

        data_id = str(uuid.uuid4())
        filename = f"rollout_{data_id}.json"

        save_dir = self.val_rollout_dir if val else self.train_rollout_dir

        # This function now uses the `step_n` passed as an argument.
        step_dir = os.path.join(save_dir, f"step_{step_n}")
        
        idx_dir = os.path.join(step_dir, f"idx_{idx}")
        os.makedirs(idx_dir, exist_ok=True)

        json_count = sum(
            len([f for f in files if f.endswith(".json")])
            for root, dirs, files in os.walk(idx_dir)
        )
        assert json_count < self.rollout_num, \
            f"Too many rollouts for idx {idx}: already {json_count} >= {self.rollout_num}"

        save_path = os.path.join(idx_dir, filename)

        with open(save_path, "w") as f:
            json.dump(rollout_data, f, indent=2)

        print(f"Rollout data saved to: {save_path}")


    async def _initialize_run_once(self, resources: NamedResources):
        """
        Ensures that the rollout directory is set up only once per run,
        in a process-safe way.
        """
        if self.rollout_dir is not None:
            return

        os.makedirs(self.base_rollout_dir, exist_ok=True)
        
        init_lock = FileLock(self.init_lock_file, timeout=50)
        with init_lock:
            if os.path.exists(self.run_info_file):
                with open(self.run_info_file, 'r') as f:
                    final_rollout_dir = f.read().strip()
            else:
                model_name = resources.get("main_llm").model
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") 
                model_name = model_name.rsplit('/', 1)[-1]
                final_rollout_dir = os.path.join(
                    self.base_rollout_dir, f"{model_name}_{timestamp}"
                )
                
                with open(self.run_info_file, 'w') as f:
                    f.write(final_rollout_dir)
                print(f"Run directory created by process {os.getpid()}: {final_rollout_dir}")

        self.rollout_dir = final_rollout_dir
        self.train_rollout_dir = os.path.join(self.rollout_dir, "train")
        self.val_rollout_dir = os.path.join(self.rollout_dir, "validation")
        
        os.makedirs(self.train_rollout_dir, exist_ok=True)
        os.makedirs(self.val_rollout_dir, exist_ok=True)
        
        self.train_lock_file = os.path.join(self.train_rollout_dir, ".train.lock")
        self.val_lock_file = os.path.join(self.val_rollout_dir, ".val.lock")
        
    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources, val: bool = False) -> Any:
        await self._initialize_run_once(resources)

        if self.training_agent is None:
            print("Initializing training agent...")
            llm: LLM = resources.get("main_llm")
            self.training_agent = get_agent(
                llm.model,
                llm.endpoint,
                temperature=self.train_temperature,
                tools = self.tools,
                max_steps = self.max_steps,
                tool_engine = self.tool_engine,
                model_engine = self.model_engine,
                resources = resources,
                max_tokens = self.max_tokens,
                output_type= self.output_type,
                timeout= self.timeout,
            )
        
        # filelock to determine step_n ---
        lock = FileLock(self.train_lock_file, timeout=30)
        with lock:
            step_dirs = [d for d in os.listdir(self.train_rollout_dir) if d.startswith("step_")]
            step_nums = [int(d.replace("step_", "")) for d in step_dirs if d.replace("step_", "").isdigit()]
            
            current_step_n = 1
            if step_nums:
                current_step_n = max(step_nums)

            current_step_dir = os.path.join(self.train_rollout_dir, f"step_{current_step_n}")
            if os.path.exists(current_step_dir):
                num_items_in_step = len(os.listdir(current_step_dir))
                if num_items_in_step >= self.train_batch_size:
                    current_step_n += 1
            
            step_n = current_step_n

        await self._solve_and_evaluate(self.training_agent, task, step_n, val)



    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources, val: bool = True) -> Any:
        await self._initialize_run_once(resources)

        # Lazy initialization of the agent and one-time determination of the validation step number.
        # This lock ensures that only the first validation task of a run calculates the step number,
        # preventing the creation of thousands of folders.
        val_lock = FileLock(self.val_lock_file, timeout=50)
        with val_lock:
            if self.validation_agent is None:
                print("Initializing validation agent and determining validation step...")
                llm: LLM = resources.get("main_llm")
                self.validation_agent = get_agent(
                    llm.model,
                    llm.endpoint,
                    temperature=self.test_temperature,
                    tools = self.tools,
                    max_steps = self.max_steps,
                    tool_engine = self.tool_engine,
                    model_engine = self.model_engine,
                    resources = resources,
                    max_tokens = self.max_tokens,
                    output_type=self.output_type,
                    timeout=self.timeout,
                )

            print(f"Scanning '{self.train_rollout_dir}' to find current training step...")
            train_step_dirs = [d for d in os.listdir(self.train_rollout_dir) if d.startswith("step_")]
            train_step_nums = [int(d.replace("step_", "")) for d in train_step_dirs if d.replace("step_", "").isdigit()]
            
            current_train_step = max(train_step_nums) if train_step_nums else 0
            self.val_step_n = current_train_step
            print(f"Validation run started. Synchronizing with training progress. Saving results to validation step folder: {self.val_step_n}")

        await self._solve_and_evaluate(self.validation_agent, task, self.val_step_n, val)

if __name__ == "__main__":
    from util.parse_config import get_values_from_yaml
    from util.port_cleanup import kill_process_on_port
    from util.get_pub_ip import get_public_ip_with_fallback
    from pprint import pprint

    server_public_ip = get_public_ip_with_fallback()

    keys_to_retrieve = [
        "EXPERIMENT_NAME",
        'data.train_batch_size',
        'actor_rollout_ref.rollout.n',
        'agentflow.port',
        'N_WORKERS',
        'ENABLE_TOOLS',
        'TOOL_ENGINE',
        'MODEL_ENGINE',
        "TOOL_STEPS",
        "TRAIN_TEMPERATURE",
        "TEST_TEMPERATURE",
        "data.max_response_length",
        "OUTPUT_TYPE",
        "AGENT_MAX_TIMEOUT"
    ]

    config_file = 'train/config.yaml'

    values = get_values_from_yaml(config_file, keys_to_retrieve)

    config_keys_map = {
        "EXPERIMENT_NAME": "exp_name",
        "data.train_batch_size": "batch_size",
        "actor_rollout_ref.rollout.n": "rollout_n",
        "agentflow.port": "port",
        "N_WORKERS": "n_workers",
        "ENABLE_TOOLS": "enabled_tools",
        "TOOL_ENGINE": "tool_engine",
        "MODEL_ENGINE": "model_engine",
        "TOOL_STEPS": "max_steps",
        "TRAIN_TEMPERATURE": "train_temperature",
        "TEST_TEMPERATURE": "test_temperature",
        "data.max_response_length": "max_tokens",
        "OUTPUT_TYPE": "output_type",
        "AGENT_MAX_TIMEOUT": "timeout",
    }

    config_dict = dict(zip(config_keys_map.values(), values))

    port_to_use = config_dict.get("port")
    if port_to_use:
        print(f"[INFO] Checking and freeing port {port_to_use}...")
        kill_process_on_port(port_to_use)
    else:
        print("[WARNING] No port specified in config, skipping port cleanup.")

    print("Agent params:")
    pprint(config_dict, indent=2, width=80, compact=True)

    trainer = Trainer(n_workers=config_dict["n_workers"])
    agent = Rollout(server_public_ip=server_public_ip, **{k: v for k, v in config_dict.items() if k != "n_workers" and k != "port"})
    trainer.fit(agent, f"http://localhost:{config_dict['port']}/")