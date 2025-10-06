import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from agentflow.agentflow.models.initializer import Initializer
from agentflow.agentflow.models.planner import Planner
from agentflow.agentflow.models.memory import Memory
from agentflow.agentflow.models.executor import Executor
from agentflow.agentflow.models.utils import make_json_serializable_truncated

class Solver:
    def __init__(
        self,
        planner,
        memory,
        executor,
        task: str,
        data_file: str,
        task_description: str,
        output_types: str = "base,final,direct",
        index: int = 0,
        verbose: bool = False,
        max_steps: int = 10,
        max_time: int = 60,
        max_tokens: int = 4000,
        output_json_dir: str = "results",
        root_cache_dir: str = "cache",
        temperature: float = 0.7
    ):
        self.planner = planner
        self.memory = memory
        self.executor = executor
        self.task = task
        self.data_file = data_file
        self.task_description = task_description
        self.index = index
        self.verbose = verbose
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.output_json_dir = output_json_dir
        self.root_cache_dir = root_cache_dir
        self.temperature = temperature

        self.output_types = output_types.lower().split(',')
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."

        self.benchmark_data = self.load_benchmark_data()

    def load_benchmark_data(self) -> List[Dict[str, Any]]:
        # Add task description to the query
        if self.task_description:
            print(f"Task description: {self.task_description}")
            self.task_description = f"Task description: {self.task_description}\n"

        with open(self.data_file, 'r') as f:
            data = json.load(f) 
        for problem in data:
            problem['query'] = problem['query'] if 'query' in problem else problem['question']
            if self.task_description:
                problem['query'] = self.task_description + problem['query']

            if 'image' in problem and problem['image'] not in [None, ""]:
                # NOTE: This is a hack to make the absolute image path relative to the data file
                problem['image'] = os.path.abspath(os.path.join(os.path.dirname(self.data_file), problem['image']))
                assert os.path.exists(problem['image']), f"Error: Image file {problem['image']} does not exist."

        return data

    def solve(self):
        total_problems = len(self.benchmark_data)

        # Solve a single problem
        if self.index is not None:
            if not 0 <= self.index < total_problems:
                print(f"Error: Invalid problem index {self.index}. Valid indices are 0 to {total_problems-1}).")
            else:
                self.solve_single_problem(self.index)
            return

    def solve_single_problem(self, index: int):
        """
        Solve a single problem from the benchmark dataset.
        
        Args:
            index (int): Index of the problem to solve
        """
        # Update cache directory for the executor
        _cache_dir = os.path.join(self.root_cache_dir, f"{index}")
        self.executor.set_query_cache_dir(_cache_dir)
    
        # Create output directory and file path
        json_dir = os.path.join(self.output_json_dir)
        os.makedirs(json_dir, exist_ok=True)
        output_file = os.path.join(json_dir, f"output_{index}.json")

        # Get the problem
        problem = self.benchmark_data[index]
        # use 'query' by default for LLM inputs
        question = problem.get("query") if "query" in problem else problem["question"]
        image_path = problem.get("image", None)
        print(f"image_path: {image_path}")  
        pid = problem['pid']
        answer = problem['answer']

        if self.verbose:
            print("\n\n")
            print("#"*100)
            print(f"## Problem {index}:")
            print(f"Question:\n{question}")
            print(f"Image: {image_path}")
            print("#"*100)

        # Initialize json_data with basic problem information
        json_data = {
            "pid": pid,
            "query": question,
            "image": image_path,
            "answer": answer,
        }

        if 'metadata' in problem:
            json_data['metadata'] = problem['metadata']

        # Generate base response if requested
        if 'base' in self.output_types:
            base_response = self.planner.generate_base_response(question, image_path, self.max_tokens)
            json_data["base_response"] = base_response
            if self.verbose:
                print("\n## Base Response:")
                print("#"*50)
                print(f"{base_response}")
                print("#"*50)

        # If only base response is needed, save and return
        if set(self.output_types) == {'base'}:
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=4)
                print(f"\n==>Base response output saved to: {output_file}")
            return
    
        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct'} & set(self.output_types):

             # Analyze query
            query_analysis = self.planner.analyze_query(question, image_path)
            json_data["query_analysis"] = query_analysis

            if self.verbose:
                print("\n## Query Analysis:")
                print("#"*50)
                print(f"{query_analysis}")
                print("#"*50)

            start_time = time.time()
            step_count = 0
            action_times = []

            # Main execution loop
            while step_count < self.max_steps and (time.time() - start_time) < self.max_time:
                step_count += 1
                if self.verbose:
                    print(f"\n## [Step {step_count}]")

                # Generate next step
                start_time_step = time.time()
                next_step = self.planner.generate_next_step(
                    question, 
                    image_path, 
                    query_analysis, 
                    self.memory, 
                    step_count, 
                    self.max_steps,
                    json_data
                )
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)

                if self.verbose:
                    print(f"\n## [{step_count}] Next Step:")
                    print("#"*50)
                    print(f"Next Step:\n{next_step}")
                    print("#"*50)
                    print(f"\n==>Extracted Context:\n{context}")
                    print(f"\n==>Extracted Sub-goal:\n{sub_goal}\n")
                    print(f"\n==>Extracted Tool:\n{tool_name}")

                if tool_name is None or tool_name not in self.planner.available_tools:
                    print(f"Error: Tool '{tool_name}' is not available or not found.")
                    command = "Not command is generated due to the tool not found."
                    result = "Not result is generated due to the tool not found."

                else:
                    # Generate the tool command
                    tool_command = self.executor.generate_tool_command(
                        question, 
                        image_path, 
                        context, 
                        sub_goal, 
                        tool_name, 
                        self.planner.toolbox_metadata[tool_name],
                        step_count,  # step_count
                        json_data  # json_data
                    )
                    analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
                    
                    if self.verbose:
                        print(f"\n## [{step_count}] Tool Command:")
                        print("#"*50)
                        print(f"{tool_command}")
                        print("#"*50)
                        print(f"\n==>Extracted Command:\n{command}\n")

                    # Execute the tool command
                    result = self.executor.execute_tool_command(tool_name, command)
                    result = make_json_serializable_truncated(result) # Convert to JSON serializable format

                    if self.verbose:
                        print(f"\n## [{step_count}] Tool Execution:")
                        print("\n==>Executed Result:")
                        print(json.dumps(result, indent=4))

                # Track execution time
                end_time_step = time.time()
                execution_time_step = round(end_time_step - start_time_step, 2)
                action_times.append(execution_time_step)

                if self.verbose:
                    print(f"Execution time for step {step_count}: {execution_time_step:.2f} seconds")

                # Update memory
                self.memory.add_action(step_count, tool_name, sub_goal, command, result)
                memeory_actions = self.memory.get_actions()

                # Verify memory
                stop_verification = self.planner.verificate_context(
                    question, 
                    image_path, 
                    query_analysis, 
                    self.memory,
                    step_count,
                    json_data 
                )
                context_verification, conclusion = self.planner.extract_conclusion(stop_verification)
                
                if self.verbose:
                    print(f"\n## [{step_count}] Stopping Verification:")
                    print("#"*50)
                    print(f"{context_verification}")
                    print("#"*50)
                    print(f"\n==>Extracted Conclusion:\n{conclusion}")

                if conclusion == 'STOP':
                    break

            # Check if we've hit a limit
            if self.verbose:
                if step_count >= self.max_steps:
                    print(f"\n==>Maximum number of steps ({self.max_steps}) reached. Stopping execution.")
                elif (time.time() - start_time) >= self.max_time:
                    print(f"\n==>Maximum time limit ({self.max_time} seconds) reached. Stopping execution.")

                # Print memory
                print(f"\n## [{step_count}] Memory:")
                print("#"*50)
                if isinstance(memeory_actions, dict):
                    print(json.dumps(memeory_actions, indent=4))
                elif isinstance(memeory_actions, list):
                    print(json.dumps(memeory_actions, indent=4))
                else:
                    print(memeory_actions)
                print("#"*50)

            # Add memory and statistics to json_data
            json_data.update({
                "memory": memeory_actions,
                "step_count": step_count,
                "execution_time": round(time.time() - start_time, 2),
            })

            # Generate final output if requested
            if 'final' in self.output_types:
                final_output = self.planner.generate_final_output(question, image_path, self.memory)
                json_data["final_output"] = final_output
                if self.verbose:
                    print("\n## Final Output:")
                    print("#"*50)
                    print(f"{final_output}")
                    print("#"*50)

            # Generate direct output if requested
            if 'direct' in self.output_types:
                direct_output = self.planner.generate_direct_output(question, image_path, self.memory)
                json_data["direct_output"] = direct_output
                if self.verbose:
                    print("\n## Direct Output:")
                    print("#"*50)
                    print(f"{direct_output}")
                    print("#"*50)

        # Save results
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=4)
            print(f"\n==>Output saved to: {output_file}")

        # Print execution statistics if we ran the full pipeline
        if {'final', 'direct'} & set(self.output_types):
            print(f"\n## Execution Statistics for Problem {index}:")
            print(f"==>Total steps executed: {step_count}")
            print(f"==>Total execution time: {time.time() - start_time:.2f} seconds")
            
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the agentflow demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens for LLM generation.")
    parser.add_argument("--run_baseline_only", type=bool, default=False, help="Run only the baseline (no toolbox).")
    parser.add_argument("--task", default="minitoolbench", help="Task to run.")
    parser.add_argument("--data_file", default="data/data.json", help="Data file to run.")
    parser.add_argument("--task_description", default="", help="Task description.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Generalist_Solution_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--tool_engine", default="Default", help="List of tool engines corresponding to enabled_tools, separated by commas.")
    parser.add_argument("--index", type=int, default=0, help="Index of the problem in the benchmark file.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--output_json_dir", default="results", help="Path to output JSON directory.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Enable verbose output.")
    parser.add_argument("--vllm_config_path", type=str, default=None, help="Path to VLLM configuration file.")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for the LLM API.")
    parser.add_argument("--check_model", type=bool, default=True, help="Check if the model is available.")
    return parser.parse_args()


def main(args):
    # Initialize Tools
    enabled_tools = args.enabled_tools.split(",") if args.enabled_tools else []
    tool_engine = args.tool_engine.split(",") if args.tool_engine else ["Default"]
    print(args.base_url, args.llm_engine_name)

    if len(tool_engine) < len(enabled_tools):
        tool_engine += ["Default"] * (len(enabled_tools) - len(tool_engine))

    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=args.llm_engine_name,
        verbose=args.verbose,
        vllm_config_path=args.vllm_config_path,
        base_url=args.base_url,
        check_model=args.check_model
    )

    # Instantiate Planner
    planner = Planner(
        llm_engine_name=args.llm_engine_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=args.verbose,
        base_url=args.base_url,
        temperature=args.temperature
    )

    # Instantiate Memory
    memory = Memory()

    # Instantiate Executor
    executor = Executor(
        llm_engine_name=args.llm_engine_name,
        root_cache_dir=args.root_cache_dir,
        verbose=args.verbose,
        base_url=args.base_url,
        temperature=args.temperature
    )

    # Instantiate Solver
    solver = Solver(
        planner=planner,
        memory=memory,
        executor=executor,
        task=args.task,
        data_file=args.data_file,
        task_description=args.task_description,
        output_types=args.output_types,
        index=args.index,
        verbose=args.verbose,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        output_json_dir=args.output_json_dir,
        root_cache_dir=args.root_cache_dir,
        temperature=args.temperature
    )

    # Solve the task or problem
    solver.solve()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
