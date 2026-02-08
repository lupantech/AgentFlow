import os
import sys
import importlib
import inspect
import traceback
from typing import Dict, Any, List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def _get_optimal_workers(max_workers: int = None) -> tuple[int, str]:
    """
    Intelligently determine optimal number of worker threads based on:
    1. External parallelism (GNU parallel, SLURM, etc.)
    2. Available CPU cores
    3. User-specified max_workers

    Returns:
        (optimal_workers, reason): Number of workers and explanation
    """
    cpu_count = multiprocessing.cpu_count()

    # Check for GNU parallel environment
    parallel_seq = os.environ.get('PARALLEL_SEQ')
    parallel_jobslot = os.environ.get('PARALLEL_JOBSLOT')

    # Check for SLURM scheduler
    slurm_ntasks = os.environ.get('SLURM_NTASKS')
    slurm_cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')

    external_parallelism = None
    detection_method = None

    # Detect external parallelism
    if parallel_seq or parallel_jobslot:
        # GNU parallel is running
        # Estimate: typically parallel -j N means N parallel jobs
        # Conservative estimate: assume N = cpu_count / 2
        external_parallelism = max(cpu_count // 2, 1)
        detection_method = "GNU parallel"
    elif slurm_ntasks:
        try:
            external_parallelism = int(slurm_ntasks)
            detection_method = "SLURM"
        except:
            pass

    # Calculate optimal workers
    if max_workers is not None:
        # User explicitly specified max_workers
        optimal = max_workers
        reason = f"user-specified: {max_workers} workers"
    elif external_parallelism:
        # External parallelism detected
        # Formula: optimal = max(1, cpu_count / external_parallelism)
        # But cap at 4 to avoid too few workers
        optimal = max(1, min(4, cpu_count // external_parallelism))
        reason = f"{detection_method} detected, auto-adjusted to {optimal} workers (CPU={cpu_count}, external_jobs≈{external_parallelism})"
    else:
        # No external parallelism, use moderate default
        optimal = min(4, max(2, cpu_count // 4))
        reason = f"standalone mode: {optimal} workers (CPU={cpu_count})"

    return optimal, reason

class Initializer:
    def __init__(self,
    enabled_tools: List[str] = [],
    tool_engine: List[str] = [],
    model_string: str = None,
    verbose: bool = False,
    vllm_config_path: str = None,
    base_url: str = None,
    check_model: bool = True,
    parallel_loading: bool = True,
    max_workers: int = None):
        """
        Initialize the tool initializer with intelligent parallel loading.

        Args:
            enabled_tools: List of tool names to enable
            tool_engine: List of engine names corresponding to each tool
            model_string: Default model string
            verbose: Whether to print verbose output
            vllm_config_path: Path to vllm config
            base_url: Base URL for API
            check_model: Whether to check model availability
            parallel_loading: Whether to load tools in parallel (default: True)
            max_workers: Maximum number of parallel workers (default: None for auto-detect)
                        If None, will intelligently detect based on:
                        - External parallelism (GNU parallel, SLURM, etc.)
                        - Available CPU cores
        """
        self.toolbox_metadata = {}
        self.available_tools = []
        self.enabled_tools = enabled_tools
        self.tool_engine = tool_engine
        self.load_all = self.enabled_tools == ["all"]
        self.model_string = model_string
        self.verbose = verbose
        self.vllm_server_process = None
        self.vllm_config_path = vllm_config_path
        self.base_url = base_url
        self.check_model = check_model
        self.parallel_loading = parallel_loading

        # Intelligently determine optimal workers
        optimal_workers, worker_reason = _get_optimal_workers(max_workers)
        self.max_workers = optimal_workers

        # Add tool instance cache - stores instantiated tools with their engines
        self.tool_instances_cache = {}

        print("\n==> Initializing agentflow...")
        print(f"Enabled tools: {self.enabled_tools} with {self.tool_engine}")
        print(f"LLM engine name: {self.model_string}")
        print(f"Parallel loading: {self.parallel_loading} ({worker_reason})")
        self._set_up_tools()
        
        # if vllm, set up the vllm server
        # if model_string.startswith("vllm-"):
        #     self.setup_vllm_server()

    def get_project_root(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != '/':
            if os.path.exists(os.path.join(current_dir, 'agentflow')):
                return os.path.join(current_dir, 'agentflow')
            current_dir = os.path.dirname(current_dir)
        raise Exception("Could not find project root")

    def build_tool_name_mapping(self, tools_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Build a mapping dictionary by extracting TOOL_NAME from each tool file.

        Returns:
            Dict with two keys:
            - 'short_to_long': Maps short names (class names) to long names (external TOOL_NAME)
            - 'long_to_internal': Maps long names to internal class names and directory names
        """
        short_to_long = {}  # e.g., Base_Generator_Tool -> Generalist_Solution_Generator_Tool
        long_to_internal = {}  # e.g., Generalist_Solution_Generator_Tool -> {class_name, dir_name}

        for root, dirs, files in os.walk(tools_dir):
            if 'tool.py' in files:
                dir_name = os.path.basename(root)
                tool_file_path = os.path.join(root, 'tool.py')

                try:
                    # Read the tool.py file and extract TOOL_NAME
                    with open(tool_file_path, 'r') as f:
                        content = f.read()

                    # Extract TOOL_NAME using simple string parsing
                    external_tool_name = None
                    for line in content.split('\n'):
                        if line.strip().startswith('TOOL_NAME ='):
                            # Extract the value between quotes
                            external_tool_name = line.split('=')[1].strip().strip('"\'')
                            break

                    if external_tool_name:
                        # Find the class name from the file
                        for line in content.split('\n'):
                            if 'class ' in line and 'BaseTool' in line:
                                class_name = line.split('class ')[1].split('(')[0].strip()

                                # Build both mappings
                                short_to_long[class_name] = external_tool_name
                                long_to_internal[external_tool_name] = {
                                    "class_name": class_name,
                                    "dir_name": dir_name
                                }
                                print(f"Mapped: {class_name} -> {external_tool_name} (dir: {dir_name})")
                                break
                except Exception as e:
                    print(f"Warning: Could not extract TOOL_NAME from {tool_file_path}: {str(e)}")
                    continue

        return {"short_to_long": short_to_long, "long_to_internal": long_to_internal}

    def _load_single_tool(self, root: str, import_path: str, agentflow_dir: str) -> Dict[str, Any]:
        """
        Load all tools from a single module and return their metadata.
        This method is designed to be called in parallel.

        Note: A single module may contain multiple tool classes (either defined locally or imported).

        Returns:
            Dict with lists of tool metadata/instances, or error information
        """
        result = {'metadata_list': [], 'instance_list': [], 'errors': []}

        try:
            module = importlib.import_module(import_path)
            current_dir_name = os.path.basename(root)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.endswith('Tool') and name != 'BaseTool':
                    try:
                        # Check if the tool requires specific llm engine
                        tool_index = -1
                        for i, tool_name in enumerate(self.enabled_tools):
                            # First check short_to_long mapping
                            if hasattr(self, 'tool_name_mapping'):
                                short_to_long = self.tool_name_mapping.get('short_to_long', {})
                                long_to_internal = self.tool_name_mapping.get('long_to_internal', {})

                                # If input is short name, convert to long name
                                long_name = short_to_long.get(tool_name, tool_name)

                                # Check if long name matches this directory
                                if long_name in long_to_internal:
                                    if long_to_internal[long_name]["dir_name"] == current_dir_name:
                                        tool_index = i
                                        break

                            # Fallback to original behavior
                            if tool_name.lower().replace('_tool', '') == current_dir_name:
                                tool_index = i
                                break

                        if tool_index >= 0 and tool_index < len(self.tool_engine):
                            engine = self.tool_engine[tool_index]
                            if engine == "Default":
                                tool_instance = obj()
                            elif engine == "self":
                                tool_instance = obj(model_string=self.model_string, base_url=self.base_url)
                            else:
                                tool_instance = obj(model_string=engine, base_url=self.base_url)
                        else:
                            tool_instance = obj()

                        # Use the external tool name (from TOOL_NAME) as the key
                        metadata_key = getattr(tool_instance, 'tool_name', name)

                        metadata = {
                            'tool_name': getattr(tool_instance, 'tool_name', 'Unknown'),
                            'tool_description': getattr(tool_instance, 'tool_description', 'No description'),
                            'tool_version': getattr(tool_instance, 'tool_version', 'Unknown'),
                            'input_types': getattr(tool_instance, 'input_types', {}),
                            'output_type': getattr(tool_instance, 'output_type', 'Unknown'),
                            'demo_commands': getattr(tool_instance, 'demo_commands', []),
                            'user_metadata': getattr(tool_instance, 'user_metadata', {}),
                            'require_llm_engine': getattr(obj, 'require_llm_engine', False),
                        }

                        result['metadata_list'].append((metadata_key, metadata))
                        result['instance_list'].append((metadata_key, tool_instance))

                    except Exception as e:
                        result['errors'].append(f"Error instantiating {name}: {str(e)}")

        except Exception as e:
            result['errors'].append(f"Error loading module {import_path}: {str(e)}")

        return result

    def load_tools_and_get_metadata(self, parallel: bool = True, max_workers: int = 4) -> Dict[str, Any]:
        """
        Load tools and get metadata. Can be done in parallel for faster initialization.

        Args:
            parallel: If True, load tools in parallel using ThreadPoolExecutor
            max_workers: Maximum number of worker threads (default: 4)
        """
        print(f"Loading tools and getting metadata... (parallel={parallel}, max_workers={max_workers})")
        start_time = time.time()

        self.toolbox_metadata = {}
        agentflow_dir = self.get_project_root()
        tools_dir = os.path.join(agentflow_dir, 'tools')

        # Add the agentflow directory and its parent to the Python path
        sys.path.insert(0, agentflow_dir)
        sys.path.insert(0, os.path.dirname(agentflow_dir))
        print(f"Updated Python path: {sys.path}")

        if not os.path.exists(tools_dir):
            print(f"Error: Tools directory does not exist: {tools_dir}")
            return self.toolbox_metadata

        # Build tool name mapping if not already built
        if not hasattr(self, 'tool_name_mapping'):
            self.tool_name_mapping = self.build_tool_name_mapping(tools_dir)
        print(f"\n==> Tool name mapping (short to long): {self.tool_name_mapping.get('short_to_long', {})}")
        print(f"==> Tool name mapping (long to internal): {self.tool_name_mapping.get('long_to_internal', {})}")

        # Collect all tool directories to process, maintaining the order from available_tools
        tool_dirs_to_process = []

        if self.load_all:
            # If loading all tools, use os.walk order
            for root, dirs, files in os.walk(tools_dir):
                if 'tool.py' in files:
                    file = 'tool.py'
                    module_path = os.path.join(root, file)
                    relative_path = os.path.relpath(module_path, agentflow_dir)
                    import_path = '.'.join(os.path.split(relative_path)).replace(os.sep, '.')[:-3]
                    tool_dirs_to_process.append((root, import_path))
        else:
            # Build a map of directory names to paths for efficient lookup
            dir_to_paths = {}
            for root, dirs, files in os.walk(tools_dir):
                if 'tool.py' in files:
                    dir_name = os.path.basename(root)
                    file = 'tool.py'
                    module_path = os.path.join(root, file)
                    relative_path = os.path.relpath(module_path, agentflow_dir)
                    import_path = '.'.join(os.path.split(relative_path)).replace(os.sep, '.')[:-3]
                    dir_to_paths[dir_name] = (root, import_path)

            # Process in the order of available_tools (which matches enabled_tools order)
            for tool_dir in self.available_tools:
                if tool_dir in dir_to_paths:
                    tool_dirs_to_process.append(dir_to_paths[tool_dir])
                else:
                    print(f"⚠️  Warning: Tool directory '{tool_dir}' not found in tools directory")

        if parallel and len(tool_dirs_to_process) > 1:
            # Parallel loading
            print(f"\n==> Loading {len(tool_dirs_to_process)} tool modules in parallel...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tool loading tasks and maintain order
                futures = [
                    (executor.submit(self._load_single_tool, root, import_path, agentflow_dir), import_path)
                    for root, import_path in tool_dirs_to_process
                ]

                # Wait for all to complete, then process in original order
                print(f"⏳ Waiting for all {len(futures)} modules to load...")

                # Process results in the original submission order
                for future, import_path in futures:
                    try:
                        result = future.result()

                        # Report any errors
                        if result['errors']:
                            for error in result['errors']:
                                print(f"Error loading {import_path}: {error}")

                        # Process all tools found in this module (in the order they appear in the module)
                        for metadata_key, metadata in result['metadata_list']:
                            self.toolbox_metadata[metadata_key] = metadata

                        for instance_key, instance in result['instance_list']:
                            self.tool_instances_cache[instance_key] = instance
                            print(f"✓ Loaded: {instance_key} with engine: {getattr(instance, 'model_string', 'default')}")
                    except Exception as e:
                        print(f"Exception loading {import_path}: {str(e)}")
        else:
            # Serial loading (original behavior)
            print(f"\n==> Loading {len(tool_dirs_to_process)} tool modules serially...")
            for root, import_path in tool_dirs_to_process:
                print(f"\n==> Attempting to import: {import_path}")
                result = self._load_single_tool(root, import_path, agentflow_dir)

                # Report any errors
                if result['errors']:
                    for error in result['errors']:
                        print(f"Error: {error}")

                # Process all tools found in this module
                for metadata_key, metadata in result['metadata_list']:
                    self.toolbox_metadata[metadata_key] = metadata
                    print(f"Metadata for {metadata_key}: {metadata}")

                for instance_key, instance in result['instance_list']:
                    self.tool_instances_cache[instance_key] = instance
                    print(f"Cached tool instance: {instance_key} with engine: {getattr(instance, 'model_string', 'default')}")

        elapsed_time = time.time() - start_time
        print(f"\n==> Total number of tools imported: {len(self.toolbox_metadata)} (took {elapsed_time:.2f}s)")

        return self.toolbox_metadata

    def run_demo_commands(self) -> List[str]:
        print("\n==> Running demo commands for each tool...")
        self.available_tools = []

        # Process tools in alphabetical order by tool name
        for tool_name, tool_data in sorted(self.toolbox_metadata.items()):
            print(f"Checking availability of {tool_name}...")

            try:
                # Use the cached tool instance instead of creating a new one
                # This preserves the engine configuration from load_tools_and_get_metadata
                if tool_name in self.tool_instances_cache:
                    tool_instance = self.tool_instances_cache[tool_name]
                    print(f"✓ Using cached instance with engine: {getattr(tool_instance, 'model_string', 'default')}")
                else:
                    # Fallback: create new instance if not in cache
                    # tool_name here is the long external name from metadata
                    # We need to get the internal class name and directory
                    if hasattr(self, 'tool_name_mapping'):
                        long_to_internal = self.tool_name_mapping.get('long_to_internal', {})

                        if tool_name in long_to_internal:
                            dir_name = long_to_internal[tool_name]["dir_name"]
                            class_name = long_to_internal[tool_name]["class_name"]
                        else:
                            # Fallback to original behavior
                            dir_name = tool_name.lower().replace('_tool', '')
                            class_name = tool_name
                    else:
                        # Fallback to original behavior
                        dir_name = tool_name.lower().replace('_tool', '')
                        class_name = tool_name

                    # Import the tool module
                    module_name = f"tools.{dir_name}.tool"
                    module = importlib.import_module(module_name)

                    # Get the tool class
                    tool_class = getattr(module, class_name)

                    # Instantiate the tool
                    tool_instance = tool_class()
                    print(f"⚠️  Created new instance (not in cache)")

                # FIXME This is a temporary workaround to avoid running demo commands
                self.available_tools.append(tool_name)

            except Exception as e:
                print(f"Error checking availability of {tool_name}: {str(e)}")
                print(traceback.format_exc())

        # update the toolmetadata with the available tools
        self.toolbox_metadata = {tool: self.toolbox_metadata[tool] for tool in self.available_tools}
        print("\n✅ Finished running demo commands for each tool.")
        # print(f"Updated total number of available tools: {len(self.toolbox_metadata)}")
        # print(f"Available tools: {self.available_tools}")
        return self.available_tools
    
    def _set_up_tools(self) -> None:
        print("\n==> Setting up tools...")

        # First, build a temporary mapping by scanning all tools
        agentflow_dir = self.get_project_root()
        tools_dir = os.path.join(agentflow_dir, 'tools')
        self.tool_name_mapping = self.build_tool_name_mapping(tools_dir) if os.path.exists(tools_dir) else {}

        # Map input tool names (short) to internal directory names for filtering
        mapped_tools = []
        short_to_long = self.tool_name_mapping.get('short_to_long', {})
        long_to_internal = self.tool_name_mapping.get('long_to_internal', {})

        for i, tool in enumerate(self.enabled_tools):
            # If tool is a short name, convert to long name first
            long_name = short_to_long.get(tool, tool)
            print(f"  [{i}] {tool} -> {long_name}", end="")

            # Then get the directory name
            if long_name in long_to_internal:
                dir_name = long_to_internal[long_name]["dir_name"]
                mapped_tools.append(dir_name)
                print(f" -> {dir_name}")
            else:
                # Fallback to original behavior for unmapped tools
                dir_name = tool.lower().replace('_tool', '')
                mapped_tools.append(dir_name)
                print(f" -> {dir_name} (fallback)")

        self.available_tools = mapped_tools
        print(f"\n==> Mapped tools (directory names): {mapped_tools}")

        # Now load tools and get metadata (with optional parallel loading)
        self.load_tools_and_get_metadata(
            parallel=self.parallel_loading,
            max_workers=self.max_workers
        )

        # Run demo commands to determine available tools
        # This will update self.available_tools to contain external names
        self.run_demo_commands()

        # available_tools is now already updated by run_demo_commands with external names
        print("✅ Finished setting up tools.")
        print(f"✅ Total number of final available tools: {len(self.available_tools)}")
        print(f"✅ Final available tools: {self.available_tools}")

if __name__ == "__main__":
    import time

    enabled_tools = ["Base_Generator_Tool", "Python_Coder_Tool", "Google_Search_Tool", "Wikipedia_Search_Tool"]
    tool_engine = ["gpt-4o-mini", "gpt-4o-mini", "Default", "Default"]

    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: Serial vs Parallel Tool Loading")
    print("="*80)

    # Test 1: Serial loading
    print("\n[Test 1] Serial Loading...")
    print("-"*80)
    start = time.time()
    init_serial = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        parallel_loading=False  # Serial
    )
    serial_time = time.time() - start
    print(f"\n✓ Serial loading completed in {serial_time:.2f}s")
    print(f"  Tools loaded: {len(init_serial.available_tools)}")

    # Test 2: Parallel loading (4 workers)
    print("\n[Test 2] Parallel Loading (4 workers)...")
    print("-"*80)
    start = time.time()
    init_parallel = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        parallel_loading=True,  # Parallel
        max_workers=4
    )
    parallel_time = time.time() - start
    print(f"\n✓ Parallel loading completed in {parallel_time:.2f}s")
    print(f"  Tools loaded: {len(init_parallel.available_tools)}")

    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Serial loading:   {serial_time:>6.2f}s")
    print(f"Parallel loading: {parallel_time:>6.2f}s (4 workers)")
    print(f"Speedup:          {serial_time/parallel_time:>6.2f}x")
    print(f"Time saved:       {serial_time - parallel_time:>6.2f}s")
    print("="*80)
    