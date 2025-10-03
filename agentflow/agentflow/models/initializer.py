import os
import sys
import importlib
import inspect
import traceback
from typing import Dict, Any, List, Tuple
import time

class Initializer:
    def __init__(self, 
    enabled_tools: List[str] = [], 
    tool_engine: List[str] = [], 
    model_string: str = None, 
    verbose: bool = False, 
    vllm_config_path: str = None, 
    base_url: str = None, 
    check_model: bool = True):

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
        print("\n==> Initializing agentflow...")
        print(f"Enabled tools: {self.enabled_tools} with {self.tool_engine}")
        print(f"LLM engine name: {self.model_string}")
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

    def load_tools_and_get_metadata(self) -> Dict[str, Any]:
        # Implementation of load_tools_and_get_metadata function
        print("Loading tools and getting metadata...")
        self.toolbox_metadata = {}
        agentflow_dir = self.get_project_root()
        tools_dir = os.path.join(agentflow_dir, 'tools')
        # print(f"agentflow directory: {agentflow_dir}")
        # print(f"Tools directory: {tools_dir}")

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

        for root, dirs, files in os.walk(tools_dir):
            # print(f"\nScanning directory: {root}")
            if 'tool.py' in files and (self.load_all or os.path.basename(root) in self.available_tools):
                file = 'tool.py'
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]
                relative_path = os.path.relpath(module_path, agentflow_dir)
                import_path = '.'.join(os.path.split(relative_path)).replace(os.sep, '.')[:-3]

                print(f"\n==> Attempting to import: {import_path}")
                try:
                    module = importlib.import_module(import_path)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and name.endswith('Tool') and name != 'BaseTool':
                            print(f"Found tool class: {name}")
                            try:
                                # Check if the tool requires specific llm engine
                                tool_index = -1
                                current_dir_name = os.path.basename(root)
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
                                        tool_instance = obj(model_string=self.model_string)
                                    else:
                                        tool_instance = obj(model_string=engine)
                                else:
                                    tool_instance = obj()
                                # Use the external tool name (from TOOL_NAME) as the key
                                metadata_key = getattr(tool_instance, 'tool_name', name)

                                self.toolbox_metadata[metadata_key] = {
                                    'tool_name': getattr(tool_instance, 'tool_name', 'Unknown'),
                                    'tool_description': getattr(tool_instance, 'tool_description', 'No description'),
                                    'tool_version': getattr(tool_instance, 'tool_version', 'Unknown'),
                                    'input_types': getattr(tool_instance, 'input_types', {}),
                                    'output_type': getattr(tool_instance, 'output_type', 'Unknown'),
                                    'demo_commands': getattr(tool_instance, 'demo_commands', []),
                                    'user_metadata': getattr(tool_instance, 'user_metadata', {}), # This is a placeholder for user-defined metadata
                                    'require_llm_engine': getattr(obj, 'require_llm_engine', False),
                                }
                                print(f"Metadata for {metadata_key}: {self.toolbox_metadata[metadata_key]}")
                            except Exception as e:
                                print(f"Error instantiating {name}: {str(e)}")
                except Exception as e:
                    print(f"Error loading module {module_name}: {str(e)}")
                    
        print(f"\n==> Total number of tools imported: {len(self.toolbox_metadata)}")

        return self.toolbox_metadata

    def run_demo_commands(self) -> List[str]:
        print("\n==> Running demo commands for each tool...")
        self.available_tools = []

        for tool_name, tool_data in self.toolbox_metadata.items():
            print(f"Checking availability of {tool_name}...")

            try:
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

        for tool in self.enabled_tools:
            # If tool is a short name, convert to long name first
            long_name = short_to_long.get(tool, tool)

            # Then get the directory name
            if long_name in long_to_internal:
                mapped_tools.append(long_to_internal[long_name]["dir_name"])
            else:
                # Fallback to original behavior for unmapped tools
                mapped_tools.append(tool.lower().replace('_tool', ''))

        self.available_tools = mapped_tools

        # Now load tools and get metadata
        self.load_tools_and_get_metadata()

        # Run demo commands to determine available tools
        # This will update self.available_tools to contain external names
        self.run_demo_commands()

        # available_tools is now already updated by run_demo_commands with external names
        print("✅ Finished setting up tools.")
        print(f"✅ Total number of final available tools: {len(self.available_tools)}")
        print(f"✅ Final available tools: {self.available_tools}")

if __name__ == "__main__":
    enabled_tools = ["Base_Generator_Tool", "Python_Coder_Tool"]
    tool_engine = ["Default", "Default"]
    initializer = Initializer(enabled_tools=enabled_tools,tool_engine=tool_engine)

    print("\nAvailable tools:")
    print(initializer.available_tools)

    print("\nToolbox metadata for available tools:")
    print(initializer.toolbox_metadata)
    