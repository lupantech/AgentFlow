import os
from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Generalist_Solution_Generator_Tool"

LIMITATION = f"""
The {TOOL_NAME} may provide hallucinated or incorrect responses.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Use it for general queries or tasks that don't require specialized knowledge or specific tools in the toolbox.
2. Provide clear, specific query.   
3. Use it to answer the original query through step by step reasoning for tasks without complex or multi-step reasoning.
4. For complex queries, break them down into subtasks and use the tool multiple times.
5. Use it as a starting point for complex tasks, then refine with specialized tools.
6. Verify important information from its responses.
"""

class Base_Generator_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A generalized tool that takes query from the user, and answers the question step by step to the best of its ability. It can also accept an image.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The query that includes query from the user to guide the agent to generate response.",
                # "query": "str - The query that includes query from the user to guide the agent to generate response (Examples: 'Describe this image in detail').",
                # "image": "str - The path to the image file if applicable (default: None).",
            },
            output_type="str - The generated response to the original query",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Summarize the following text in a few lines")',
                    "description": "Generate a short summary given the query from the user."
                },
                # {
                #     "command": 'execution = tool.execute(query="Explain the mood of this scene.", image="path/to/image1.png")',
                #     "description": "Generate a caption focusing on the mood using a specific query and image."
                # },
                # {
                    # "command": 'execution = tool.execute(query="Give your best coordinate estimate for the pacemaker in the image and return (x1, y1, x2, y2)", image="path/to/image2.png")',
                    # "description": "Generate bounding box coordinates given the image and query from the user. The format should be (x1, y1, x2, y2)."
                # },
                # {
                #     "command": 'execution = tool.execute(query="Is the number of tiny objects that are behind the small metal jet less than the number of tiny things left of the tiny sedan?", image="path/to/image2.png")',
                #     "description": "Answer a question step by step given the image."
                # }
            ],

            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }

        )
        self.model_string = model_string  
        print(f"Initializing Generalist Tool with model: {self.model_string}")
        # multimodal = True if image else False
        multimodal = False
        # llm_engine = create_llm_engine(model_string=self.model_string, is_multimodal=multimodal, base_url=self.base_url)
        
        # NOTE: deterministic mode
        self.llm_engine = create_llm_engine(
            model_string=self.model_string, 
            is_multimodal=multimodal, 
            temperature=0.0, 
            top_p=1.0, 
            frequency_penalty=0.0, 
            presence_penalty=0.0
            )


    def execute(self, query, image=None):
        
        try:
            input_data = [query]
            response = self.llm_engine(input_data[0])
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/base_generator
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Example usage of the Generalist_Tool
    tool = Base_Generator_Tool()

    tool = Base_Generator_Tool(model_string="gpt-4o-mini") # NOTE: strong LLM for tool
    # tool = Base_Generator_Tool(model_string="gemini-1.5-flash") # NOTE: weak 8B model for tool
    # tool = Base_Generator_Tool(model_string="dashscope") # NOTE: weak Qwen2.5-7B model for tool


    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    query = "What is the capital of France?"

    # Execute the tool with default query
    try:
        execution = tool.execute(query=query)

        print("Generated Response:")
        print(execution)
    except Exception as e: 
        print(f"Execution failed: {e}")

    print("Done!")
