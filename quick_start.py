# Import the solver
from agentflow.agentflow.solver import construct_solver

# Set the LLM engine name
llm_engine_name = "gpt-4o" 
# llm_engine_name = "dashscope" # you can use "dashscope" as well, to use the default API key in the environment variables qwen2.5-7b-instruct

# Construct the solver
solver = construct_solver(llm_engine_name=llm_engine_name)

# Solve the user query
output = solver.solve("What is the capital of France?")
print(output["direct_output"])