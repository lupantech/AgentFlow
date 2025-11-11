# Import the solver
from agentflow.agentflow.solver import construct_solver

# Set the LLM engine name.
# Supported engines are "dashscope" and "gpt-4o".
# By default, it uses "dashscope".
# To use "gpt-4o", uncomment the line below and comment out the "dashscope" line.
llm_engine_name = "dashscope"
# llm_engine_name = "gpt-4o"

# Construct the solver
solver = construct_solver(llm_engine_name=llm_engine_name)

# Solve the user query
output = solver.solve("What is the capital of France?")
print(output["direct_output"])