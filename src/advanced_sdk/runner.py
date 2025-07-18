import asyncio
import os
from dotenv import load_dotenv
from agents import AgentHooks, AsyncOpenAI , OpenAIChatCompletionsModel , RunConfig , Agent , Runner , function_tool 
import weave
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor
weave.init("Advanced_SDK")
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")




external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
    workflow_name='' \
    'Advanced SDK Workflow',
)
@function_tool(name_override='Calculate_FIBONACCI')
def calculate_fibonacci(n: int) -> str:
    if n <= 1:
        return n
    return str(calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2))

# we have to analyze the behaviour of the Runner when calling this recursive tool , in the async and sync runners
# for this we have to use hooks to hook into the lifecycle of the agent and see the tool behaviour

class hooks(AgentHooks):
    async def on_start(self, context, agent):
        print('Agent started')
    async def on_tool_start(self, context, agent, tool):
        print('Tool started')
        print(tool.name)
    async def on_tool_end(self, context, agent, tool, result):
        print('Tool ended')
        print(result)
    async def on_end(self, context, agent, output):
        print('Agent ended')
        print(output)
agent_hooks = hooks()
fib_agent = Agent(
    name = 'Fibonacci Agent',
    instructions='''You help users with the fibonacci sequence using the tool "Calculate_FIBONACCI"''',
    model=model,
    tools=[calculate_fibonacci],
    handoff_description='''Get the fibonacci sequence of the specified number''',
    hooks=agent_hooks
)
# Case 1 : Async Runner 
# An error occurred while running the tool. Please try again. Error: 'FunctionTool' object is not callable
def run_agent():
    result =  Runner.run_sync(
        starting_agent=fib_agent,
        input='Calculate the fibonacci sequence of 10',
        run_config=config,
       
    )
    print(result.final_output)

run_agent()