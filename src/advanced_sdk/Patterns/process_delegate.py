import datetime
import os
import random
import uuid
from agents import ( Agent, AgentOutputSchema, FunctionToolResult, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, ItemHelpers, ModelSettings, OutputGuardrailTripwireTriggered, RunHooks, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, TResponseInputItem, Tool, ToolsToFinalOutputResult , handoff,Handoff,function_tool , AgentOutputSchemaBase, input_guardrail, output_guardrail, set_trace_processors , RunContextWrapper , handoffs , HandoffInputData , HandoffInputFilter , AgentHooks  )
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX , prompt_with_handoff_instructions
from agents.run import RunConfig
from dotenv import load_dotenv
import asyncio
from agents.tracing import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from dataclasses import dataclass
import weave
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor
from typing import Any, List, Literal
from agents.extensions import handoff_filters
from pydantic import BaseModel
import agentops
import mlflow
import weave
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor
import logfire
# set_tracing_disabled(True)

# logfire.configure()
# logfire.instrument_openai_agents()
# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
weave.init("Advanced_SDK")
set_trace_processors([WeaveTracingProcessor()])
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

# PROCESS AND DELEGATE PATTERN 
# In this pattern we are implementing a Customer support example , which consists of the Three Agents :
# Triage Agent(Basic Agent) , Refund Agent , Billing Agent , Technical Agent
# We will all do this dynamically to make this agent fully dynamic

Refund_Agent = Agent(
    name = 'Refund Agent',
    instructions='''You help users with the refund queries''',
    model=model,
    handoff_description='''Assist users with the refund queries''',
)

Billing_Agent = Agent(
    name = 'Billing Agent',
    instructions='''You help users with the billing queries''',
    model=model,
    handoff_description='''Assist users with the billing queries''',
)

Technical_Agent = Agent(
    name = 'Technical Agent',
    instructions='''You help users with the technical queries''',
    model=model,
    handoff_description='''Assist users with the technical queries''',
)
class QueryType(BaseModel):
    query_type : Literal['refund' , 'billing' , 'echnical' , 'general']
query_analyzer_agent = Agent(
    name = 'Query Analyzer Agent',
    instructions='''You are a query analyzer agent , you analyze the query  and return the answer in one of the three types : refund , billing , technical''',
    model=model,
    output_type=QueryType
)
def on_handoff(ctx):
    print('Handoff Triggered')
@function_tool(name_override="Process_And_Delegate")
async def process_and_delegate(query : str)-> Handoff:
    query_input = f'''Analyze this query : {query}'''
    result = await Runner.run(
        starting_agent=query_analyzer_agent,
        input = query_input,
        run_config=config
        )
    query_type = result.final_output
    print(f'User query is {query_type.query_type}')
    print(type(query_type.query_type))
    if query_type.query_type == 'refund':
        return Handoff(
            agent_name='Refund Agent',
            tool_name ='Transfer_to_Refund_Agent',
            tool_description='''Transfer the query to the refund agent''',
            input_json_schema={},
            on_invoke_handoff=on_handoff
        )
    elif query_type.query_type == 'billing':
        return Handoff(
            agent_name='Billing_Agent',
            tool_name='Transfer_to_Billing_Agent',
            tool_description='''Transfer the query to the billing agent''',
            input_json_schema={},
            on_invoke_handoff=on_handoff
        )
    elif query_type.query_type == 'technical':
        return Handoff(
             agent_name='Technical_Agent',
             tool_name='Transfer_to_Technical_Agent',
             tool_description='''Transfer the query to the technical agent''',
             input_json_schema={},
             on_invoke_handoff=on_handoff

        )
    else:
        return Handoff(
            agent_name='Basic Agent',
            tool_name='Transfer_to_Basic_Agent',
            tool_description='''Transfer the query to the basic agent''',
            input_json_schema={},
            on_invoke_handoff=on_handoff
        )
basic_agent = Agent(
    name = 'Basic Agent',
    instructions = ''' You are a customer Support Assistant , you have to call the tool to analyze the query and then answer it accoding to the query type''',
    model=model,
    tools=[process_and_delegate]
)

async def run_agent():
    result = await Runner.run(
        starting_agent=basic_agent,
        input='I have purchase the product and mistakenly got the wrong product , can i refund it??',
        run_config=config
    )
    print(result.final_output)
    print(result.last_agent)

asyncio.run(run_agent())