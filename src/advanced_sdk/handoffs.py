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
from typing import Any, List
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
openai_api_key=os.getenv("OPENAI_API_KEY")
# weave.init("Advanced_SDK")
# Set up tracing with Weave
# set_trace_processors([WeaveTracingProcessor()])
# agent_ops_api_key = os.getenv("AGENTOPS_API_KEY")
# langmsith_api_key = os.getenv("LANGSMITH_API_KEY")
# set_trace_processors([OpenAIAgentsTracingProcessor()])
# agentops.init(api_key=agent_ops_api_key)
# mlflow.openai.autolog()
# mlflow.set_tracking_uri("http://localhost:8000")
# mlflow.set_experiment("OpenAI Agent")
# set_trace_processors(
#     [
#         KeywordsAITraceProcessor(
#             api_key=os.getenv("KEYWORDSAI_API_KEY"),
#             endpoint="https://api.keywordsai.co/api/openai/v1/traces/ingest"
#         ),
#     ]
# )
# Check if the API key is present; if not, raise an error
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
@function_tool
def get_weather(location: str) -> str:
    '''
    Get the weather for a given location
    Args:
        location (str): The location to get the weather for
    Returns:
    str: The weather    
    '''
    return f"The weather in {location} is sunny"
@function_tool
def get_user_id() -> str:
    '''
    Returns : Get the user id from the DB for furhter processing 

    '''
    # Here we can get the user id from the DB and write the logic
    return uuid.uuid4().hex

@function_tool
def get_order_status(order_id : str) -> str:
    '''
    Get the order details/status against the order id

    Args:
        order_id (str): The order id to get the order details for
    Returns :
        str: The order details
    '''
    return f'''Your order status for the order id {order_id} is delivered'''

weather_agent = Agent(
    name = 'Weather Agent',
    instructions='''You help users with the weather information using the tool "get_weather"''',
    model=model,
    tools=[get_weather],
    handoff_description='''Get the weather of the specified location '''
)
order_agent = Agent(
    name = 'Order Agent',
    instructions='''You help users with the order information , checking the order status of the product against a order id using the tool "get_order_id"''',
    model=model,
    handoff_description='''Get the order status of the product against the order id ''',
    tools=[get_order_status]
)

basic_agent_instructions = '''
# Roles and Objecitves
You are a basic agent , that helps the user with their order queries and as a customer support , You help with the basic queries of the user , but you also have the specialized agents to work with who are specialized in getting the order status of the product
# Instructions
- Analyze the query and 
- If the query is general , respond to them generically
- If the user ask about checking the order status of the product , then you first you have to call the 'get_order_id' tool to get the order status of the product and then you should handoff to the order_agent to get the status of the product. (REMEMBER : You should have call first the tool and then handoff to the agent)
- The tool output that you get from the tool should be the input to the order_agent

# EXAMPLES :
## User : How can you help me ?
## Basic_Agent : You can ask me about the order status of the product and general queries.

## User : What is the order status of the product that i have ordered?
## Basic_Agent : --> Tool Called `get_order_id` --> Handoff to `order_agent`
## Order_Agent : The order status of the product that you have ordered is delivered

'''
def check_the_input_history(data : HandoffInputData)-> HandoffInputData:
    # For now just checking the input data passed to the handed off to the agent
    # print('Data passed to the order agent : ')
    # print(data.pre_handoff_items)
    # print(data.new_items)
    # print(data.input_history
          
    return data

# agent = Agent(
#     name = 'Basic Agent',
#     instructions=basic_agent_instructions,
#     model=model,
#     tools=[get_user_id],
#     handoffs=[handoff(
#         agent=order_agent,
#         tool_name_override='get_order_status_by_Order_agent',
#         input_filter=check_the_input_history
#     )]

# )
class Summary(BaseModel):
    summary : str
summarizer_agent = Agent(
    name = 'Summarizer Agent',
    instructions = 'You are a summarizer agent , you summarize the conversation and return the summary of the conversation to the user',
    model = model 
)
def summarize_conversation(data : HandoffInputData)-> HandoffInputData:
    print('Data passed to the suuport agent')
    # input_query = f'Summarize this conversation : {data.input_history}'
    # result = await Runner.run(
    #     starting_agent=summarizer_agent,
    #     input = input_query,
    #     run_config=config)
    # summary = result.final_output.summary
    # print('\n\n INPUT HISTORY \n\n')
    # print(data.input_history)
    # print('\n\n PRE HANDOFF ITEMS \n\n')
    # print(data.pre_handoff_items)
    # print('\n\n NEW ITEMS \n\n')
    # print(data.new_items)
    return data
    # return HandoffInputData(
    #     input_history=summary,
    #     pre_handoff_items=data.pre_handoff_items

    # )
support_agent = Agent(
    name = 'Support Agent',
    instructions='''You help users with their problems related to customer support and provide the guidance and the solution''',
    model=model,
    handoff_description='Help the users with the customer support queries'
)

technical_agent = Agent(
    name = 'Technical Agent',
    instructions='''You help users with the technical queries and issues''',
    model=model,
    handoff_description='''Assist users with the technical queries and issues'''
    
)

agent = Agent(
    name = 'Basic Agent',
    instructions=f'''You are a basic agent , you have a tool to get the weather of any city , when the user demands you can call the tool and get the weather of the required city , you also have a specialiazed agents in your workflow one is responsible for handling customer support quries , user issues and billing queries etc, while the other is responsible for handling the technical issues queries
    # Objectives :
    - Analyze the query 
    - if the user is demanding or asking about the weather of the any city use the tool to get the weather of the city
    - if the user's query is related to some customer support queries , then you should handoff to the support agent
    - if the  user's query is related to some technical queries , then you should handoff to the technical agent
    ''',
    tools=[get_weather],
    handoffs=[handoff(
        agent=support_agent,
        input_filter=summarize_conversation
        ) , handoff(
            agent=technical_agent
        )],
    model = model
    )
# Case 1 : IF the agent has a handoff available but we try the first agent to perform its task and then handoff to a specific agent 
# Yeah , it worked , we can first call the triage agent to do the action then it will handoff to the other agent if needed

# Case 2 : If both the tool and handoff are called simultaneously then what will happen ?

# Ans :  For this query : "Hi ! what is the weather in new york and i have ordered the product , but still can received yet ? help me with that ?"  ,firstly the tool is called and then handoff happened as expected and we get the response
# For this query : 'Hi ! I have ordered the product , but still can received yet ? firstly help me with that ? and then aftwerwards also tell me the weather in new york' , the handoff is done firstly and then as we know we have moved to the other agent , and the other agent dont have the access to the tool so  then there will be no tool call

# Case 3: When multiple handoffs are called , then what will happen ?

# Ans : The llm will initiate both the handoffs but the first one will be proceeded and the other will be ignored

# creating a custom handoff input filter in which we summarize all the conversation history and pass it to the next agent



async def run_agent():
    result = await Runner.run(
        starting_agent=agent,
        input='Hi !I ordered the product but got the wrong on delivery help me with that , and also there is a bug in the website can you make it fixed ?',
        run_config=config
    )
    print(result.final_output)
    print(result.last_agent.name)

asyncio.run(run_agent())