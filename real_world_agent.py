from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
# 短期记忆 记录单个 thread_id 的交互记录
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import requests
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

@tool('get_weather_for_location', description="Get the current weather for a given city.", return_direct=False)
def get_weather_for_location(city: str) -> dict:
    """Get the current weather for a given city."""
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool('get_user_location', description="Retrieve user information based on user ID.")
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    mock_db = {
        'u123': 'London',
        'u456': 'Beijing',
        'u789': 'Tokyo'
    }
    return mock_db.get(user_id, 'Guangzhou')

model = init_chat_model(
    model='gpt-4.1-mini',
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punnyResponse: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'what is the weather outside?'}]},
    config=config,
    context=Context(user_id='u123')
)

print(response['structured_response'])

# 采用相同的 thread id 接着进行交流
response = agent.invoke(
    {"messages": [{"role": "user", "content": "how are you?"}]},
    config=config,
    context=Context(user_id='u123')
)
print(response['structured_response'])


