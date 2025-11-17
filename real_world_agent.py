from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
# 短期记忆 记录单个 thread_id 的交互记录
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import requests
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a  helpful weather assistant, who always cracks jokes and is humorous while remaining helpful"""

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
)

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather_for_location, get_user_location],
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
    {"messages": [{"role": "user", "content": "What you said just now?"}]},
    config=config,
    context=Context(user_id='u123')
)
print(response['structured_response'])


