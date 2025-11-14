import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool('get_weather', description="Get the current weather for a given city.", return_direct=False)
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

agent = create_agent(
    model='mistral-large-latest',
    tools=[get_weather],
    system_prompt='You are a  helpful weather assistant, who always cracks jokes and is humorous while remaining helpful',
)

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': '北京什么天气？'}
    ]
})

print(response)