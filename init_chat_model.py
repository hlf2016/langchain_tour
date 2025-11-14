from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = init_chat_model(
    model='mistral-medium-2508',
    temperature=0.1
)

conversations = [
    SystemMessage('You are a helpful assistant for questions regarding programming'),
    HumanMessage('What is Python?'),
    AIMessage('Python is an interpreted programming language'),
    HumanMessage('When was it released')
]

# response = model.invoke(conversations)
# print(response.content)

# stream  response
for chunk in model.stream(conversations):
    print(chunk.text, end='', flush=True)

