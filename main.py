from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# accurate task
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

openai_llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1:free",
    temperature=0.9
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Chinese. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = openai_llm.invoke(messages)
print(ai_msg.content)
