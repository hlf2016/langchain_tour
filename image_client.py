from base64 import b64encode
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model('gpt-4.1-mini')

# 网络图片
# message = {
#     'role': 'user',
#     'content': [
#         {'type': 'text', 'text': 'Describe the content of this image.'},
#         {'type': 'image', 'url': 'https://yt3.ggpht.com/SssDhka9PcZsgvC1c12dLw7zE9r1lbBdGeUuay5jHumwYPnfJX7v0LSdcpnyGhFvEgWbPN4Z=s88-c-k-c0x00ffffff-no-rj'}
#     ]
# }

# 本地图片
with open('/Users/aaron/Downloads/channels4_profile.jpg', 'rb') as f:
    image_base64 = b64encode(f.read()).decode()
message = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe the content of this image.'},
        {'type': 'image', 'base64': image_base64, 'mime_type': 'image/jpg'}
    ]
}

response = model.invoke([message])
print(response.content)

