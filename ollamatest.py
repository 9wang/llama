from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

prompt_sys ="你现在的身份是元数智慧科技有限公司的客服助手小元"

llm = Ollama(model="llama3.1:8b",request_timeout=120.0,prompt_key="prompt_sys")
#ollama支持json模式设置json_mode=True
response = llm.complete("那你是谁？")
# print(response)

messages = [
    ChatMessage(
        role="system",content="你是元数智慧科技有限公司的客服助手小元"
    ),
    ChatMessage(role="user",content="What is your name"),
]
# print(messages)
# resp = llm.complete("Who are you?")

# 流式输出
# response = llm.stream_complete("Who are you")
# resp = llm.stream_chat(messages) 
# for r in resp:
#     print(r.delta,end="")

resp = llm.chat(messages)
print(resp)