# from llama_index.core import PromptTemplate

# template = (
#     "我们已经提供了下面的上下文信息。 \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "根据这些信息，请回答问题：{query_str}\n"
# )

# qa_template = PromptTemplate(template)
# print(qa_template)
# prompt = qa_template.format(context_str = ...,query_str=...)
# print(prompt)
# messages = qa_template.format_messages(context_str = ...,query_str=...)
# print(messages)

from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage,MessageRole

messages_template = [
    ChatMessage(content="You are an expert system.",role=MessageRole.SYSTEM),
    ChatMessage(
        content="Generate a short story about {topic}",
        role=MessageRole.USER
    )
]

chat_template = ChatPromptTemplate(message_templates=messages_template)

messages = chat_template.format_messages(topic="metaAI")
print(messages)

prompt = chat_template.format(topic="metaAI")
print(prompt)

query_engine= index.as_query_engine