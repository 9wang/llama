from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(model_name=r"/home/ubuntu/RAG/models/bge-large-zh")


dahui_doc = SimpleDirectoryReader(input_files=["./data/大会.docx","./data/AGIC大会.docx"]).load_data()
# ##token分块
# from llama_index.core.node_parser import TokenTextSplitter
# splitter = TokenTextSplitter(
#     chunk_size = 256,
#     chunk_overlap=128,
#     separator="  "
# )

##语义分块
# from llama_index.core.node_parser import SemanticSplitterNodeParser
# splitter = SemanticSplitterNodeParser(
#     buffer_size=1,breakpoint_percentile_threshold=95,embed_model=Settings.embed_model
# )

# ##句子分块，尊重句子边界
# from llama_index.core.node_parser import SentenceSplitter
# splitter = SentenceSplitter(
#     chunk_size=400,
#     chunk_overlap=64,
# )

##窗口节点
from llama_index.core.node_parser import SentenceWindowNodeParser,SentenceSplitter
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

nodes = node_parser.get_nodes_from_documents(dahui_doc)

from llama_index.core.storage.docstore import SimpleDocumentStore
import os
PERSIST_DOC = "./docstore.json"
if not os.path.exists(PERSIST_DOC):
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    ##保存
    docstore.persist("./docstore.json")
else:
    ##加载
    docstore = SimpleDocumentStore.from_persist_path(PERSIST_DOC)

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core import StorageContext

client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore("composable",client=client)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,docstore=docstore
)

index = VectorStoreIndex(nodes,embed_model=Settings.embed_model)
vector_retriever = index.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore,similarity_top_k=2
)

from llama_index.core.schema import IndexNode

vector_obj = IndexNode(
    index_id="vector",
    obj=vector_retriever,
    text="Vector retriever",
)
bm25_obj = IndexNode(
    index_id="bm25",
    obj=bm25_retriever,
    text="BM25 retriever"
)

from llama_index.core import SummaryIndex
summary_index=SummaryIndex(objects=[vector_obj,bm25_obj])

from llama_index.core.postprocessor import MetadataReplacementPostProcessor
# from llama_index.core.tools import QueryEngineTool,ToolMetadata
# from llama_index.core.callbacks import CallbackManager,LlamaDebugHandler
# from llama_index.core.query_engine import SubQuestionQueryEngine
# llama_debug = LlamaDebugHandler(print_trace_on_end=True)
# callback_manager = CallbackManager([llama_debug])
# Settings.callback_manager = callback_manager

# vector_query_engine = VectorStoreIndex.from_documents(
#     dahui_doc,
#     use_async=True,
# ).as_query_engine()

# query_engine_tools = [
#     QueryEngineTool(
#         query_engine=vector_query_engine,
#         metadata=ToolMetadata(
#             name="dahui",
#             description="AIGC大会相关介绍",
#         ),
#     ),
# ]

# query_engine = SubQuestionQueryEngine.from_defaults(
#     query_engine_tools=query_engine_tools,
# )


# response = query_engine.query(
#     "大会的主要目的是什么"
# )
# print(response)
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage,MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine

custom_prompt = PromptTemplate(
    """\
给定一段对话（人类和助手之间）以及人类的后续消息，重写消息以成为一个独立的问题，其中包含对话的所有相关上下文。

<聊天历史>
{chat_history}

<后续消息>
{question}

<独立问题>
"""
)

#对象列表
custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="你好助手，今天我们正在对人工智能协会以及AGIC大会进行深入讨论。"
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT,
        content="好的，听起来很不错"),
]

query_engine = summary_index.as_query_engine(
    similar_top_k=2,
    response_mode="tree_summarize",
    verbose=True,
    llm=Settings.llm,
    node_postprocessors=[
        MetadataReplacementPostProcessor(
            target_metadata_key="window",
        )
    ]
)
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=custom_prompt,
    chat_history=custom_chat_history,
    verbose=True
)
chat_engine.chat_repl()



# chat_engine = summary_index.as_chat_engine(
#     chat_mode="context",
#     similar_top_k=2,
#     response_mode="tree_summarize",
#     verbose=True,
#     llm=Settings.llm,
#     node_postprocessors=[
#         MetadataReplacementPostProcessor(
#             target_metadata_key="window",
#         )
#     ]
# )
# chat_engine.chat_repl()
# window_response = chat_engine.chat("介绍一下这次大会的规模")

# window = window_response.source_nodes[0].node.metadata(["window"])
# sentence = window_response.source_nodes[0].node.metadata(["original_text"])


##运行前先起ollama
# print(window_response)
# print("---------------------------------------")
# print(f"window: {window}")
# print("---------------------------------------")
# print(f"sentence: {sentence}")
# noeds = splitter.get_nodes_from_documents(documents)
# print(nodes)