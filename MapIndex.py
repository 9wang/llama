from llama_index.core import PropertyGraphIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings,VectorStoreIndex,SimpleDirectoryReader

Settings.llm = Ollama(model="gemma2:2b",request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(model_name=r"D:\workspace\llama_index\model\bge-large-zh")

documents = SimpleDirectoryReader(
    input_files=[r"D:\workspace\llama_index\llamaindex\data\doc.txt"]
).load_data()

##创建
index = PropertyGraphIndex.from_documents(documents,embed_model=Settings.embed_model)

##使用
retriever =index.as_retriever(
    include_text=True,##包含与匹配路径的源块
    similarity_top_k=2,
)
nodes = retriever.retrieve("Test")

query_engine = index.as_query_engine(
    include_text=True,##包含与匹配路径的源块
    similarity_top_k=2,
    llm=Settings.llm,
)
response = query_engine.query("Test")
print(index)
print("--------------------------------------")
print(response)

##保存与加载
index.storage_context.persist(persist_dir="./storage")

# from llama_index.core import StorageContext,load_index_from_storage
