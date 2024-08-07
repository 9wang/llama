from llama_index.core import VectorStoreIndex,download_loader
from llama_index.readers.google import GoogleDocsReader
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Ollama(model="llama3.1:8b",request_timeout=600.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\workspace\llama_index\model\bge-small-zh-v1.5")



gdoc_ids = ["1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec"]
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)
index = VectorStoreIndex.from_documents(documents,embedding=Settings.embed_model)
query_engine = index.as_query_engine(llm=Settings.llm)
res = query_engine.query("请帮我总结一下这个文档的内容")
print(res)