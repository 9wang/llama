from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\workspace\llama_index\model\bge-large-en"
)

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./data/pg_essay.txt"]
).load_data()

from llama_index.core.node_parser import SemanticSplitterNodeParser,SentenceSplitter
#实例化SemanticSplitterNodeParser模型
splitter = SemanticSplitterNodeParser(
    buffer_size=1,breakpoint_percentile_threshold=95,embed_model=Settings.embed_model
)
base_splitter = SentenceSplitter(chunk_size=512)

nodes = splitter.get_nodes_from_documents(documents)
# print("---------------------------------------------------------")
# print(nodes[1].get_content())
# print("---------------------------------------------------------")
# print(nodes[2].get_content())
# print("---------------------------------------------------------")
# print(nodes[3].get_content())
# print("---------------------------------------------------------")

base_nodes = base_splitter.get_nodes_from_documents(documents)
# print(base_nodes[2].get_content())
# print("---------------------------------------------------------")

from llama_index.core import VectorStoreIndex
from llama_index.core.response.notebook_utils import display_source_node

vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine()

base_vector_index = VectorStoreIndex(base_nodes)
base_query_engine = base_vector_index.as_query_engine()


response = query_engine.query("Tell me about the author's programming journey through childhood to college")
base_response = base_query_engine.query("Tell me about the author's programming journey through childhood to college")
print("---------------------------------------------------------")
for n in base_response.source_nodes:
    print("Node ID:", n.node_id)
    print("Similarity:", n.score)
    print("Text:\n", n.get_content())
    print("\n")
print("---------------------------------------------------------")    

# base_response = base_query_engine.query("Tell me about the author's programming journey through childhood to college")
# for n in base_response.source_nodes:
#     print(display_source_node(n,source_length=20000))
# print("---------------------------------------------------------")
# print("---------------------------------------------------------")
# print(str(response))
# print("---------------------------------------------------------")
# print(str(base_response))