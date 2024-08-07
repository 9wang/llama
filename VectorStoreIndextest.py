from llama_index.core import download_loader
from llama_index.readers.file import PyMuPDFReader

llama2_docs = PyMuPDFReader().load_data(
    file_path="./data/llama2.pdf",metadata=True
)
attention_docs = PyMuPDFReader().load_data(
    file_path="./data/attention.pdf",metadata=True
)

from llama_index.core.node_parser import TokenTextSplitter

nodes = TokenTextSplitter(
    chunk_size=1024, chunk_overlap=128
).get_nodes_from_documents(llama2_docs+attention_docs)

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.docstore.firestore import FirestoreDocumentStore
from llama_index.storage.docstore.dynamodb import DynamoDBDocumentStore

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="../model/bge-large-zh")
Settings.llm = Ollama(model="gemma2:2b")

client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore("composable",client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store,docstore=docstore)

index = VectorStoreIndex(nodes=nodes,embed_model=Settings.embed_model)
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
    text="BM25 retriever",
)

from llama_index.core import SummaryIndex
summary_index = SummaryIndex(objects=[vector_obj,bm25_obj])
# summary_index = SummaryIndex(objects=[bm25_obj])

query_engine = summary_index.as_query_engine(
    llm=Settings.llm,response_mode="tree_summarize",verbose=True,
)

response =  query_engine.query(
    "How does attention work in transformers?"
)

print(str(response))