from llama_index.core import Document,SimpleDirectoryReader,VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionCache,IngestionPipeline

from llama_index.vector_stores.qdrant import QdrantVectorStore##用于创建和管理向量库
import qdrant_client##用于与Qdrant向量库进行交互

client = qdrant_client.QdrantClient(path="./pipeline_storage")
vector_store = QdrantVectorStore(client=client,collection_name="test_store")

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="gemma2:2b",request_timeout=600.0)


Settings.embed_model = HuggingFaceEmbedding(model_name="../model/bge-large-zh")


# documents = SimpleDirectoryReader(input_files=[r"D:\workspace\llama_index\QA\data\标准.docx"]).load_data()
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(
            chunk_size=25, chunk_overlap=1
        ),
        TitleExtractor(),
        Settings.embed_model,
    ],
    # vector_store=vector_store,
)
pipeline.persist("./pipeline_storage")

new_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(
            chunk_size=25, chunk_overlap=1
        ),
        TitleExtractor(),
    ]
)

new_pipeline.load("./pipeline_storage")
nodes = pipeline.run(documents=[Document.example()])
# pipeline.run(documents=[Document.example()])

# index = VectorStoreIndex.from_vector_store(vector_store,embed_model=Settings.embed_model)
# nodes = pipeline.run(documents=[Document.example()])
# print(Document.example())
# print(nodes)
