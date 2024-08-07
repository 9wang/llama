# from llama_index.core import Document
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.extractors import TitleExtractor
# from llama_index.core.ingestion import IngestionPipeline, IngestionCache
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings
# from llama_index.vector_stores.qdrant import QdrantVectorStore

# Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name=r"D:\workspace\llama_index\model\bge-small-zh-v1.5")

# import qdrant_client
# client = qdrant_client.QdrantClient(location=":memory:")
# vector_store = QdrantVectorStore(client=client,collection_name="test_store")

# pipeline = IngestionPipeline(
#     transformations=[
#         SentenceSplitter(chunk_size=25,chunk_overlap=0),
#         TitleExtractor(),
#         Settings.embed_model,
#     ]
#     vector_store=vector_store,
# )
# ##直接将数据摄取到向量数据库
# pipeline.run(documents=[Document.example()])

# ##创建索引
# from llama_index.core import VectorStoreIndex

# index = VectorStoreIndex.from_vector_store(vector_store)

# #保存
# pipeline.presist("./pipeline_storage")

# #加载和恢复状态
# # new_pipeline - IngestionPipeline(
# #     transformations=[
# #         SentenceSplitter(chunk_size=25,chunk_overlap=0),
# #         TitleExtractor(),
# #     ]
# # )
# # new_pipeline.load("./pipeline_storage")





# # nodes = pipeline.run(documents=[Document.example()])
# # print(nodes)
# # print("-------------------------------------")
# # print(Document.example().text)