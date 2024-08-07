from qdrant_client import QdrantClient
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    SummaryIndex,
    load_index_from_storage
)
from llama_index.core.schema import IndexNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.node_parser import SentenceWindowNodeParser,SentenceSplitter ##窗口节点
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

import os


Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(model_name=r"/home/ubuntu/RAG/models/bge-large-zh")

def load_documents(file_paths):
    documents = SimpleDirectoryReader(input_files=file_paths).load_data()
    # for path in file_paths:
    return documents


def parse_nodes(documents):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes

def manage_docstore(nodes,persist_path= "./docstore.json"):
    if not os.path.exists(persist_path):    
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        ##保存
        docstore.persist("./docstore.json")
    else:
        docstore = SimpleDocumentStore.from_persist_path(persist_path)
    return docstore

def initialize_retrievers(docstore,nodes):
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
    return vector_retriever,bm25_retriever,storage_context

def initialize_summary_index(vector_retriever,bm25_retriever,persist_path="./summary_index.json"):
    if not os.path.exists(persist_path):    
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
        summary_index=SummaryIndex(objects=[vector_obj,bm25_obj])
        summary_index.storage_context.persist(persist_path)
    else:
        storage_context=StorageContext.from_defaults(persist_path)
        summary_index=load_index_from_storage(storage_context)
    return summary_index

def initialize_query_engine(summary_index):
    qurey_engine = summary_index.as_query_engine(
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
    return qurey_engine

if __name__ == "__main__":
    
    document_paths = ["./data/大会.docx", "./data/AGIC大会.docx"]
    documents = load_documents(document_paths)
    nodes = parse_nodes(documents)
    docstore = manage_docstore(nodes)
    
    vector_retriever, bm25_retriever,storage_context = initialize_retrievers(docstore,nodes)
    summary_index = initialize_summary_index(vector_retriever, bm25_retriever)
    query_engine = initialize_query_engine(summary_index)
    
    window_response = query_engine.query("介绍一下这次大会主要的目的")

    print(window_response)
    print("---------------------------------------")

# window = window_response.source_nodes[0].node.metadata(["window"])
# sentence = window_response.source_nodes[0].node.metadata(["original_text"])


##运行前先起ollama
# print(f"window: {window}")
# print("---------------------------------------")
# print(f"sentence: {sentence}")
# noeds = splitter.get_nodes_from_documents(documents)
# print(nodes)