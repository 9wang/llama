import nest_asyncio
nest_asyncio

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\workspace\llama_index\model\bge-large-en"
)
Settings.chunk_size = 512

from IPython.display import Markdown,display,HTML

documents = SimpleDirectoryReader(
    input_files=["D:/workspace/llama_index/llamaindex/data/Great.txt"]).load_data()

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model,
)

#检索
##VectorIndexRetriever用于根据向量索引检索数据
##QueryBundle用于处理查询
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
import pandas as pd
# from FlagEmbedding import FlagReranker
# from llama_index.core.llms import LLM



pd.set_option("display.max_colwidth",None)
def get_retrieved_nodes(
        query_str,
        vector_top_k=10,##向量检索时返回top_k个结果
        reranker_top_n=3,##reranker返回的top_n个结果
        with_reranker=True##是否使用reranker
):
    query_bundle=QueryBundle(query_str)

    ##配置VectorIndexRetriever检索器
    retriever=VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes=retriever.retrieve(query_bundle)

    if with_reranker:

        reranker=LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
            llm=Settings.llm
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes,
            query_bundle,
        )
    return retrieved_nodes

def save_to_html(df, filename):
    """保存 DataFrame 到 HTML 文件"""
    html_content = df.to_html().replace("\\n", "<br>")
    with open(filename, "w",encoding="utf-8") as f:
        f.write(html_content)

def visualize_retrieved_nodes(nodes,output_file="output.html"):
    result_dicts=[]
    for node in nodes:
        result_dict={"Score":node.score,"Text":node.node.get_text()}
        result_dicts.append(result_dict)
    df = pd.DataFrame(result_dicts)
    save_to_html(df,output_file)
    print(f"已保存到 {output_file}")

new_nodes = get_retrieved_nodes(
    "Who was driving the car that hit Myrtle?",
    vector_top_k=5,
    reranker_top_n=3,
    with_reranker=True,
)

# visualize_retrieved_nodes(new_nodes)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessor=[
        LLMRerank(
            choice_batch_size=5,
            top_n=2,
        )
    ],
    response_mode="tree_summarize",
    llm=Settings.llm
)

response = query_engine.query("Who is Daisy?")
print(response)