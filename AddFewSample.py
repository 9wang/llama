from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate



Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\workspace\llama_index\model\bge-small-zh-v1.5")

few_shot_nodes = []
for line in open("./llama2_qa_citation_events.jsonl","r"):
    few_shot_nodes.append(TextNode(text=line))

few_shot_index = VectorStoreIndex(few_shot_nodes)
few_shot_retriever=few_shot_index.as_retriever(similarity_top_k=2)

import json

def few_shot_examples_fn(**kwargs):
    query_str = kwargs["query_str"]
    retrieved_nodes=few_shot_retriever.retrieve(query_str)

    result_strs=[]
    for n in retrieved_nodes:
        raw_dict=json.loads(n.get_content())
        query=raw_dict["query"]
        response_dict=json.loads(raw_dict["response"])
        result_str=f"""\
Query: {query}
Response: {response_dict}"""
        result_strs.append(result_str)

    return "\n\n".join(result_strs)

##编写带有函数的提示模板
qa_prompt_tmpl_str = """\
下面是上下文信息。
---------------------
{context_str}
---------------------
根据上下文信息和非先验知识，回答有关不同主题引用的查询。
请以结构化的JSON格式提供您的答案，其中包含作者列表作为引用。以下是一些示例。

{few_shot_examples}

查询：{query_str}
答案：\
"""

qa_prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str,
    function_mappings={"few_shot_examples": few_shot_examples_fn},
)

citation_query_str =(
    "Which citations are mentioned in the section on Safety RLHF?"
)

print(
    qa_prompt_tmpl.format(
        query_str=citation_query_str,context_str="test_context"
    )
)
def display_prompt_dict(prompts_dict):
    for k,p in prompts_dict.items():
        text_md = f"**提示键**:{k}<br>" f"**文本:**<br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))

from IPython.display import display,Markdown

from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
loader = PyMuPDFReader()
documents = loader.load_data(Path("data/llama2.pdf"))

from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents,embed_model=Settings.embed_model)


query_engine = index.as_query_engine(similarity_top_k=2, llm=Settings.llm)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)
display_prompt_dict(query_engine.get_prompts())

response = query_engine.query(citation_query_str)
print(response)

print(response.source_nodes[1].get_content())