from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\workspace\llama_index\model\bge-small-zh-v1.5")

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
from IPython.display import display,Markdown

from pathlib import Path
from llama_index.readers.file import PyMuPDFReader

loader = PyMuPDFReader()
documents = loader.load_data(Path("data/llama2.pdf"))

from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents,embed_model=Settings.embed_model)

query_str = "What are the potential risks associated with the use of Llama 2 as mentioned in the context?"

query_engine = index.as_query_engine(similarity_top_k=2, llm=Settings.llm)
vector_retriever = index.as_retriever(similarity_top_k=2)

# response = query_engine.query(query_str)
# print(response)


##查看提示
def display_prompt_dict(prompts_dict):
    for k,p in prompts_dict.items():
        text_md = f"**提示键**:{k}<br>" f"**文本:**<br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))

# prompts_dict = query_engine.get_prompts()

# display_prompt_dict(prompts_dict)

from langchain import hub
from llama_index.core.prompts import LangchainPromptTemplate

langchain_prompt = hub.pull("rlm/rag-prompt")

lc_prompt_tmpl=LangchainPromptTemplate(
    template=langchain_prompt,
    template_var_mappings={"query_str":"question","context_str":"context"}
)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template":lc_prompt_tmpl}
)

prompts_dict = query_engine.get_prompts()

display_prompt_dict(prompts_dict)

response = query_engine.query(query_str)
print(response)