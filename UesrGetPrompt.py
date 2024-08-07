from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\workspace\llama_index\model\bge-small-zh-v1.5")

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

from llama_index.core import(
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    PromptTemplate,
)

from IPython.display import display, Markdown


documents = SimpleDirectoryReader('./data/paul_graham').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(response_mode = "tree_summarize")

# query_engine = index.as_query_engine(response_mode = "compact")
# from llama_index.core import PromptTemplate
# 重置query_engine = index.as_query_engine(response_mode="tree_summarize")
# new_summary_tmpl_str = (    "下面是上下文信息。\n"    
#                         "---------------------\n"    
#                         "{context_str}\n"    
#                         "---------------------\n"
#                         "根据上下文信息和非先验知识，以莎士比亚戏剧的风格回答查询。\n"
#                         "查询：{query_str}\n"
#                         "答案："
#                         )
# new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
# query_engine.update_prompts(
#     {"response_synthesizer:summary_template": new_summary_tmpl}
# )

def display_prompt_dict(prompts_dict):
    for k,p in prompts_dict.items():
        text_md = f"**提示键**:{k}<br>" f"**文本:**<br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))

# prompts_dict = query_engine._response_synthesizer.get_prompts()


# response = query_engine.query("What did the author do growing up?")
# print(response)
    

from llama_index.core.query_engine import (
    RouterQueryEngine,
    FLAREInstructQueryEngine
)
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.evaluation import FaithfulnessEvaluator,DatasetGenerator
from llama_index.core.postprocessor import LLMRerank

# 设置样本路由查询引擎
# from llama_index.core.tools import QueryEngineTool
# query_tool = QueryEngineTool.from_defaults( query_engine=query_engine,
#                                             description="测试描述")
# router_query_engine = RouterQueryEngine.from_defaults([query_tool])

# flare_query_engine = FLAREInstructQueryEngine(query_engine)
# prompts_dict = flare_query_engine.get_prompts()
# display_prompt_dict(prompts_dict)
selector = LLMMultiSelector.from_defaults()
prompts_dict = selector.get_prompts()
display_prompt_dict(prompts_dict)


# prompts_dict = query_engine.get_prompts()
# display_prompt_dict(prompts_dict)
