import nest_asyncio
nest_asyncio.apply()

##设置数据
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode

docs0=PyMuPDFReader().load(file_path=("./data/llama2.pdf"))
# print("----------------------------------------------")
# print(docs0)
doc_text="\n\n".join([d.get_content() for d in docs0])##输出切分后的文档
# print("----------------------------------------------")
# print(doc_text)
docs=[Document(text=doc_text)]
# print("----------------------------------------------")
# print(docs)
node_parser=SentenceSplitter(chunk_size=1024)
# print("----------------------------------------------")
# print(node_parser)
base_nodes=node_parser.get_nodes_from_documents(docs)
# print("----------------------------------------------")
# print(base_nodes)

#在这些数据上设置向量索引
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\workspace\llama_index\model\bge-small-zh-v1.5"
)

index=VectorStoreIndex(base_nodes,embed_model=Settings.embed_model)
query_engine=index.as_query_engine(similarity_top_k=2,llm=Settings.llm)


#评估设置
##加载数据集
from llama_index.core.evaluation import QueryResponseDataset

eval_dataset=QueryResponseDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

##获得评估器
from llama_index.core.evaluation.eval_utils import get_responses
from llama_index.core.evaluation import CorrectnessEvaluator,BatchEvalRunner

evaluator_C=CorrectnessEvaluator()
evaluator_dict={"correctness":evaluator_C}
batch_runner=BatchEvalRunner(evaluator_dict,workers=2,show_progress=True)

##定义正确性评估函数
import numpy as np

async def get_correctness(query_engine,eval_qa_pairs,batch_runner):
    ###进行评估
    ###TODO：评估生成结果的样本
    eval_qa=[q for q, _ in eval_qa_pairs]
    eval_answers =[a for _, a in eval_qa_pairs]
    pred_responses=get_responses(eval_qa,query_engine,show_progress=True)

    eval_results = await batch_runner.aevaluate_responses(
        eval_qa,responses=pred_responses,reference=eval_answers
    )
    avg_correctness=np.array(
        [r.score for r in eval_results["correctness"]]
    )
    return avg_correctness

#尝试情绪提示
##从论文中提取了一些情绪刺激来尝试。
emotion_stimuli_dict = {
    "ep01": "Write your answer and give me a confidence score between 0-1 for your answer. ",
    "ep02": "This is very important to my career. ",
    "ep03": "You'd better be sure.",
    # add more from the paper here!!
}

# NOTE: ep06 is the combination of ep01, ep02, ep03
emotion_stimuli_dict["ep06"] = (
    emotion_stimuli_dict["ep01"]
    + emotion_stimuli_dict["ep02"]
    + emotion_stimuli_dict["ep03"]
)


##初始化基本的问答提示
QA_PROMPT_KEY = "response_synthesizer:text_qa_template"
from llama_index.core import PromptTemplate

qa_tmpl_str = """\
下面是上下文信息。
---------------------
{context_str}
---------------------
根据上下文信息和没有先验知识，回答问题。
{emotion_str}
问题: {query_str}
答案: \
"""
qa_tmpl = PromptTemplate(qa_tmpl_str)


##在情绪前添加情感

async def run_and_evaluate(
        query_engine,eval_qa_pairs,batch_runner,emotion_stimuli_str,qa_tmpl
):
    """运行并评估"""
    new_qa_tmpl=qa_tmpl.partial_format(emotion_str=emotion_stimuli_str)

    old_qa_tmpl=query_engine.get_prompts()[QA_PROMPT_KEY]
    query_engine.update_prompts({QA_PROMPT_KEY:new_qa_tmpl})
    avg_correctness=await get_correctness(
        query_engine,eval_qa_pairs,batch_runner
    )
    query_engine.updata_prompts({QA_PROMPT_KEY:old_qa_tmpl})
    return avg_correctness

async def main():
    correctness_ep01 = await run_and_evaluate(
        query_engine,
        eval_dataset.qr_pairs,
        batch_runner,
        emotion_stimuli_dict["ep01"],
        qa_tmpl,
    )
    print(correctness_ep01) 

import asyncio
asyncio.run(main())

