# from llama_index.core import Document,SimpleDirectoryReader
# from llama_index.core.node_parser import SentenceSplitter

# node_parser = SentenceSplitter(chunk_size=20, chunk_overlap=1)
# # document = SimpleDirectoryReader(
# #     input_files=[r"D:\workspace\llama_index\llamaindex\data\paul_graham\paul_graham_essay.txt"])
# text = '''What I Worked On
# February 2021

# Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

# The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.

# The language we used was an early version of Fortran. You had to type programs on punch cards, then stack them in the card reader and press a button to load the program into memory and run it. The result would ordinarily be to print something on the spectacularly loud printer.

# I was puzzled by the 1401. I couldn't figure out what to do with it. And in retrospect there's not much I could have done with it. The only form of input to programs was data stored on punched cards, and I didn't have any data stored on punched cards. The only other option was to do things that didn't rely on any input, like calculate approximations of pi, but I didn't know enough math to do anything interesting of that type. So I'm not surprised I can't remember any programs I wrote, because they can't have done much. My clearest memory is of the moment I learned it was possible for programs not to terminate, when one of mine didn't. On a machine without time-sharing, this was a social as well as a technical error, as the data center manager's expression made clear.

# With microcomputers, everything changed. Now you could have a computer sitting right in front of you, on a desk, that could respond to your keystrokes as it was running instead of just churning through a stack of punch cards and then stopping. [1]
# '''


# nodes = node_parser.get_nodes_from_documents(
#     [Document(text=text)],show_progress=True
# )
# print(nodes)


##构建索引时自动使用节点解析器
# from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core import Settings

# documents = SimpleDirectoryReader("./data/paul_graham/paul_graham_esaay.txt").load_data()

# Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# index = VectorStoreIndex.from_documents(
#     documents,
#     transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)],
# )


# 基于文件的节点解析器
##SimpleFileNodeParser
# from llama_index.core.node_parser import SimpleFileNodeParser
# from llama_index.readers.file import FlatReader
# from pathlib import Path

# md_docs = FlatReader().load_data(
#     Path("./data/paul_graham/document.md")
# )
# parser = SimpleFileNodeParser()

# md_nodes = parser.get_nodes_from_documents(md_docs)

# print(md_nodes)

##HTMLNodeParser
'''
该节点解析器使用 beautifulsoup 解析原始 HTML。
默认情况下，它会解析一组选定的 HTML 标签，但您可以覆盖这一设置。
默认标签包括：["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "b", "i", "u", "section"]
'''
# from llama_index.core.readers.file import HTMLNodeParser
# parser = HTMLNodeParser(tags=["p","h1"])
# nodes = parser.get_nodes_from_documents(html_docs)


# CodeSplitter 
# from llama_index.core.node_parser import CodeSplitter
# splitter  = CodeSplitter(
#     language="python",
#     chunk_line=40,
#     chunk_lines_overlap=15,
#     max_chars=1500,
#  )

# nodes = splitter.get_nodes_from_documents(documents)

# LangchainNodeParser#
# 还可以使用节点解析器包装 langchain 中的任何现有文本分割器。

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from llama_index.core.node_parser import LangchainNodeParser

# parser = LangchainNodeParser(RecursiveCharacterTextSplitter())
# nodes = parser.get_nodes_from_documents(documents)

#SentenceSplitter
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core import SimpleDirectoryReader

# documents = SimpleDirectoryReader(
#     input_files=["./data/doc.txt"]).load_data()

# splitter = SentenceSplitter(
#     chunk_size=1024,
#     chunk_overlap=20
# )
# nodes = splitter.get_nodes_from_documents(documents)
# print(nodes)

import nltk
from llama_index.core.node_parser import SentenceWindowNodeParser,SentenceSplitter

node_parser = SentenceWindowNodeParser.from_defaults(
    #捕获两侧句子的数量
    window_size=3,
    #包含周围句子的窗口的元数据键
    window_metadata_key="window",
    #包含原始数据的元数据键
    original_text_metadata_key="original_text",
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.llm = Ollama(
    model="llama3.1:8b",
    request_timeout=300,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\workspace\llama_index\model\bge-small-zh-v1.5")

from llama_index.core import SimpleDirectoryReader,VectorStoreIndex

documents = SimpleDirectoryReader(
    input_files=["./data/IPCC_AR6_WGII_Chapter03.pdf"]
    ).load_data()
text_splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=20)


nodes = node_parser.get_nodes_from_documents(documents)
base_nodes = text_splitter.get_nodes_from_documents(documents)
# print(nodes)

sentence_index = VectorStoreIndex(nodes)
base_index = VectorStoreIndex(base_nodes)

from llama_index.core.postprocessor import MetadataReplacementPostProcessor

query_engine =sentence_index.as_query_engine(
    similarity_top_k=2,# 目标键默认为`window`，以匹配node_parser的默认设置
    node_postprocessors=[
        MetadataReplacementPostProcessor(
            target_metadata_key="window",
        )
    ]
)
window_response = query_engine.query("What are the concerns surrounding the AMOC?")
print(window_response)

# for source_node in window_response.source_nodes:
#     print(source_node.node.metadata["original_text"])
#     print("--------------------")
# window = window_response.source_nodes[0].node.metadata["window"]
# sentence = window_response.source_nodes[0].node.metadata["original_text"]

# print(f"Window: {window}")
# print("------------------------------------------------------")
# print(f"Original Sentence: {sentence}")

# query_engine = base_index.as_query_engine(similarity_top_k=2)

# vector_response = query_engine.query("What are the concerns surrounding the AMOC?")
# print(vector_response)