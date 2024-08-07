from llama_index.core.extractors import TitleExtractor,QuestionsAnsweredExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Ollama(model="llama3.1:8b",request_timeout=600.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\workspace\llama_index\model\bge-small-zh-v1.5")

text_splitter = TokenTextSplitter(
    separator=" ",chunk_size=512, chunk_overlap=128
)

title_extractor = TitleExtractor(nodes=5)
qa_extractor = QuestionsAnsweredExtractor(questions=3)

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_dir=r"D:\workspace\llama_index\llamaindex\data\paul_graham").load_data(show_progress=True)

pipeline = IngestionPipeline(
    transformations=[text_splitter,title_extractor,qa_extractor]
)

nodes = pipeline.run(
    documents=documents,
    in_place=True,
    show_progress=True
)

# output_file_path = r"D:\workspace\llama_index\llamaindex\data\paul_graham\output.txt"
# with open(output_file_path, "w", encoding="utf-8") as file:
#     file.write(str(nodes))

# print(f"Nodes have been saved to {output_file_path}")

from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model,
    transformations=[text_splitter,title_extractor,qa_extractor]
)

output_file_path = r"D:\workspace\llama_index\llamaindex\data\paul_graham\index.txt"
with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(str(index))

print(f"Index have been saved to {output_file_path}")
