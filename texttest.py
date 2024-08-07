# from llama_index.core import SimpleDirectoryReader
# from llama_index.core import Document
# from llama_index.core.schema import MetadataMode

# text1 = "This is a test document.and this is a topic"
# text2 = "This is another test document.and this is another topic"
# text_list = [text1,text2]
# documents = [Document(text=t) for t in text_list]

# document = Document(
#     text=text1,
#     metadata={"filename":"<doc_file_name>","catefory":"<category>"},
# )

# print("docme\n",document.metadata)
# document.excluded_llm_metadata_keys = ["filename"]
# print("----------------------\n",document.metadata)
# print("----------------------\n",document.get_content(metadata_mode=MetadataMode.LLM))
# print("docme\n",document.metadata)
# document.metadata = {"filename":"<doc_file_name>"}


# print("doc\n",document)
# print("docme\n",document.metadata)

# # from llama_index.core import SimpleDirectoryReader

# # documents = SimpleDirectoryReader('./data',filename_as_id=True).load_data()
# # print("documents:\n",documents)
# # print("doc_id:\n",[x.doc_id for x in documents])


from llama_index.core import Document
from llama_index.core.schema import MetadataMode

document = Document(
    text="这是一个超级定制的文档",
    metadata={
        "file_name": "super_secret_document.txt",
        "category": "finance",
        "author": "LlamaIndex",
    },
    excluded_llm_metadata_keys=["file_name"],
    metadata_seperator="::",
    metadata_template="{key}=>{value}",
    text_template="元数据: {metadata_str}\n-----\n内容: {content}",
)

print(
    "LLM看到的内容：\n",
    document.get_content(metadata_mode=MetadataMode.LLM),
)

print(
    "嵌入模型看到的内容: \n",
    document.get_content(metadata_mode=MetadataMode.EMBED),
)