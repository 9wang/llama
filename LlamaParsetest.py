import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-TislyIuXkq5YzqwTcebIsvt0ub6nMBtoT7xatXMEaLJ5QPEE",
    result_type="markdown",##可选markdown或者text
    verbose=True,
)

##同步
documents = parser.load_data("./data/llama2.pdf")
print(documents[:100])
# documents_str = "\n".join([documents.text for doc in documents if hasattr(doc, "text")])

# output_file = "./data/llama.md"
# with open(output_file,"w",encoding="utf-8") as f:
#     f.write(documents_str)

# print(f"Markdown saved to {output_file}")
#同步批处理
# documents = parser.load_data(["./data/llama2.pdf","./data/llama1.pdf"])

##异步
# documents = await parser.load_data_async("./data/llama2.pdf")

##异步批处理
# documents = await parser.load_data_async(["./data/llama2.pdf","./data/llama1.pdf"])

# print(documents)