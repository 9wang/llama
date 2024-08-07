# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core import SimpleDirectoryReader
# documents = SimpleDirectoryReader(
#     input_dir=r"D:\workspace\llama_index\llamaindex\data\paul_graham").load_data(show_progress=True)


# parser = SentenceSplitter()
# nodes = parser.get_nodes_from_documents(documents)
# print(nodes)

from llama_index.core.schema import TextNode,NodeRelationship,RelatedNodeInfo

node1 = TextNode(
    text="<text_chunk>",id_="<node_id>"
)

node2 = TextNode(
    text="<text_chunk>",id_="<node_id>"
)

node1.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
    node_id = node2.node_id
)

node2.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
    node_id = node1.node_id
)
node2.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
    node_id=node1.node_id,metadata={"key":"val"}
)

nodes = [node1,node2]

print(nodes)