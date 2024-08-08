# from llama_index.core import PropertyGraphIndex
# from llama_index.core import SimpleDirectoryReader,VectorStoreIndex


# documents = SimpleDirectoryReader(input_files=["./data/大会.docx"]).load_data()

# ##创建
# index = PropertyGraphIndex.from_documents(documents)

# ##使用
# retriever = index.as_retriever(
#     include_text=True,
#     similarity_top_k=2,
# )

# nodes = retriever.retrieve("Test")
from llama_index.graph_stores.neo4j import Neo4jPGStore
username="neo4j"
password="12345678"
url="bolt://localhost:7687"
graph_store = Neo4jPGStore(
    username=username,
    password=password,
    url=url
)

import pandas as pd
from llama_index.core import Document
from typing import Literal
news=pd.read_csv(
      "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv")
documents = [Document(text=f"{row['title']}:{row['text']}") for i,row in news.iterrows()]
entities = Literal["PERSON", "LOCATION", "ORGANIZATION", "PRODUCT", "EVENT"]
relations = Literal[
    "SUPPLIER_OF",
    "COMPETITOR",
    "PARTNERSHIP",
    "ACQUISITION",
    "WORKS_AT",
    "SUBSIDIARY",
    "BOARD_MEMBER",
    "CEO",
    "PROVIDES",
    "HAS_EVENT",
    "IN_LOCATION",
]

# define which entities can have which relations
validation_schema={
    "Person": ["WORKS_AT","BOARD_MEMBER","CEO","HAS_EVENT"],
    "Organization":[
        "SUPPLIER_OF",
        "COMPETITOR",
        "PARTNERSHIP",
        "ACQUISITION",
        "WORKS_AT",
        "SUBSIDIARY",
        "BOARD_MEMBER",
        "CEO",
        "PROVIDES",
        "HAS_EVENT",
        "IN_LOCATION",
    ],
    "Product":["PROVIDES"],
    "Event":["HAS_EVENT","IN_LOCATION"],
    "Location":["HAPPENED_AT","IN_LOCATION"]
}

from llama_index.core import PropertyGraphIndex,Settings
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor 
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name=r"D:\workspace\llama_index\model\bge-large-zh")
Settings.llm = Ollama(model="llama3.1:8b",request_timeout=300.0)

kg_extractor = SchemaLLMPathExtractor(
    llm=Settings.llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,
)

NUMBER_OF_ARTTICLES = 20
index = PropertyGraphIndex.from_documents(
    documents[:NUMBER_OF_ARTTICLES],
    kg_extractors=[kg_extractor],
    llm = Settings.llm,
    embed_model = Settings.embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)


graph_store.structured_query("""
CREATE VECTOR INDEX entity IF NOT EXISTS
FOR (m:`__Entity__`)
ON m.embedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
""")

similarity_threshold = 0.9
word_edit_distance = 5
data = graph_store.structured_query("""
MATCH (n:`__Entity__`)
CALL {
  WITH e
  CALL db.index.vector.queryNodes('entity', 10, e.embedding)
  YIELD node, score
  WITH node, score
  WHERE score > toFLoat($cutoff)
      AND (toLower(node.name) CONTAINS toLower(e.name) OR toLower(e.name) CONTAINS toLower(node.name)
           OR apoc.text.distance(toLower(node.name), toLower(e.name)) < $distance)
      AND labels(e) = labels(node)
  WITH node, score
  ORDER BY node.name
  RETURN collect(node) AS nodes
}
WITH distinct nodes
WHERE size(nodes) > 1
WITH collect([n in nodes | n.name]) AS results
UNWIND range(0, size(results)-1, 1) as index
WITH results, index, results[index] as result
WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
        CASE WHEN index <> index2 AND
            size(apoc.coll.intersection(acc, results[index2])) > 0
            THEN apoc.coll.union(acc, results[index2])
            ELSE acc
        END
)) as combinedResult
WITH distinct(combinedResult) as combinedResult
// extra filtering
WITH collect(combinedResult) as allCombinedResults
UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1) 
    WHERE x <> combinedResultIndex
    AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
)
RETURN combinedResult 
""", param_map={'cutoff':similarity_threshold, 'distance':word_edit_distance})
for row in data:
    print(row)





# from neo4j import GraphDatabase

# #连接到数据库
# username="neo4j"
# password="12345678"
# url="bolt://localhost:7687"

# ##创建连接
# driver = GraphDatabase.driver(url,auth=(username,password))

# ##创建节点
# def create_preson(tx,name):
#     tx.run("CREATE (:Person {name: $name})", name=name)

# ##使用事务创建节点
# with driver.session() as session:
#     session.execute_write(create_preson,name="Alice")
#     session.execute_write(create_preson,name="Bob")



# def create_knows_relationship(tx,person1, person2):
#     tx.run(
#         "MATCH (a:Person {name: $person1})"
#         "MATCH (b:Person {name: $person2})"
#         "CREATE (a)-[:KNOWS]->(b)",person1=person1, person2=person2
#     )

# ##使用事务创建关系
# with driver.session() as session:
#     session.execute_write(create_knows_relationship,"Alice","Bob")

# def get_all_nodes(tx):
#     result = tx.run("MATCH (n) RETURN n")
#     return result.data()

# ##使用事务查询所有节点
# with driver.session() as session:
#     nodes = session.execute_read(get_all_nodes)
#     print(nodes)

# ##使用事务查询所有KNOWS关系
# def get_knows_relationships(tx):
#     result = tx.run("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN r")
#     return result.data()

# with driver.session() as session:
#     relationships=session.execute_read(get_knows_relationships)
#     print(relationships)


# ##关闭数据库连接
# driver.close()