from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv(verbose=True)

# 1. 初始化数据库
NEO4J_URI="neo4j://127.0.0.1:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="88888888"
NEO4J_DATABASE="neo4j"

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# 2. 定义自定义 Cypher 生成提示词
CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
If the question is about relationships, ensure the variable for relationship is used correctly.
Note: type() function only works on relationships, not nodes.

Schema:
{schema}

Question: {question}
Cypher Query:"""

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE
)

# 3. 初始化 LLM
llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0
)

# 4. 构造链
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=CYPHER_PROMPT, # 使用自定义提示词纠正语法
    return_intermediate_steps=True, # 开启此项以获取生成的 Cypher 语句
    return_direct=False # 设置为 False 以便让 LLM 将结果组织成自然语言
)

# 5. 执行并输出结果
query = "谁是唐僧，他的主要关系是什么？"
response = chain.invoke({"query": query})

print("\n--- 最终答案 ---")
print(response["result"])

print("\n--- 生成的 Cypher 查询语句 ---")
# 这里的 intermediate_steps[0] 通常是生成的 Cypher
print(response["intermediate_steps"][0]["query"])

