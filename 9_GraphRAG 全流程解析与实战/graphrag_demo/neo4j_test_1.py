# 实现了完整的图数据构建流程：
# 1. 导入原始文档（可选）
# 2. 创建文本块节点并连接到文档
# 3. 识别并导入实体，连接到相应的文本块
# 4. 建立实体间的关系
# 5. 将实体组织到社区中
# 6. 为每个社区添加综合报告和发现

import pandas as pd
from neo4j import GraphDatabase
import time

# 用于高效批量导入数据
# 接收Cypher语句、DataFrame和批次大小作为参数
# 将大DataFrame分割成小批次处理，避免内存溢出
# 使用UNWIND语句在Neo4j中高效处理批量数据
def batched_import(statement, df, batch_size=1000):
    total = len(df)
    start_s = time.time()
    for start in range(0,total, batch_size):
        batch = df.iloc[start: min(start + batch_size,total)]
        result = driver.execute_query("UNWIND $rows AS value " + statement,
                                      rows=batch.to_dict('records'),
                                      database_=NEO4J_DATABASE)
        print(result.summary.counters)
    print(f'{total} rows in {time.time() - start_s} s.')
    return total

# neo4j 环境配置
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "88888888"
NEO4J_DATABASE = "neo4j"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# parquet 文件存储目录
GRAPHRAG_FOLDER = "./output"

documents = pd.read_parquet(GRAPHRAG_FOLDER + "/documents.parquet")
text_units = pd.read_parquet(GRAPHRAG_FOLDER + "/text_units.parquet")
entities = pd.read_parquet(GRAPHRAG_FOLDER + "/entities.parquet")
relationships = pd.read_parquet(GRAPHRAG_FOLDER + "/relationships.parquet")
communities = pd.read_parquet(GRAPHRAG_FOLDER + "/communities.parquet")
community_reports = pd.read_parquet(GRAPHRAG_FOLDER + "/community_reports.parquet")

# 导入 documents（可选）
statement = """
MERGE (d:__Document__ {id:value.id})
SET d += value {.title, .text}
"""
batched_import(statement, documents)

# 导入 text_units（文本块）
statement = """
MERGE (c:__Chunk__ {id:value.id})
SET c += value {.text, .n_tokens}
WITH c, value
UNWIND value.document_ids AS document
MATCH (d:__Document__ {id:document})
MERGE (c)-[:PART_OF]->(d)
"""
batched_import(statement, text_units)

# 导入 entities（实体）
entity_statement = """
MERGE (e:__Entity__ {id:value.id})
SET e += value {.human_readable_id, .description, name:replace(value.title,'"','')}
WITH e, value
CALL apoc.create.addLabels(e, case when coalesce(value.type,"") = "" then [] else [apoc.text.upperCamelCase(replace(value.type,'"',''))] end) yield node
UNWIND value.text_unit_ids AS text_unit
MATCH (c:__Chunk__ {id:text_unit})
MERGE (c)-[:HAS_ENTITY]->(e)
"""
batched_import(entity_statement, entities)

# 导入 relationships（实体间关系）
relationships_statement = """
    MATCH (source:__Entity__ {name:replace(value.source,'"','')})
    MATCH (target:__Entity__ {name:replace(value.target,'"','')})
    MERGE (source)-[rel:RELATED {id: value.id}]->(target)
    SET rel += value {.weight, .human_readable_id, .description, .text_unit_ids}
    RETURN count(*) as createdRels
"""
batched_import(relationships_statement, relationships)

# 导入 communities（社区）
# 1. 创建或合并__Community__节点（代表实体的聚类社区）
# 2. 通过遍历关系ID，将相关实体连接到社区
# 3. 为社区内的每个实体创建IN_COMMUNITY关系
communities_statement = """
MERGE (c:__Community__ {id:value.id})
SET c += value {.level, .title, .community}
/*
UNWIND value.text_unit_ids as text_unit_id
MATCH (t:__Chunk__ {id:text_unit_id})
MERGE (c)-[:HAS_CHUNK]->(t)
WITH distinct c, value
*/
WITH *
UNWIND value.relationship_ids as rel_id
MATCH (start:__Entity__)-[:RELATED {id:rel_id}]->(end:__Entity__)
MERGE (start)-[:IN_COMMUNITY]->(c)
MERGE (end)-[:IN_COMMUNITY]->(c)
RETURN count(distinct c) as createdCommunities
"""
batched_import(communities_statement, communities)

# 导入 community_reports（社区报告）
# 1. 匹配或创建社区节点
# 2. 设置社区详细属性，包括排名、解释、完整内容和摘要
# 3. 为报告中的每个发现(Finding)创建节点
# 4. 建立社区与发现之间的HAS_FINDING关系
community_reports_statement = """
MERGE (c:__Community__ {community:value.community})
SET c += value {.level, .title, .rank, .rank_explanation, .full_content, .summary}
WITH c, value
UNWIND range(0, size(value.findings)-1) AS finding_idx
WITH c, value, finding_idx, value.findings[finding_idx] as finding
MERGE (c)-[:HAS_FINDING]->(f:Finding {id:finding_idx})
SET f += finding
"""
batched_import(community_reports_statement, community_reports)
