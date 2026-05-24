# 1、NEO4J数据库相关
# NEO4J_URI=neo4j://127.0.0.1:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=88888888
# NEO4J_DATABASE=neo4j

# 2、安装必要的依赖包
# pip install neo4j

# 3、NEO4J数据库查询使用方法
# （1）Show a few __Entity__ nodes and their relationships (Entity Graph)
# MATCH path = (:__Entity__)-[:RELATED]->(:__Entity__)
# RETURN path LIMIT 200
# （2）Show the Chunks and the Document (Lexical Graph)
# MATCH (d:__Document__) WITH d LIMIT 1
# MATCH path = (d)<-[:PART_OF]-(c:__Chunk__)
# RETURN path LIMIT 100
# （3）Show a Community and it's Entities
# MATCH (c:__Community__) WITH c LIMIT 1
# MATCH path = (c)<-[:IN_COMMUNITY]-()-[:RELATED]-(:__Entity__)
# RETURN path LIMIT 100
# (4) 清除数据
# MATCH (n)
# CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 25000 ROWS;

# 4、节点间相关描述
# （1）节点包括:
# 原始文档(__Document__)
# 文本块(__Chunk__)
# 实体(__Entity__,又可分为不同类型)
# 社区(__Community__)
# 协变量(__Covariate__)

# （2）关系包括:
# RELATED(entity之间)
# PART_OF(chunk与document之间)
# HAS_ENTITY(chunk与entity之间)
# IN_COMMUNITY(entity与community之间)
# HAS_FINDING
# HAS_COVARIATE(chunk与covariate之间)


# 导入相关的包
import pandas as pd
from neo4j import GraphDatabase
import time


# 指定Parquet文件路径
# 首次更新
GRAPHRAG_FOLDER="./output"
# 增量更新
# GRAPHRAG_FOLDER="./update_output"


# 数据库连接相关参数配置
NEO4J_URI="neo4j://127.0.0.1:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="88888888"
NEO4J_DATABASE="neo4j"


# 实例化一个图数据库实例
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# 在图数据库中创建约束  初始化
statements = """
create constraint document_id if not exists for (d:__Document__) require d.id is unique;
create constraint chunk_id if not exists for (c:__Chunk__) require c.id is unique;
create constraint entity_id if not exists for (e:__Entity__) require e.id is unique;
create constraint entity_title if not exists for (e:__Entity__) require e.title is unique;
create constraint entity_id if not exists for (c:__Community__) require c.community is unique;
create constraint related_id if not exists for ()-[rel:RELATED]->() require rel.id is unique;
""".split(";")
for statement in statements:
    if len((statement or "").strip()) > 0:
        print(statement)
        driver.execute_query(statement)


# 使用批处理方式将数据帧导入Neo4j
# 参数：statement 是要执行的 Cypher 查询，df 是要导入的数据帧，batch_size 是每批要导入的行数
def batched_import(statement, df, batch_size=1000):
    # 计算数据帧df中的总行数，并将其存储在total变量中
    total = len(df)
    # 记录当前时间，以便后续计算导入操作所花费的总时间
    start_s = time.time()
    # 每次循环处理一批数据，步数为batch_size
    for start in range(0,total, batch_size):
        # 使用Pandas的iloc方法提取当前批次的数据子集
        # start是当前批次的起始行号
        # min(start + batch_size, total)是当前批次的结束行号，确保不会超过总行数
        batch = df.iloc[start: min(start+batch_size,total)]
        # "UNWIND $rows AS value "是Cypher中的一个操作，它将 $row中的每个元素逐个解包，并作为value传递给Cypher语句statement
        result = driver.execute_query("UNWIND $rows AS value " + statement,
                                      # 将当前批次的 DataFrame 转换为字典的列表
                                      # 每一行数据变成一个字典，columns 作为键
                                      rows=batch.to_dict('records'),
                                      database_=NEO4J_DATABASE)
        # 打印执行结果的摘要统计信息，包括创建的节点、关系等计数
        print(result.summary.counters)
    # 计算并打印导入总行数和耗时
    print(f'{total} rows in { time.time() - start_s} s.')
    # 返回导入的总行数
    return total


# 按顺序依次执行如下步骤

# 1、创建或更新documents
# 从指定的 Parquet 文件 documents.parquet 中读取 id、title 和 text 这三列
# 并将它们加载到一个名为 doc_df 的 Pandas 数据帧中。这个数据帧可以进一步用于数据处理、分析或导入操作
doc_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/documents.parquet', columns=["id", "title", "text"])
# # 打印输出数据帧 doc_df 的前 30 行内容
# print(doc_df.head(30))
# MERGE (d:__Document__ {id:value.id})尝试在数据库中找到一个具有 id 属性值为 value.id 的 __Document__ 节点
# 如果找到，则匹配这个节点；如果找不到，则创建一个新的 __Document__ 节点，并将 id 属性设置为 value.id
# SET d += value {.title, .text}将外部 value 对象的 title 属性值赋给节点 d 的 title 属性
# 如果 d 节点已经存在 title 属性，它将被更新为新值；如果 d 节点没有 title 属性，则会新建一个
statement = """
MERGE (d:__Document__ {id:value.id})
SET d += value {.title, .text}
"""
total = batched_import(statement, doc_df)
print("返回的结果：",total)


# 2、创建或更新chunks与documents之间的关系
# 从指定的 Parquet 文件 text_units.parquet 中读取列
# 并将它们加载到一个名为 text_df 的 Pandas 数据帧中。这个数据帧可以进一步用于数据处理、分析或导入操作
text_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/text_units.parquet',
                          columns=["id","text","n_tokens","document_ids","entity_ids","relationship_ids","covariate_ids"])
# 打印输出数据帧 text_df 的前 30 行内容
print(text_df.head(30))
# MERGE (c:__Chunk__ {id:value.id})尝试匹配或创建一个节点。如果具有指定属性的节点存在，则返回该节点；否则，创建一个新的节点
# SET c += value {.text, .n_tokens}从 value 对象中提取属性，并将它们赋值给节点 c 的同名属性
# WITH c, value用于将当前查询上下文中的变量传递给接下来的查询部分。在这里，c 和 value 被传递到下一步的查询中
# UNWIND value.document_ids AS document将列表 value.document_ids 中的每个元素依次展开为单独的记录,并将每个元素命名为 document，进行单独处理
# MATCH (d:__Document__ {id:document})查找 __Document__ 标签的节点，并且 id 属性值等于 document
# MERGE (c)-[:PART_OF]->(d)在__Chunk__节点与 __Document__ 节点之间创建一个 PART_OF 类型的关系。如果关系已经存在，则不创建新的关系，表示 c 是 d 的一部分
statement = """
MERGE (c:__Chunk__ {id:value.id})
SET c += value {.text, .n_tokens}
WITH c, value
UNWIND value.document_ids AS document
MATCH (d:__Document__ {id:document})
MERGE (c)-[:PART_OF]->(d)
"""
batched_import(statement, text_df)


# 3、创建或更新entities与chunks之间的关系
# 从指定的 Parquet 文件 entities.parquet 中读取列
# 并将它们加载到一个名为 entity_df 的 Pandas 数据帧中。这个数据帧可以进一步用于数据处理、分析或导入操作
entity_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/entities.parquet',
                            columns=["title","type","description","human_readable_id","id","text_unit_ids"])
# 打印输出数据帧 entity_df 的前 30 行内容
print(entity_df.head(30))
# MERGE (e:__Entity__ {id:value.id})尝试匹配或创建一个节点。如果具有指定属性的节点存在，则返回该节点；否则，创建一个新的节点
# SET e += value {name:replace(value.title,'"',''), .type, .description, .human_readable_id, .id, .description_embedding, .text_unit_ids}从 value 对象中提取属性，并将它们赋值给节点 e 的同名属性
# WITH e, value用于将当前查询上下文中的变量传递给接下来的查询部分。在这里，e 和 value 被传递到下一步的查询中
# CALL db.create.setNodeVectorProperty(e, "description_embedding", value.description_embedding)调用 Neo4j 中的自定义过程，设置 e 节点的 description_embedding 属性,将 value.description_embedding 的值作为向量属性存储在 e 节点的 description_embedding 属性中
# CALL apoc.create.addLabels()使用 APOC 库中的方法为节点 e 添加标签,根据 value.type 的值决定要添加的标签即将entity的类型均设置为标签
# UNWIND value.text_unit_ids AS text_unit将列表 value.text_unit_ids 中的每个元素依次展开为单独的记录,并将每个元素命名为 text_unit，进行单独处理
# MATCH (c:__Chunk__ {id:text_unit})查找 __Chunk__ 标签的节点，并且 id 属性值等于 text_unit
# MERGE (c)-[:HAS_ENTITY]->(e)在__Chunk__节点与 __Entity__ 节点之间创建一个 HAS_ENTITY 类型的关系。如果关系已经存在，则不创建新的关系，表示表示该文本块包含该实体
entity_statement = """
MERGE (e:__Entity__ {id:value.id})
SET e += value {name:replace(value.title,'"',''), .type, .description, .human_readable_id, .id, .text_unit_ids}
WITH e, value
CALL apoc.create.addLabels(e, case when coalesce(value.type,"") = "" then [] else [apoc.text.upperCamelCase(replace(value.type,'"',''))] end) yield node
UNWIND value.text_unit_ids AS text_unit
MATCH (c:__Chunk__ {id:text_unit})
MERGE (c)-[:HAS_ENTITY]->(e)
"""
batched_import(entity_statement, entity_df)


# 4、创建或更新entity节点之间的关系
# 从指定的 Parquet 文件 relationships.parquet 中读取列
# 并将它们加载到一个名为 rel_df 的 Pandas 数据帧中。这个数据帧可以进一步用于数据处理、分析或导入操作
rel_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/relationships.parquet',
                         columns=["source","target","id","combined_degree","weight","human_readable_id","description","text_unit_ids"])
# 打印输出数据帧 rel_df 的前 30 行内容
print(rel_df.head(30))
# MATCH (source:__Entity__ {name:replace(value.source,'"','')})查找 __Entity__ 标签的节点，并且 name 属性等于 value.source 中的内容（其中所有双引号 " 被替换为空字符串）。找到的节点赋值给变量 source
# MATCH (target:__Entity__ {name:replace(value.target,'"','')})查找 __Entity__ 标签的节点，并且 name 属性等于 value.target 中的内容（其中所有双引号 " 被替换为空字符串）。找到的节点赋值给变量 target
# MERGE (source)-[rel:RELATED {id: value.id}]->(target)在 source 和 target 之间查找或创建一个 RELATED 类型的关系，并为该关系设置 id 属性。如果已经存在具有相同 id 的关系，则更新其属性
# SET rel += value {.weight, .human_readable_id, .description, .text_unit_ids}将 value 对象中的 weight、human_readable_id、description 和 text_unit_ids 属性合并到该关系上
# RETURN count(*) as createdRels返回创建或更新的关系数量，并将结果命名为 createdRels
rel_statement = """
    MATCH (source:__Entity__ {name:replace(value.source,'"','')})
    MATCH (target:__Entity__ {name:replace(value.target,'"','')})
    MERGE (source)-[rel:RELATED {id: value.id}]->(target)
    SET rel += value {.weight, .human_readable_id, .description, .text_unit_ids}
    RETURN count(*) as createdRels
"""
batched_import(rel_statement, rel_df)


# 5、创建或更新community与entity、chunk节点之间的关系
# 从指定的 Parquet 文件 community_reports.parquet 中读取列
# 并将它们加载到一个名为 community_report_df 的 Pandas 数据帧中。这个数据帧可以进一步用于数据处理、分析或导入操作
community_report_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/community_reports.parquet',
                               columns=["id","community","findings","title","summary", "level","rank","rating_explanation","full_content"])
# 打印输出数据帧 rel_df 的前 30 行内容
print(community_report_df.head(30))
# MERGE (c:__Community__ {id:value.id})查找或创建一个 __Community__ 节点，并设置其 id 属性为 value.id 的值。如果具有相同 id 的节点已存在，则返回该节点；否则，创建新的节点
# SET c += value {.community, .level,}将 value 对象中指定的属性和值合并到节点 c 上，不覆盖现有属性。从 value 对象中提取属性，并赋值给 c 节点的同名属性
# WITH c, value将当前查询上下文中的 c 和 value 变量传递给接下来的查询部分
# UNWIND：将列表展开为多个行，生成一个从 0 到 value.findings 列表大小减一的范围列表，表示所有 finding 项的索引，将每个索引值赋值给 finding_idx 变量
# WITH c, value, finding_idx, value.findings[finding_idx] as finding将当前查询上下文中的变量传递到下一个查询部分
# MERGE (c)-[:HAS_FINDING]->(f:Finding {id:finding_idx})查找或创建 __Community__ 节点 c 与 Finding 节点 f 之间的 HAS_FINDING 关系。Finding 节点的 id 属性设置为 finding_idx
# SET f += finding将 finding 对象中的属性和值合并到 f 节点上
community_statement = """
MERGE (c:__Community__ {id:value.id})
SET c += value {.community, .level, .title, .rank, .rating_explanation, .full_content, .summary}
WITH c, value
UNWIND range(0, size(value.findings)-1) AS finding_idx
WITH c, value, finding_idx, value.findings[finding_idx] as finding
MERGE (c)-[:HAS_FINDING]->(f:Finding {id:finding_idx})
SET f += finding
"""
batched_import(community_statement, community_report_df)


# 6、创建或更新community与entity、chunk节点之间的关系
# 从指定的 Parquet 文件 communities.parquet 中读取列
# 并将它们加载到一个名为 community_df 的 Pandas 数据帧中。这个数据帧可以进一步用于数据处理、分析或导入操作
community_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/communities.parquet',
                     columns=["id","level","title","text_unit_ids","relationship_ids"])
# 打印输出数据帧 rel_df 的前 30 行内容
print(community_df.head(30))
# MERGE (c:__Community__ {community:value.id})查找或创建一个 __Community__ 节点，并将其 community 属性设置为 value.id 的值。如果具有该属性的节点已存在，则返回该节点；否则，创建新的节点
# SET c += value {.level, .title}将 value 对象中的 level 和 title 属性赋值给 c 节点的同名属性
# UNWIND value.text_unit_ids as text_unit_id将 value.text_unit_ids 列表中的每个元素展开为单独的记录，并命名为 text_unit_id
# MATCH (t:__Chunk__ {id:text_unit_id})：查找具有对应 id 属性的 __Chunk__ 节点
# MERGE (c)-[:HAS_CHUNK]->(t)：在 __Community__ 节点 c 和 __Chunk__ 节点 t 之间创建或查找 HAS_CHUNK 关系
# WITH *：将当前查询上下文中的所有变量传递给接下来的查询部分。这里会保留 c 和 value 的上下文
# UNWIND value.relationship_ids as rel_id将 value.relationship_ids 列表中的每个元素展开为单独的记录，并命名为 rel_id
# MATCH (start:__Entity__)-[:RELATED {id:rel_id}]->(end:__Entity__)查找两个 __Entity__ 节点之间的 RELATED 关系，其中 id 属性等于 rel_id
# MERGE (start)-[:IN_COMMUNITY]->(c)和MERGE (end)-[:IN_COMMUNITY]->(c)
# MERGE在 start 节点和 c 节点之间创建或查找 IN_COMMUNITY 关系，以及在 end 节点和 c 节点之间创建或查找 IN_COMMUNITY 关系
# 这样将两个 __Entity__ 节点与 __Community__ 节点关联起来
# RETURN count(distinct c) as createdCommunities返回创建或更新的 __Community__ 节点的数量，并将结果命名为 createdCommunities
statement = """
MERGE (c:__Community__ {community:value.id})
SET c += value {.level}
WITH *
UNWIND value.text_unit_ids as text_unit_id
MATCH (t:__Chunk__ {id:text_unit_id})
MERGE (c)-[:HAS_CHUNK]->(t)
WITH *
UNWIND value.relationship_ids as rel_id
MATCH (start:__Entity__)-[:RELATED {id:rel_id}]->(end:__Entity__)
MERGE (start)-[:IN_COMMUNITY]->(c)
MERGE (end)-[:IN_COMMUNITY]->(c)
RETURN count(distinct c) as createdCommunities
"""
batched_import(statement, community_df)

