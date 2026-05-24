# Milvus 向量数据库构建高性能 RAG 系统



<div style="border:1px solid rgb(186, 206, 253);background:rgb(240, 244, 255);padding:3px 10px;border-radius:3px">
    <span><b>前言</b></span><br>
    在 AI 应用快速发展的今天，向量数据库已成为构建智能检索系统的核心基础设施。Milvus 作为一款开源的高性能向量数据库，在 RAG（Retrieval-Augmented Generation）系统中发挥着关键作用。
</div>



## 1. Milvus 简介与核心特性

官网地址：https://milvus.io/zh

### 1.1 什么是 Milvus？

Milvus 是一个开源的向量数据库，专为 AI 应用设计，支持大规模向量数据的存储、索引和检索。它具备以下核心特性：

- **高性能**：毫秒级向量检索，支持百万到百亿级向量
- **易用性**：提供 Python SDK，API 简洁直观
- **可扩展性**：支持分布式部署，水平扩展
- **多索引支持**：HNSW、IVF、DiskANN 等多种索引算法
- **多模态友好**：支持文本、图像、音频等多种数据类型

### 1.2 为什么选择 Milvus？

在构建 RAG 系统时，我们需要一个能够：

1. **快速检索**：从大规模文档库中快速找到相关文档
2. **精确匹配**：通过向量相似度找到语义相关的文档
3. **易于集成**：与现有 AI 工具链无缝集成
4. **生产就绪**：支持高并发、高可用部署

Milvus 完美满足这些需求，已成为 RAG 系统的首选向量数据库

### 1.3 Milvus Lite vs Milvus Server

Milvus 提供了两种使用方式：

| 特性         | Milvus Lite                    | Milvus Server                           |
| :----------- | :----------------------------- | :-------------------------------------- |
| **部署方式** | 本地文件，无需启动服务         | 独立服务，需要启动                      |
| **适用场景** | 开发、测试、小规模应用         | 生产环境、大规模应用                    |
| **数据存储** | 本地文件（.db）                | 分布式存储                              |
| **性能**     | 适合中小规模（<100 万向量）    | 支持大规模（百万到百亿级）              |
| **使用方式** | `MilvusClient(uri="./xxx.db")` | `connections.connect(host="localhost")` |



## 2. 环境准备与安装

### 2.1 系统要求

- Python 3.8+
- 8GB+ RAM（推荐 16GB）
- 10GB+ 磁盘空间（用于存储向量和模型）

### 2.2 安装依赖

首先创建Conda环境（推荐）：

```bash
conda create -n milvus python=3.11
conda activate milvus
```

安装必要的依赖：

```bash
pip install pymilvus[milvus_lite]   # 包含Milvus Lite
pip install sentence-transformers   # Embedding模型
pip install transformers            # 重排模型
pip install torch                   # PyTorch（重排模型需要）
```

**重要提示**：必须安装 `pymilvus[milvus_lite]`（不是 `pymilvus`），这样才能使用 Milvus Lite 的本地文件功能。

### 2.3 验证安装

```python
from pymilvus import MilvusClient

# 测试Milvus Lite连接
client = MilvusClient("./milvus_demo.db")
print("✓ Milvus Lite 安装成功！")
```



## 3. 数据准备：从文档到向量

### 3.1 文档预处理

在实际应用中，我们需要将原始文档转换为向量。这个过程包括：

1. **文档加载**：从文件、数据库或 API 加载文档
2. **文档分块**：将长文档切分为较小的块（chunks）
3. **向量化**：使用 Embedding 模型将文本转换为向量

### 3.2 文档分块策略

文档分块是 RAG 系统的关键步骤，直接影响检索效果：

```python
def chunk_document(text, chunk_size=512, overlap=50):
    """
    文档分块

    Args:
        text: 原始文档文本
        chunk_size: 每块的大小（字符数或tokens）
        overlap: 块之间的重叠大小，保证上下文连贯性

    Returns:
        文档块列表
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 重叠，保证上下文连贯

    return chunks
```

**分块策略选择**：

- **固定大小分块**：简单快速，适合结构化文档
- **语义分块**：按句子或段落边界分块，保持语义完整性
- **滑动窗口**：使用重叠窗口，避免信息丢失

### 3.3 向量化：选择 Embedding 模型

选择合适的 Embedding 模型至关重要：

| 模型                  | 维度 | 语言支持 | 适用场景         | 性能  |
| :-------------------- | :--- | :------- | :--------------- | :---- |
| **all-MiniLM-L6-v2**  | 384  | 英文     | 快速原型、演示   | ⭐⭐⭐   |
| **BGE-large-en-v1.5** | 1024 | 英文     | 生产环境、高精度 | ⭐⭐⭐⭐⭐ |
| **BGE-large-zh-v1.5** | 1024 | 中文     | 中文检索         | ⭐⭐⭐⭐⭐ |
| **E5-mistral-7b**     | 4096 | 多语言   | 长文档检索       | ⭐⭐⭐⭐  |

**代码示例**：

```python
from sentence_transformers import SentenceTransformer

# 选择模型（根据需求）
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # 快速，80MB
# embedding_model = "BAAI/bge-large-en-v1.5"  # 高精度，1.3GB

# 加载模型
encoder = SentenceTransformer(embedding_model)

# 生成向量
texts = ["Milvus is a vector database...", "RAG combines retrieval and generation..."]
embeddings = encoder.encode(texts, normalize_embeddings=True)

print(f"向量维度: {embeddings.shape}")  # (2, 384) 或 (2, 1024)
```

### 3.4 创建 Milvus 集合

在插入数据之前，需要先创建集合（Collection）：

```python
from pymilvus import MilvusClient

# 连接Milvus（自动使用Milvus Lite）
client = MilvusClient(uri="./milvus_demo.db")

# 集合配置
collection_name = "documents"
dimension = 384# 根据Embedding模型选择：all-MiniLM-L6-v2=384, BGE-large=1024

# 检查集合是否存在
if client.has_collection(collection_name):
    print(f"集合 {collection_name} 已存在，删除旧集合...")
    client.drop_collection(collection_name)

# 创建集合
client.create_collection(
    collection_name=collection_name,
    dimension=dimension,      # 向量维度
    metric_type="L2",         # 距离度量：L2（欧氏距离）/ IP（内积）/余弦相似度（COSINE）
    auto_id=True,             # 自动生成ID
)

print(f"✓ 集合 {collection_name} 创建成功")
```

### 3.5 插入数据

将向量和元数据插入 Milvus：

```python
# 准备数据
documents = [
    {
        "text": "Milvus is an open-source vector database...",
        "doc_id": "doc_001",
        "title": "Introduction to Milvus"
    },
    # ... 更多文档
]

# 生成向量
texts = [doc["text"] for doc in documents]
embeddings = encoder.encode(texts, normalize_embeddings=True)

# 准备插入数据
data = []
for i, (emb, doc) in enumerate(zip(embeddings, documents)):
    data.append({
        "vector": emb.tolist(),  # 向量（必须）
        "text": doc["text"],      # 文本内容
        "doc_id": doc["doc_id"],  # 业务ID
        "title": doc["title"],    # 标题
    })

# 插入数据
client.insert(collection_name=collection_name, data=data)
print(f"✓ 成功插入 {len(data)} 个文档")
```

**完整的数据准备脚本**：

```python
"""
完整的数据准备示例
"""
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

def prepare_data():
    # 1. 连接Milvus
    client = MilvusClient(uri="./milvus_demo.db")
    collection_name = "documents"

    # 2. 准备文档
    documents = [
        "Milvus is an open-source vector database...",
        "RAG combines information retrieval with language models...",
        # ... 更多文档
    ]

    # 3. 初始化Embedding模型
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dimension = 384

    # 4. 创建集合
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        metric_type="L2",
        auto_id=True,
    )

    # 5. 生成向量并插入
    embeddings = encoder.encode(documents, normalize_embeddings=True)
    data = [
        {"vector": emb.tolist(), "text": doc}
        for emb, doc in zip(embeddings, documents)
    ]

    client.insert(collection_name=collection_name, data=data)
    print(f"✓ 数据准备完成，共 {len(data)} 个文档")

if __name__ == "__main__":
    prepare_data()
```



## 4. 核心实现：构建 RAG 系统

### 4.1 RAG 系统架构

一个完整的 RAG 系统包含以下组件：

```mermaid
flowchart LR
   
    1(用户查询) --> 2(向量检索（知识库）)
    2 --> 3(结果重排（可选）)
    3 --> 4(上下文构建)
    4 --> 5(LLM生成答案)
```

### 4.2 基础 RAG 实现

让我们从最简单的 RAG 系统开始：

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

class SimpleRAGSystem:
    """基础RAG系统"""

    def __init__(self, milvus_uri="./milvus_demo.db", collection_name="documents"):
        # 连接Milvus
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name

        # 初始化Embedding模型
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 10):
        """向量检索"""
        # 1. 编码查询
        query_vector = self.encoder.encode(query, normalize_embeddings=True)
        query_vector = query_vector.tolist()

        # 2. 在Milvus中搜索
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            search_params={"metric_type": "L2", "params": {}},
            output_fields=["text", "doc_id", "title"]
        )

        # 3. 格式化结果
        retrieved_docs = []
        for hit in results[0]:
            retrieved_docs.append({
                "id": hit.get("id", ""),
                "text": hit.get("entity", {}).get("text", ""),
                "score": hit.get("distance", 0.0),
                "doc_id": hit.get("entity", {}).get("doc_id", ""),
                "title": hit.get("entity", {}).get("title", ""),
            })

        return retrieved_docs

    def query(self, user_query: str, top_k: int = 5):
        """查询流程"""
        # 1. 检索
        retrieved_docs = self.retrieve(user_query, top_k=top_k)

        # 2. 构建上下文
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])

        # 3. 返回结果（这里只返回上下文，实际应用中会调用LLM）
        return {
            "query": user_query,
            "context": context,
            "documents": retrieved_docs
        }

# 使用示例
rag = SimpleRAGSystem()
result = rag.query("What is Milvus?")
print(result["context"])
```

### 4.3 增强版 RAG：添加重排

向量检索虽然快速，但可能不够精确。我们可以添加重排（Reranking）步骤来提升准确性：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class EnhancedRAGSystem(SimpleRAGSystem):
    """增强版RAG系统（带重排）"""

    def __init__(self, milvus_uri="./milvus_demo.db", collection_name="documents"):
        super().__init__(milvus_uri, collection_name)

        # 初始化重排模型（可选）
        self.reranker = None
        self.reranker_tokenizer = None
        try:
            print("正在加载重排模型...")
            reranker_model = "BAAI/bge-reranker-base"
            self.reranker = AutoModelForSequenceClassification.from_pretrained(reranker_model)
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
            self.reranker.eval()
            print("✓ 重排模型加载成功")
        except Exception as e:
            print(f"⚠️ 重排模型加载失败: {e}，将跳过重排步骤")

    def rerank(self, query: str, documents: List[str], top_k: int = 10):
        """重排：使用BGE-Reranker对检索结果进行精排"""
        if self.reranker isNoneor len(documents) == 0:
            return documents[:top_k]

        # 构建查询-文档对
        pairs = [[query, doc] for doc in documents]

        # Tokenize
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            # 计算相关性分数
            scores = self.reranker(**inputs).logits.squeeze(-1)

            # 按分数排序
            ranked_indices = scores.argsort(descending=True)

            # 返回Top-K
            reranked_docs = [documents[idx] for idx in ranked_indices[:top_k]]

        return reranked_docs

    def query(self, user_query: str, retrieve_top_k: int = 100, rerank_top_k: int = 10):
        """查询流程：检索 → 重排 → 构建上下文"""
        # 1. 向量检索（粗排）
        retrieved_docs = self.retrieve(user_query, top_k=retrieve_top_k)

        # 2. 重排（精排）
        doc_texts = [doc["text"] for doc in retrieved_docs]
        reranked_texts = self.rerank(user_query, doc_texts, top_k=rerank_top_k)

        # 3. 构建上下文
        context = "\n\n".join(reranked_texts)

        return {
            "query": user_query,
            "context": context,
            "retrieved_count": len(retrieved_docs),
            "reranked_count": len(reranked_texts)
        }
```

**重排的优势**：

- **提升准确性**：重排模型考虑查询和文档的交互，比单纯向量相似度更准确
- **灵活调整**：可以调整重排后的 Top-K，平衡准确性和成本
- **效果显著**：通常能提升 15-22%的准确率



## 5. 关键优化：突破 U 型陷阱

### 5.1 什么是 U 型陷阱？

研究发现，长上下文语言模型存在**位置偏差**（Position Bias）：

- **开头位置**（Primacy Bias）：准确率 ~75.8% ✅
- **中间位置**（Lost in the Middle）：准确率 ~53.8% ❌
- **结尾位置**（Recency Bias）：准确率 ~63.2% ✅

这意味着，如果关键信息放在中间位置，模型可能无法有效利用，导致性能下降。

### **5.2 位置优化策略**

通过**位置优化**，我们可以突破 U 型陷阱：

**核心策略**：

1. **最相关文档 → 开头**：利用 Primacy Bias
2. **次相关文档 → 中间**：低优先级
3. **用户问题 → 结尾**：利用 Recency Bias

### 5.3 实现位置优化

```python
class OptimizedRAGSystem(EnhancedRAGSystem):
    """优化版RAG系统（位置优化）"""

    def build_context(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        构建上下文（位置优化：突破U型陷阱的关键步骤）

        策略：
        - 最相关文档 → 开头（利用Primacy Bias）
        - 次相关文档 → 中间（低优先级）
        - 用户问题 → 结尾（利用Recency Bias）
        """
        if len(retrieved_docs) == 0:
            returnf"系统提示：请回答问题。\n\n用户问题：{query}"

        # 按相关性排序（最相关的在前）
        sorted_docs = sorted(
            retrieved_docs,
            key=lambda x: x.get("score", 0) if isinstance(x.get("score"), (int, float)) else0,
            reverse=True# 分数越高越好（如果是相似度分数）
        )

        # 构建上下文
        context_parts = []

        # 系统提示
        context_parts.append("系统提示：请基于以下文档回答问题。\n")

        # 最相关文档（开头位置 - 利用Primacy Bias）
        context_parts.append("# 最相关文档（开头位置）\n")
        top_docs = sorted_docs[:min(3, len(sorted_docs))]
        for i, doc in enumerate(top_docs, 1):
            text = doc.get("text", "")
            context_parts.append(f"文档 {i}：{text}\n")

        # 次相关文档（中间位置）
        if len(sorted_docs) > 3:
            context_parts.append("\n# 次相关文档（中间位置）\n")
            secondary_docs = sorted_docs[3:min(7, len(sorted_docs))]
            for i, doc in enumerate(secondary_docs, 4):
                text = doc.get("text", "")
                context_parts.append(f"文档 {i}：{text}\n")

        # 用户问题（结尾位置 - 利用Recency Bias）
        context_parts.append("\n# 用户问题（结尾位置）\n")
        context_parts.append(f"用户问题：{query}")

        return"\n".join(context_parts)

    def query(self, user_query: str, retrieve_top_k: int = 100, rerank_top_k: int = 10):
        """完整查询流程：检索 → 重排 → 位置优化 → 生成"""
        print(f"\n查询：{user_query}\n")

        # 1. 向量检索
        print(f"步骤 1: 向量检索（Top-{retrieve_top_k}）...")
        retrieved_docs = self.retrieve(user_query, top_k=retrieve_top_k)
        print(f"✓ 检索到 {len(retrieved_docs)} 个文档")

        if len(retrieved_docs) == 0:
            return {"query": user_query, "context": "", "answer": "未找到相关文档。"}

        # 2. 重排（可选）
        if self.reranker isnotNone:
            print(f"\n步骤 2: 重排（Top-{rerank_top_k}）...")
            doc_texts = [doc["text"] for doc in retrieved_docs]
            reranked_texts = self.rerank(user_query, doc_texts, top_k=rerank_top_k)

            # 更新文档列表（只保留重排后的文档）
            reranked_docs = [
                doc for doc in retrieved_docs
                if doc["text"] in reranked_texts
            ]
            # 按重排顺序重新排序
            text_to_doc = {doc["text"]: doc for doc in reranked_docs}
            reranked_docs = [text_to_doc[text] for text in reranked_texts]
            print(f"✓ 重排完成，返回 Top-{len(reranked_docs)} 个文档")
        else:
            print(f"\n步骤 2: 跳过重排（未配置Reranker）...")
            reranked_docs = retrieved_docs[:rerank_top_k]
            print(f"✓ 使用检索结果，返回 Top-{len(reranked_docs)} 个文档")

        # 3. 位置优化构建上下文
        print(f"\n步骤 3: 位置优化构建上下文...")
        context = self.build_context(user_query, reranked_docs)
        print(f"✓ 上下文构建完成（长度：{len(context)} 字符）")

        # 4. 生成答案（这里只返回上下文，实际应用中会调用LLM）
        print(f"\n步骤 4: 上下文已准备就绪\n")

        return {
            "query": user_query,
            "retrieved_docs": retrieved_docs[:5],  # 只返回前5个用于展示
            "reranked_docs": reranked_docs,
            "context": context,
            "answer": "【提示】未配置LLM，仅返回上下文。\n\n" + context
        }
```

### 5.4 效果对比

位置优化带来的效果提升：

| 指标           | 优化前（中间位置） | 优化后（开头位置） | 提升     |
| :------------- | :----------------- | :----------------- | :------- |
| **准确率**     | 53.8%              | 75.8%              | **+22%** |
| **信息利用率** | 低                 | 高                 | 显著提升 |

**关键结论**：位置优化是突破 U 型陷阱的核心步骤，生产环境必须包含！



## 6. 性能优化实战

### 6.1 索引选择

Milvus 支持多种索引类型，选择合适的索引对性能至关重要：

| 索引类型     | 特点           | 适用场景               | 推荐参数              |
| :----------- | :------------- | :--------------------- | :-------------------- |
| **HNSW**     | 高精度、低延迟 | 高精度要求、低延迟需求 | M=16, ef=64           |
| **IVF_FLAT** | 成本友好       | 大规模数据、成本敏感   | nlist=1024, nprobe=10 |
| **IVF_PQ**   | 压缩存储       | 超大规模数据           | nlist=1024, m=8       |
| **DiskANN**  | 大规模场景     | 百亿级向量             | 适合磁盘存储          |

**代码示例**（使用 HNSW 索引）：

```python
# 注意：MilvusClient方式创建集合时，索引参数在create_collection中设置
# 如果需要更精细的索引控制，可以使用pymilvus的Collection方式

from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType

# 连接Milvus Server（不是Lite）
connections.connect(alias="default", host="localhost", port="19530")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
]

schema = CollectionSchema(fields=fields, description="Documents collection")
collection = Collection(name="documents", schema=schema)

# 创建HNSW索引
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {
        "M": 16,              # 每个节点的连接数
        "efConstruction": 200# 构建时的搜索范围
    }
}

collection.create_index(field_name="vector", index_params=index_params)

# 搜索时设置ef参数
search_params = {"metric_type": "L2", "params": {"ef": 64}}
```

### 6.2 批量处理

对于大量查询，使用批量处理可以显著提升吞吐量：

```python
def batch_search(self, queries: List[str], batch_size: int = 32):
    """批量检索"""
    all_results = []

    # 批量编码
    query_vectors = self.encoder.encode(queries, normalize_embeddings=True)

    # 批量搜索
    for i in range(0, len(query_vectors), batch_size):
        batch_vectors = query_vectors[i:i+batch_size]
        batch_queries = queries[i:i+batch_size]

        results = self.client.search(
            collection_name=self.collection_name,
            data=batch_vectors.tolist(),
            limit=10,
            search_params={"metric_type": "L2", "params": {}},
            output_fields=["text"]
        )

        all_results.extend(results)

    return all_results
```

### 6.3 缓存策略

实现查询缓存可以大幅降低延迟和成本：

```python
from functools import lru_cache
import hashlib
import json

class CachedRAGSystem(OptimizedRAGSystem):
    """带缓存的RAG系统"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}  # 简单的内存缓存，生产环境建议使用Redis

    def _get_cache_key(self, query: str, top_k: int) -> str:
        """生成缓存键"""
        key_data = f"{query}_{top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def retrieve(self, query: str, top_k: int = 10, use_cache: bool = True):
        """带缓存的检索"""
        if use_cache:
            cache_key = self._get_cache_key(query, top_k)
            if cache_key in self.cache:
                print(f"✓ 使用缓存结果")
                return self.cache[cache_key]

        # 执行检索
        results = super().retrieve(query, top_k)

        # 存入缓存
        if use_cache:
            self.cache[cache_key] = results

        return results
```

### 6.4 异步处理

对于高并发场景，使用异步处理可以提升性能：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRAGSystem(OptimizedRAGSystem):
    """异步RAG系统"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=4)

    asyncdef async_retrieve(self, query: str, top_k: int = 10):
        """异步检索"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor,
            self.retrieve,
            query,
            top_k
        )
        return results

    asyncdef async_query(self, user_query: str, retrieve_top_k: int = 100, rerank_top_k: int = 10):
        """异步查询"""
        # 异步检索
        retrieved_docs = await self.async_retrieve(user_query, top_k=retrieve_top_k)

        # 其他步骤可以继续异步化...
        # 这里简化处理

        return {
            "query": user_query,
            "documents": retrieved_docs
        }

# 使用示例
asyncdef main():
    rag = AsyncRAGSystem()
    result = await rag.async_query("What is Milvus?")
    print(result)

# asyncio.run(main())
```



## 7. 生产环境最佳实践

### 7.1 错误处理与重试

生产环境必须包含完善的错误处理：

```python
import time
from typing import Optional

class ProductionRAGSystem(OptimizedRAGSystem):
    """生产环境RAG系统（带错误处理和重试）"""

    def retrieve_with_retry(
        self,
        query: str,
        top_k: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """带重试的检索"""
        for attempt in range(max_retries):
            try:
                return self.retrieve(query, top_k)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise# 最后一次重试失败，抛出异常

                print(f"检索失败（尝试 {attempt + 1}/{max_retries}）：{e}")
                time.sleep(retry_delay * (attempt + 1))  # 指数退避

        return []

    def query(self, user_query: str, **kwargs):
        """生产环境查询（带错误处理）"""
        try:
            return super().query(user_query, **kwargs)
        except Exception as e:
            return {
                "query": user_query,
                "error": str(e),
                "context": "",
                "answer": "抱歉，查询过程中出现错误，请稍后重试。"
            }
```

### 7.2 日志记录

添加详细的日志记录，便于监控和调试：

```python
import logging
from datetime import datetime

class LoggedRAGSystem(ProductionRAGSystem):
    """带日志的RAG系统"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def query(self, user_query: str, **kwargs):
        """带日志的查询"""
        start_time = datetime.now()
        self.logger.info(f"开始查询: {user_query}")

        try:
            result = super().query(user_query, **kwargs)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"查询完成: {user_query}, "
                f"耗时: {elapsed_time:.2f}秒, "
                f"检索到: {len(result.get('retrieved_docs', []))} 个文档"
            )

            return result
        except Exception as e:
            self.logger.error(f"查询失败: {user_query}, 错误: {e}")
            raise
```

### 7.3 监控指标

收集关键性能指标：

```python
from collections import defaultdict
import time

class MonitoredRAGSystem(LoggedRAGSystem):
    """带监控的RAG系统"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {
            "total_queries": 0,
            "total_retrieval_time": 0,
            "total_rerank_time": 0,
            "total_query_time": 0,
            "errors": 0,
        }

    def query(self, user_query: str, **kwargs):
        """带监控的查询"""
        self.metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # 监控检索时间
            retrieval_start = time.time()
            retrieved_docs = self.retrieve(user_query, top_k=kwargs.get("retrieve_top_k", 100))
            self.metrics["total_retrieval_time"] += time.time() - retrieval_start

            # 监控重排时间
            if self.reranker:
                rerank_start = time.time()
                # ... 重排逻辑
                self.metrics["total_rerank_time"] += time.time() - rerank_start

            result = super().query(user_query, **kwargs)
            self.metrics["total_query_time"] += time.time() - start_time

            return result
        except Exception as e:
            self.metrics["errors"] += 1
            raise

    def get_metrics(self):
        """获取监控指标"""
        total = self.metrics["total_queries"]
        if total == 0:
            return {}

        return {
            "total_queries": total,
            "avg_retrieval_time": self.metrics["total_retrieval_time"] / total,
            "avg_rerank_time": self.metrics["total_rerank_time"] / total,
            "avg_query_time": self.metrics["total_query_time"] / total,
            "error_rate": self.metrics["errors"] / total,
        }
```

### 7.4 配置管理

使用配置文件管理参数：

```yaml
# config.yaml
milvus:
uri:"./milvus_demo.db"
collection_name:"documents"

embedding:
model:"sentence-transformers/all-MiniLM-L6-v2"
dimension:384

reranker:
enabled:true
model:"BAAI/bge-reranker-base"

retrieval:
top_k:100
rerank_top_k:10

position_optimization:
enabled:true
top_relevant:3
secondary_relevant:5
```

```python
import yaml

class ConfigurableRAGSystem(OptimizedRAGSystem):
    """可配置的RAG系统"""

    def __init__(self, config_path: str = "config.yaml"):
        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 使用配置初始化
        super().__init__(
            milvus_uri=config["milvus"]["uri"],
            collection_name=config["milvus"]["collection_name"]
        )

        self.config = config
```



## 8. 常见问题与解决方案

### **8.1 集合不存在**

**问题**：`Collection 'documents' does not exist`

**解决方案**：

```python
# 检查集合是否存在
if not client.has_collection(collection_name):
    print(f"集合 {collection_name} 不存在，请先创建集合并插入数据")
    # 创建集合
    client.create_collection(...)
    # 插入数据
    client.insert(...)
```

### 8.2 向量维度不匹配

**问题**：`Dimension mismatch: expected 384, got 1024`

**解决方案**：

```python
# 确保Embedding模型维度与集合维度一致
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # 384维
# 或
embedding_model = "BAAI/bge-large-en-v1.5"  # 1024维

# 创建集合时使用正确的维度
dimension = 384 if "all-MiniLM-L6-v2" in embedding_model else 1024
client.create_collection(..., dimension=dimension)
```

### 8.3 内存不足

**问题**：加载大模型时内存不足

**解决方案**：

1. **使用更小的模型**：

   ```python
   # 使用小模型（80MB vs 1.3GB）
   encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   ```

2. **延迟加载**：

   ```python
   class LazyRAGSystem:
       def __init__(self):
           self._encoder = None
           self._reranker = None
   
       @property
       def encoder(self):
           if self._encoder is None:
               self._encoder = SentenceTransformer("...")
           return self._encoder
   ```

3. **使用 CPU 模式**（如果 GPU 内存不足）：

   ```python
   encoder = SentenceTransformer("...", device="cpu")
   ```

### 8.4 检索结果不准确

**问题**：检索到的文档与查询不相关

**解决方案**：

1. **使用更好的 Embedding 模型**：

   ```python
   # 从all-MiniLM-L6-v2升级到BGE-large
   encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
   ```

2. **添加重排**：

   ```python
   # 使用重排模型提升准确性
   reranker = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
   ```

3. **调整检索参数**：

   ```python
   # 增加检索数量，然后重排
   retrieved_docs = self.retrieve(query, top_k=200)  # 增加检索数量
   reranked_docs = self.rerank(query, retrieved_docs, top_k=10)  # 重排后取Top-10
   ```

### **8.5 性能优化**

**问题**：查询速度慢

**解决方案**：

1. **使用合适的索引**：

   ```python
   # HNSW索引适合低延迟场景
   index_params = {
       "index_type": "HNSW",
       "params": {"M": 16, "efConstruction": 200}
   }
   ```

2. **批量处理**：

   ```python
   # 批量查询比单个查询更高效
   results = batch_search(queries, batch_size=32)
   ```

3. **缓存结果**：

   ```python
   # 缓存常见查询的结果
   cached_results = cache.get(query)
   if cached_results:
       return cached_results
   ```



## 9. 总结

介绍了如何从零开始基于 Milvus 构建高性能 RAG 系统，涵盖了数据准备、向量检索、结果重排、位置优化等核心环节。

**关键要点**：

- ✅ **位置优化是关键**：通过将最相关文档放在开头，可以突破 U 型陷阱，提升 22%的准确率
- ✅ **重排提升准确性**：使用 BGE-Reranker 等重排模型可以显著提升检索效果
- ✅ **索引选择很重要**：根据数据规模和性能需求选择合适的索引类型（HNSW、IVF 等）
- ✅ **生产环境需完善**：错误处理、日志记录、性能监控是生产环境必备功能



## 附录：完整代码示例

### A. 完整的数据准备脚本

```python
"""
完整的数据准备脚本
"""
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

def prepare_data():
    # 1. 连接Milvus
    client = MilvusClient(uri="./milvus_demo.db")
    collection_name = "documents"

    # 2. 准备文档
    documents = [
        {
            "text": "Milvus is an open-source vector database designed for AI applications...",
            "doc_id": "doc_001",
            "title": "Introduction to Milvus"
        },
        # ... 更多文档
    ]

    # 3. 初始化Embedding模型
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dimension = 384

    # 4. 创建集合
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        metric_type="L2",
        auto_id=True,
    )

    # 5. 生成向量并插入
    texts = [doc["text"] for doc in documents]
    embeddings = encoder.encode(texts, normalize_embeddings=True)

    data = [
        {
            "vector": emb.tolist(),
            "text": doc["text"],
            "doc_id": doc["doc_id"],
            "title": doc["title"],
        }
        for emb, doc in zip(embeddings, documents)
    ]

    client.insert(collection_name=collection_name, data=data)
    print(f"✓ 数据准备完成，共 {len(data)} 个文档")

if __name__ == "__main__":
    prepare_data()
```

### B. 完整的 RAG 系统实现

```python
"""
完整的优化RAG系统实现
"""
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any
import torch

class OptimizedRAGSystem:
    """优化的RAG系统"""

    def __init__(
        self,
        milvus_uri: str = "./milvus_demo.db",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = None, # "BAAI/bge-reranker-base"
    ):
        # 连接Milvus
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name

        # 初始化Embedding模型
        self.encoder = SentenceTransformer(embedding_model)

        # 初始化重排模型（可选）
        self.reranker = None
        self.reranker_tokenizer = None
        if reranker_model:
            self.reranker = AutoModelForSequenceClassification.from_pretrained(reranker_model)
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
            self.reranker.eval()

    def retrieve(self, query: str, top_k: int = 100):
        """向量检索"""
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            search_params={"metric_type": "L2", "params": {}},
            output_fields=["text", "doc_id", "title"]
        )

        retrieved_docs = []
        for hit in results[0]:
            retrieved_docs.append({
                "id": hit.get("id", ""),
                "text": hit.get("entity", {}).get("text", ""),
                "score": hit.get("distance", 0.0),
                "doc_id": hit.get("entity", {}).get("doc_id", ""),
                "title": hit.get("entity", {}).get("title", ""),
            })

        return retrieved_docs

    def rerank(self, query: str, documents: List[str], top_k: int = 10):
        """重排"""
        if self.reranker is None or len(documents) == 0:
            return documents[:top_k]

        pairs = [[query, doc] for doc in documents]

        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            scores = self.reranker(**inputs).logits.squeeze(-1)
            ranked_indices = scores.argsort(descending=True)
            return [documents[idx] for idx in ranked_indices[:top_k]]

    def build_context(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """构建上下文（位置优化）"""
        sorted_docs = sorted(
            retrieved_docs,
            key=lambda x: x.get("score", 0) if isinstance(x.get("score"), (int, float)) else 0,
            reverse=True
        )

        context_parts = ["系统提示：请基于以下文档回答问题。\n"]

        # 最相关文档（开头）
        context_parts.append("# 最相关文档（开头位置）\n")
        for i, doc in enumerate(sorted_docs[:3], 1):
            context_parts.append(f"文档 {i}：{doc.get('text', '')}\n")

        # 次相关文档（中间）
        if len(sorted_docs) > 3:
            context_parts.append("\n# 次相关文档（中间位置）\n")
            for i, doc in enumerate(sorted_docs[3:7], 4):
                context_parts.append(f"文档 {i}：{doc.get('text', '')}\n")

        # 用户问题（结尾）
        context_parts.append("\n# 用户问题（结尾位置）\n")
        context_parts.append(f"用户问题：{query}")

        return"\n".join(context_parts)

    def query(self, user_query: str, retrieve_top_k: int = 100, rerank_top_k: int = 10):
        """完整查询流程"""
        # 1. 检索
        retrieved_docs = self.retrieve(user_query, top_k=retrieve_top_k)

        # 2. 重排
        if self.reranker:
            doc_texts = [doc["text"] for doc in retrieved_docs]
            reranked_texts = self.rerank(user_query, doc_texts, top_k=rerank_top_k)
            text_to_doc = {doc["text"]: doc for doc in retrieved_docs}
            reranked_docs = [text_to_doc[text] for text in reranked_texts]
        else:
            reranked_docs = retrieved_docs[:rerank_top_k]

        # 3. 构建上下文
        context = self.build_context(user_query, reranked_docs)

        return {
            "query": user_query,
            "context": context,
            "documents": reranked_docs
        }

# 使用示例
if __name__ == "__main__":
    rag = OptimizedRAGSystem()
    result = rag.query("What is Milvus?")
    print(result["context"])
```



## 课后打卡

1. 使用 LlamaIndex / LangChain 构建 RAG Pipeline
2. 使用 LlamaIndex / LangChain 集成 Milvus 向量数据库
3. 使用 LLM 完成最终答案的生成

