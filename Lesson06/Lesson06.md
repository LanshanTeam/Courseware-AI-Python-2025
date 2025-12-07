# 大模型本地化部署：Ollama&vLLM&LMDeploy+ModelScope

## 第一部分：大模型**训练 (Training)** 和 **推理 (Inference)** 

------

## 1. 核心目的不同

### **训练：学习参数**

- 目标：让模型通过大量数据 **更新参数**，学习语言或任务。
- 过程：前向计算 + 损失函数 + 反向传播 + 参数更新。

### **推理：使用参数**

- 目标：利用已训练好的参数 **生成答案/推断结果**。
- 过程：只有前向计算，无梯度、无更新。

------

## 2. 计算开销差异巨大

| 项目     | **训练**                        | **推理**                  |
| -------- | ------------------------------- | ------------------------- |
| 计算量   | 最大（前向 + 反向 × 批次）      | 相对较小（仅前向）        |
| 激活存储 | 必须保存用于反向                | 大多数可丢弃              |
| 内存占用 | 参数 + 优化器状态 + 梯度 + 激活 | 参数 + KV Cache（生成时） |
| 吞吐目标 | 尽量大 batch、高吞吐            | 低延迟 + 并发优先         |
| 可用精度 | FP16/BF16                       | 量化 INT8 / 4bit 常见     |

例如：

- **70B 训练显存需求 > 400 GB**
- **70B 推理（4bit 量化后）可降至约 40GB**

------

## 3. 并行策略不同

### **训练使用更多并行度：**

- Data Parallel（数据并行）
- Tensor Parallel（张量并行）
- Pipeline Parallel（流水线并行）
- ZeRO Sharding（优化器/梯度分片）

### **推理关注低延迟/吞吐：**

- 动态批量（Dynamic Batching）
- KV Cache 优化（vLLM / FasterTransformer）
- 模型分片（Shard）部署
- 模型 offloading（CPU/NVMe 分层存储）

------

## 4. 模型参数的处理方式不同

### 训练阶段：

- 参数会持续更新
- 优化器状态（如 Adam 的 m、v）会额外占用 **2×参数大小**

### 推理阶段：

- 参数固定不变
- 不需要优化器状态 → 内存需求更低
- 可做各种压缩（量化/剪枝/蒸馏）

------

## 5. 关注指标不同

### **训练：**

- Loss、训练曲线
- 验证集指标（PPL、准确率）
- 收敛速度、泛化能力

### **推理：**

- 延迟（Latency）
- 吞吐（TPS）
- 性能成本（显存/功耗）
- 生成质量、稳定性

## 一、Ollama：轻量级本地化部署框架

Ollama 是一个开源的大型语言模型（LLM）平台，旨在让用户能够轻松地在本地运行、管理和与大型语言模型进行交互。

Ollama 提供了一个简单的方式来加载和使用各种预训练的语言模型，支持文本生成、翻译、代码编写、问答等多种自然语言处理任务。

Ollama 的特点在于它不仅仅提供了现成的模型和工具集，还提供了方便的界面和 API，使得从文本生成、对话系统到语义分析等任务都能快速实现。

与其他 NLP 框架不同，Ollama 旨在简化用户的工作流程，使得机器学习不再是只有深度技术背景的开发者才能触及的领域。

Ollama 支持多种硬件加速选项，包括纯 CPU 推理和各类底层计算架构（如 Apple Silicon），能够更好地利用不同类型的硬件资源。

**定位**：专为本地设备设计的开源框架，支持 macOS/Linux/Windows（需 WSL），无需云端资源即可运 行百亿级模型

###### 核心优势详解

1. 动态内存管理 分片加载机制：将大模型拆分为多个分片（Shards），仅在推理时按需加载到显存。例如 70B 模型原生需 140GB 显存，Ollama 通过分片可降至 40GB，适配消费级显卡（如 RTX  4090）。 智能卸载：闲置模型层自动转移至系统内存或磁盘，缓解显存压力。 
2. 量化压缩支持 原生支持 GGUF 格式的 4-bit/5-bit 量化（如 Q4_K_M），70B 模型体积从 140GB 压缩至  ~40GB，精度损失低于 2%。 支持多级量化策略：Q2_K（最小体积）→ Q6_K（最高精度），用户可依硬件性能选择。
3. 跨平台硬件加速 后端支持 CUDA（NVIDIA GPU）、Metal（Apple M 系列）、Vulkan（AMD/Intel GPU）及 纯 CPU 推理，同一模型无需修改即可跨设备运行14。 集成 OpenBLAS/cuBLAS 加速库，优化矩阵运算效率。
4. 隐私与易用性 数据完全本地处理，符合 GDPR 隐私规范，适合医疗、金融等敏感场景。 类 OpenAI API 设计，支持  LlamaIndex 等生态。 /v1/chat/completions 等端点，无缝对接 LangChain

### 详细部署流程

1. #### 安装与环境配置

```python
# Linux/macOS 一键安装
curl -fsSL https://ollama.com/install.sh | sh
# Windows 下载安装包：https://ollama.com/download
```

2. #### 模型加载与交互

```python
# 下载并运行模型（如 DeepSeek-R1）
ollama run deepseek-r1:1.5b
 # 命令行对话示例
>>> "解释量子纠缠现象"
>>> /bye  # 退出会
```

3. #### API 服务化部署

```Python
# 启动服务（默认端口 11434）
export OLLAMA_HOST="0.0.0.0:11434"  # 开放远程访问
ollama serve
 # 远程调用示例（JSON 格式）
curl http://192.168.1.100:11434/api/generate -d '{
"model": "deepseek-r1",
"prompt": "写一首关于春天的诗",
"stream": false }'

```

## 部署前模型压缩技术（量化 / 剪枝 / 蒸馏）

**为什么部署前要做压缩**？

- 原始大模型参数巨大（7B ~ 70B+）
- 推理主要受限于 **显存**、**带宽**、**延迟**
- 压缩技术可实现：**更快推理速度**,**更小显存占用 * **低部署成本******  更适合边缘设备/本地部署****

------

# 一、量化（Quantization）

## 作用

将 FP16/BF16 的权重量化为 8bit / 4bit / 2bit 表示，降低模型体积和显存占用。

------

## 量化分类

### **1. PTQ（Post-Training Quantization）训练后量化**

- 速度快、对原模型无侵入
- 常见工具：
  **GGUF / AWQ / GPTQ / INT8 / INT4（bitsandbytes）**
- 适合部署（Ollama、LMDeploy、vLLM 全支持）

### **2. QAT（Quantization-Aware Training）量化感知训练**

- 在训练中模拟量化误差
- 精度最佳，但成本高
- 用于芯片落地（Ascend、NPU 等）

## 工具示例

### **LMDeploy 4bit 量化：**

```
lmdeploy lite auto_awq internlm2_5-7b-chat --w-bits 4 --work-dir ./model_4bit
```

### **Ollama 使用 GGUF 模型：**

```
ollama run llama3.1:8b-instruct-q4_K_M
```

------

# 二、剪枝（Pruning）

## 作用

删除模型中不重要的权重或结构，减少参数量。

------

## 剪枝的类型

### **1. 非结构化剪枝（Unstructured）**

- 删除单个权重 → 稀疏矩阵
- 理论很强、实际加速有限（需要稀疏库）

### **2. 结构化剪枝（Structured）**

- 删除整个 neuron、head、channel
- **可真正加速推理**

### **3. LoRA 剪枝 + 合并**

- 用于微调后的“轻量模型”回收

------

## 剪枝效果

- 保持性能前提下减少 **20%–70% 参数**
- 加速 **20%–80%**（取决于硬件）

------

## 工具链示例

- **OpenPPL / Neural Magic / LLM-Pruner**
- **ModelScope** 提供了中文模型剪枝 pipeline

------

# 三、蒸馏（Distillation）

## 作用

用大模型（Teacher）指导训练小模型（Student），达到：

- **模型更小**
- **推理更快**
- **性能接近原大模型**

例如：

- LLaMA3 70B → LLaMA3-8B-Instruct
- DeepSeek-R1 → Distill-R1 → R1-Qwen-7B

------

## 蒸馏方式

### **1. Logit Distillation（输出蒸馏）**

- 小模型学习大模型的概率分布
- 应用最广

### **2. Feature Distillation（特征蒸馏）**

- 让小模型模仿大模型中间层特征
- 更高精度

### **3. Reinforcement Distillation（行为蒸馏）**

- DeepSeek R1 采用：让小模型模仿大模型的“推理链”

------

## 效果示例

| 模型        | 原大小 | 蒸馏后         | 性能          |
| ----------- | ------ | -------------- | ------------- |
| DeepSeek-R1 | 671B   | Distill-R1 14B | 性能保留 ~80% |
| Llama2-70B  | ≈140GB | Llama2-13B     | 性能保留 >70% |

------

# 第三部分：部署前模型压缩流程

# 部署前模型优化标准流程

```
           训练好的大模型
                    ↓
           （1）剪枝 Pruning
                    ↓
           （2）量化 Quantization
                    ↓
      （3）蒸馏 Distillation（可选）
                    ↓
       （4）推理引擎优化（vLLM/LMDeploy）
                    ↓
           （5）部署（Ollama/本地/NPU）
```

# 大模型部署前必须完成三件事

## ✔ 量化：减少模型体积（INT8/4bit）

## ✔ 剪枝：减少模型结构（删除冗余参数）

## ✔ 蒸馏：训练一个更小、更快的模型

> **部署的目标不是“功能最强”，而是“性能/成本最优”。**

## 二、vLLM：高性能分布式推理框架

定位：加州伯克利分校研发的推理引擎，通过 PagedAttention 算法优化 KV 缓存，吞吐量较 
HuggingFace 提升 24 倍，适合高并发生产环境。
核心技术解析 

1. **PagedAttention 机制**
   将注意力计算的键值对（KV Cache）分页存储，类似操作系统虚拟内存管理，减少内存碎
     片，显存利用率提升 3 倍以上。
     支持 动态批处理（Dynamic Batching），自动合并请求提升 GPU 利用率。

2. 多硬件与量化支持
   适配 CUDA 12.4+，支持 FP8/BF16 量化及张量并行（Tensor Parallelism），单卡可运行 7B 
     模型，多卡扩展至 200B+。
     兼容 HuggingFace 模型库，无需转换格式直接加载5。

  ## 一、动机：为什么需要 PagedAttention？

  在自回归推理中，模型为每个生成的 token 都要访问之前所有 token 的 Key/Value（KV cache）。传统实现把每个请求的 KV 按顺序连续存放或为每个请求预留一段连续内存。这会导致两个主要问题：

    1. **内存浪费和碎片化**：为了支持 worst-case（最长序列或并发高峰），系统往往会做大幅预分配或频繁做内存重排（compaction），导致显存/主存利用率低。[arXiv+1](https://arxiv.org/pdf/2309.06180?utm_source=chatgpt.com)
    2. **批次与并发受限**：当每个请求的 KV 必须是连续且固定位置时，很难高效合并短请求以形成大 batch，从而降低 GPU 利用率。[vLLM Blog+1](https://blog.vllm.ai/2023/06/20/vllm.html?utm_source=chatgpt.com)

  **PagedAttention 的设计目标**：把 KV cache 的内存管理从“连续大块”改为“按页（blocks/pages）管理”，就像操作系统的虚拟内存分页（paging）一样，减少预分配、降低碎片并支持跨请求复用/共享，从而显著提高吞吐和显存利用率。[arXiv+1](https://arxiv.org/pdf/2309.06180?utm_source=chatgpt.com)

------

  ## 二、核心思想（一句话）

  把 KV cache 划分为固定大小的 **页（blocks）**，按需把这些页装载到物理显存或主存，Attention 的计算以页为单位读取和计算，从而允许 KV 在物理内存中**非连续**存放并支持跨请求共享与动态分页调度。[arXiv+1](https://arxiv.org/pdf/2309.06180?utm_source=chatgpt.com)

------

  ## 三、算法要点（概念性分步）

    1. **把序列按“块”分割**
       把每个请求的 token 序列分成若干固定长度的 block（例如每 block 包含 B 个 token）。每个 block 产生一组 K/V vectors（shape: B × num_heads × head_dim）。这些 K/V block 就是“页”。[arXiv](https://arxiv.org/pdf/2309.06180?utm_source=chatgpt.com)
    2. **KV Page 存放与映射（page table）**
       每个 block 对应一个 page id；系统维护一个 page table，把逻辑 page 映射到物理内存位置（显存、主存、甚至 NVMe）。当需要时可以把 page 从主存搬到显存（或反之），类似虚拟内存换页。[arXiv+1](https://arxiv.org/pdf/2309.06180?utm_source=chatgpt.com)
    3. **Attention 的分块计算（按 page 访问）**
       对于生成 token i，要计算其对所有先前 token 的注意力：不再一次性访问一个连续 KV 矩阵，而是**逐 page**地读取对应的 K block、计算 q·K^T 得到 A_block，再把 A_block 与 V_block 相乘并累加到输出。这样可只读取当前需要的页。[arXiv](https://arxiv.org/pdf/2309.06180?utm_source=chatgpt.com)
    4. **按需加载 + 缓存共享**
       只有当某 page 在显存中不存在并且当前步骤需要它时，才拷贝到显存。不同请求之间如果共享前缀（prefix sharing），可以直接共享相同的 page 数据，避免重复存储。[arXiv+1](https://arxiv.org/pdf/2309.06180?utm_source=chatgpt.com)
    5. **配合调度与批处理**
       vLLM 的调度器会试图把多个请求的 prefill/生成阶段组织起来，合并相同 page 的访问并做动态批处理（continuous batching），以提高 GPU kernel 的吞吐率

### 详细部署流程

#### 1、环境依赖安装

```Python
# 创建虚拟环境
conda create -n vllm python=3.10
 conda activate vllm
 # 安装 PyTorch 与 vLLM（需 CUDA 12.4）
pip install torch==2.5.1 torchvision==0.20.1 --index-url 
https://download.pytorch.org/whl/cu124
 pip install vllm==0.8.5
```

#### 2、模型加载和离线推理

```python
from vllm import LLM, SamplingParams
 # 初始化模型（以 DeepSeek-R1-Distill-Qwen-7B 为例）
llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
trust_remote_code=True, max_model_len=4096)
 # 批量推理
prompts = ["量子计算的优势是什么？", "如何训练 GPT 模型？"]
 outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95, 
max_tokens=100))
```

#### 3、启动 OpenAI 兼容 API 服务

```python
# 单卡启动（DeepSeek-R1-Distill-Qwen-7B）
vllm serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000
 # 多卡张量并行（DeepSeek-R1-Distill-Qwen-32B，4 卡）
vllm serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --port 8000 -
tensor-parallel-size 
```

### 三、LMDeploy：生产级量化与国产硬件适配

**定位**：由 InternLM 团队推出的端到端推理框架，专注模型压缩与异构硬件部署，支持昇腾（Ascend） NPU，显存优化达 90%+

### 详细部署流程



```python
# 安装 LMDeploy（x86 环境）
pip install lmdeploy[all]==0.5.3
 # 昇腾环境需额外安装 DLInfer
# pip install dlinfer-ascend
```

```Python
# W4A16 量化（以 InternLM2-5-7B 为例）
lmdeploy lite auto_awq internlm2_5-7b-chat --w-bits 4 --work-dir 
./model_4bit
# 启动量化模型对话
lmdeploy chat ./model_4bit --model-format awq
```

```Python
# 启动 API 服务（含量化）
lmdeploy serve api_server ./model_4bit --server-port 23333 --quant-policy 4
 # 客户端调用（Python）
from openai import OpenAI
client = OpenAI(base_url="http://localhost:23333/v1", api_key="YOUR_KEY")
response = client.chat.completions.create(model="default", messages=
[{"role":"user", "content":"解释强化学习原理"}])
```

## 四、ModelScope：一站式中文模型平台

**定位**：阿里达摩院开源的模型即服务（MaaS）平台，集成 300+ 中文优化模型，覆盖 NLP/CV/多模态任务。

1. 模型生态 覆盖 InternVL2-26B（多模态）、Qwen、DeepSeek 等国产 SOTA 模型，支持免费下载与微 调。 提供行业数据集（如阿里电商数据），预训练模型免环境配置在线运行。 

2. 高效推理 API，举个例子：

```Python
from modelscope.pipelines import pipeline
# 大语言模型调用
text_gen = pipeline('text-generation', model='deepseek-ai/DeepSeek-R1')
 print(text_gen("人工智能的未来趋势"))
```

作业一：自己部署一个现有的模型到电脑上。再封装成接口去访问。

作业二：自己去开源模型，如魔塔社区那些找一些模型，去部署。再用你们想的方式做成一些可用的聊天工具。

作业三：去读读蒸馏量化剪枝的论文，写点笔记和感想什么的能让自己理解自己看了什么。

思考：我们以后的部署可以怎么做才能轻量化到更多的小型设备上？实现多端可用的模型。利用到实际的应用场景呢？

可以提交至CyanyuMu@Outlook.com
