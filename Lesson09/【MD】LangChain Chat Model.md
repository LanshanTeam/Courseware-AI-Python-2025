# Prompt templates: Few shot、Example selector
## <font style="color:rgb(28, 30, 33);">Few shot(少量示例)</font>
### <font style="color:rgb(28, 30, 33);">创建少量示例的格式化程序</font>
<font style="color:rgb(28, 30, 33);">创建一个简单的提示模板，用于在生成时向模型提供示例输入和输出。向LLM提供少量这样的示例被称为少量示例，这是一种简单但强大的指导生成的方式，在某些情况下可以显著提高模型性能。</font>

<font style="color:rgb(28, 30, 33);">少量示例提示模板可以由一组示例或一个负责从定义的集合中选择一部分示例的</font>[<font style="color:rgb(28, 30, 33);">示例选择器</font>](https://api.python.langchain.com/en/latest/example_selectors/langchain_core.example_selectors.base.BaseExampleSelector.html)<font style="color:rgb(28, 30, 33);">类构建。</font>

<font style="color:rgb(28, 30, 33);">配置一个格式化程序，将少量示例格式化为字符串。这个格式化程序应该是一个</font>`<font style="color:rgb(28, 30, 33);">PromptTemplate</font>`<font style="color:rgb(28, 30, 33);">对象。</font>

```python
from langchain_core.prompts import PromptTemplate
example_prompt = PromptTemplate.from_template("问题：{question}\n{answer}")
```

### <font style="color:rgb(28, 30, 33);">创建示例集合</font>
<font style="color:rgb(28, 30, 33);">接下来，我们将创建一个少量示例的列表。每个示例应该是一个字典，表示我们上面定义的格式化提示的示例输入。</font>

```python
examples = [
    {
        "question": "谁活得更长，穆罕默德·阿里还是艾伦·图灵？",
        "answer": """
是否需要后续问题：是的。
后续问题：穆罕默德·阿里去世时多大年纪？
中间答案：穆罕默德·阿里去世时74岁。
后续问题：艾伦·图灵去世时多大年纪？
中间答案：艾伦·图灵去世时41岁。
所以最终答案是：穆罕默德·阿里
""",
    },
    {
        "question": "克雷格斯列表的创始人是什么时候出生的？",
        "answer": """
是否需要后续问题：是的。
后续问题：克雷格斯列表的创始人是谁？
中间答案：克雷格斯列表的创始人是克雷格·纽马克。
后续问题：克雷格·纽马克是什么时候出生的？
中间答案：克雷格·纽马克于1952年12月6日出生。
所以最终答案是：1952年12月6日
""",
    },
    {
        "question": "乔治·华盛顿的外祖父是谁？",
        "answer": """
是否需要后续问题：是的。
后续问题：乔治·华盛顿的母亲是谁？
中间答案：乔治·华盛顿的母亲是玛丽·波尔·华盛顿。
后续问题：玛丽·波尔·华盛顿的父亲是谁？
中间答案：玛丽·波尔·华盛顿的父亲是约瑟夫·波尔。
所以最终答案是：约瑟夫·波尔
""",
    },
    {
        "question": "《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？",
        "answer": """
是否需要后续问题：是的。
后续问题：《大白鲨》的导演是谁？
中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
后续问题：史蒂文·斯皮尔伯格来自哪个国家？
中间答案：美国。
后续问题：《皇家赌场》的导演是谁？
中间答案：《皇家赌场》的导演是马丁·坎贝尔。
后续问题：马丁·坎贝尔来自哪个国家？
中间答案：新西兰。
所以最终答案是：不是
""",
    },
]
```

<font style="color:rgb(28, 30, 33);">让我们使用其中一个示例测试格式化提示：</font>

```python
print(example_prompt.invoke(examples[0]).to_string())
```

```plain
问题：谁活得更长，穆罕默德·阿里还是艾伦·图灵？

是否需要后续问题：是的。
后续问题：穆罕默德·阿里去世时多大年纪？
中间答案：穆罕默德·阿里去世时74岁。
后续问题：艾伦·图灵去世时多大年纪？
中间答案：艾伦·图灵去世时41岁。
所以最终答案是：穆罕默德·阿里
```

### <font style="color:rgb(28, 30, 33);">将示例和格式化程序传递给</font>`<font style="color:rgb(28, 30, 33);">FewShotPromptTemplate</font>`
<font style="color:rgb(28, 30, 33);">最后，创建一个</font>[<font style="color:rgb(28, 30, 33);">FewShotPromptTemplate</font>](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.few_shot.FewShotPromptTemplate.html)<font style="color:rgb(28, 30, 33);">对象。该对象接受少量示例和少量示例的格式化程序。当格式化此</font>`<font style="color:rgb(28, 30, 33);">FewShotPromptTemplate</font>`<font style="color:rgb(28, 30, 33);">时，它使用</font>`<font style="color:rgb(28, 30, 33);">example_prompt</font>`<font style="color:rgb(28, 30, 33);">格式化传递的示例，然后将它们添加到</font>`<font style="color:rgb(28, 30, 33);">suffix</font>`<font style="color:rgb(28, 30, 33);">之前的最终提示中：</font>

```python
from langchain_core.prompts import FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="问题：{input}",
    input_variables=["input"],
)
print(
    prompt.invoke({"input": "乔治·华盛顿的父亲是谁？"}).to_string()
)
```

```plain
问题：谁活得更长，穆罕默德·阿里还是艾伦·图灵？

是否需要后续问题：是的。
后续问题：穆罕默德·阿里去世时多大年纪？
中间答案：穆罕默德·阿里去世时74岁。
后续问题：艾伦·图灵去世时多大年纪？
中间答案：艾伦·图灵去世时41岁。
所以最终答案是：穆罕默德·阿里


问题：克雷格斯列表的创始人是什么时候出生的？

是否需要后续问题：是的。
后续问题：克雷格斯列表的创始人是谁？
中间答案：克雷格斯列表的创始人是克雷格·纽马克。
后续问题：克雷格·纽马克是什么时候出生的？
中间答案：克雷格·纽马克于1952年12月6日出生。
所以最终答案是：1952年12月6日


问题：乔治·华盛顿的外祖父是谁？

是否需要后续问题：是的。
后续问题：乔治·华盛顿的母亲是谁？
中间答案：乔治·华盛顿的母亲是玛丽·波尔·华盛顿。
后续问题：玛丽·波尔·华盛顿的父亲是谁？
中间答案：玛丽·波尔·华盛顿的父亲是约瑟夫·波尔。
所以最终答案是：约瑟夫·波尔


问题：《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？

是否需要后续问题：是的。
后续问题：《大白鲨》的导演是谁？
中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
后续问题：史蒂文·斯皮尔伯格来自哪个国家？
中间答案：美国。
后续问题：《皇家赌场》的导演是谁？
中间答案：《皇家赌场》的导演是马丁·坎贝尔。
后续问题：马丁·坎贝尔来自哪个国家？
中间答案：新西兰。
所以最终答案是：不是


问题：乔治·华盛顿的父亲是谁？
```

<font style="color:rgb(28, 30, 33);">通过向模型提供这样的示例，我们可以引导模型做出更好的回应。</font>

## <font style="color:#080808;background-color:#ffffff;">Example selectors(</font><font style="color:rgb(28, 30, 33);">示例选择器)</font>
<font style="color:rgb(28, 30, 33);">我们将重用上一节中的示例集和格式化程序。但是，我们不会直接将示例馈送到</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">FewShotPromptTemplate</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">对象中，而是将它们馈送到名为</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">SemanticSimilarityExampleSelector</font>](https://api.python.langchain.com/en/latest/example_selectors/langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector.html)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">的</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">ExampleSelector</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">实现实例中。该类根据输入与少样本示例的相似性选择初始集合中的少样本示例。它使用嵌入模型计算输入与少样本示例之间的相似性，以及向量存储库执行最近邻搜索。</font>

<font style="color:rgb(28, 30, 33);">为了展示它的样子，让我们初始化一个实例并在隔离环境中调用它：</font>

```python
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 这是可供选择的示例列表。
    examples,
    # 这是用于生成嵌入的嵌入类，用于衡量语义相似性。
    OpenAIEmbeddings(),
    # 这是用于存储嵌入并进行相似性搜索的 VectorStore 类。
    Chroma,
    # 这是要生成的示例数量。
    k=1,
)
# 选择与输入最相似的示例。
question = "玛丽·波尔·华盛顿的父亲是谁？"
selected_examples = example_selector.select_examples({"question": question})
print(f"与输入最相似的示例: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")
```

```python
与输入最相似的示例: 玛丽·波尔·华盛顿的父亲是谁？
answer: 
是否需要后续问题：是的。
后续问题：乔治·华盛顿的母亲是谁？
中间答案：乔治·华盛顿的母亲是玛丽·波尔·华盛顿。
后续问题：玛丽·波尔·华盛顿的父亲是谁？
中间答案：玛丽·波尔·华盛顿的父亲是约瑟夫·波尔。
所以最终答案是：约瑟夫·波尔
question: 乔治·华盛顿的外祖父是谁？
```

Chroma 安装报错

```powershell
pip install langchain-chroma
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
[end of output]
```

解决方案：需要先下载visual-cpp-build-tools，再执行pip install langchain-chroma

下载地址：[https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722411859844-7a02ba32-0a30-45ee-808c-b20cfd00d1a9.png)

<font style="color:rgb(28, 30, 33);">现在，让我们创建一个 </font>`<font style="color:rgb(28, 30, 33);">FewShotPromptTemplate</font>`<font style="color:rgb(28, 30, 33);"> 对象。该对象接受示例选择器和用于少样本示例的格式化程序提示。</font>

```python
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)
print(
    prompt.invoke({"input": "玛丽·波尔·华盛顿的父亲是谁？"}).to_string()
)
```

```plain
问题：乔治·华盛顿的外祖父是谁？
是否需要后续问题：是的。
后续问题：乔治·华盛顿的母亲是谁？
中间答案：乔治·华盛顿的母亲是玛丽·波尔·华盛顿。
后续问题：玛丽·波尔·华盛顿的父亲是谁？
中间答案：玛丽·波尔·华盛顿的父亲是约瑟夫·波尔。
所以最终答案是：约瑟夫·波尔
Question: 玛丽·波尔·华盛顿的父亲是谁？
```



# LangServe
## <font style="color:rgb(28, 30, 33);">概述</font>
[<font style="color:rgb(28, 30, 33);">LangServe</font>](https://github.com/langchain-ai/langserve)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">🦜</font><font style="color:rgb(28, 30, 33);">️</font><font style="color:rgb(28, 30, 33);">🏓</font><font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">帮助开发者将 </font>`<font style="color:rgb(28, 30, 33);">LangChain</font>`<font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">可运行和链</font>](https://python.langchain.com/docs/expression_language/)<font style="color:rgb(28, 30, 33);">部署为 REST API。</font>

<font style="color:rgb(28, 30, 33);">该库集成了</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">FastAPI</font>](https://fastapi.tiangolo.com/)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">并使用</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">pydantic</font>](https://docs.pydantic.dev/latest/)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">进行数据验证。</font>

**<font style="color:rgb(28, 30, 33);">Pydantic</font>**<font style="color:rgb(28, 30, 33);"> 是一个在 Python中用于数据验证和解析的第三方库，现在是Python中使用广泛的数据验证库。</font>

+ <font style="color:rgb(28, 30, 33);">它利用声明式的方式定义数据模型和Python 类型提示的强大功能来执行数据验证和序列化，使您的代码更可靠、更可读、更简洁且更易于调试。。</font>
+ <font style="color:rgb(28, 30, 33);">它还可以从模型生成 JSON 架构，提供了自动生成文档等功能，从而轻松与其他工具集成</font>

<font style="color:rgb(28, 30, 33);">此外，它提供了一个客户端，可用于调用部署在服务器上的可运行对象。JavaScript 客户端可在</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">LangChain.js</font>](https://js.langchain.com/docs/ecosystem/langserve)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">中找到。</font>

## <font style="color:rgb(28, 30, 33);">特性</font>
+ <font style="color:rgb(28, 30, 33);">从 LangChain 对象自动推断输入和输出模式，并在每次 API 调用中执行，提供丰富的错误信息</font>
+ <font style="color:rgb(28, 30, 33);">带有 JSONSchema 和 Swagger 的 API 文档页面（插入示例链接）</font>
+ <font style="color:rgb(28, 30, 33);">高效的</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">/invoke</font>`<font style="color:rgb(28, 30, 33);">、</font>`<font style="color:rgb(28, 30, 33);">/batch</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">和</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">/stream</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">端点，支持单个服务器上的多个并发请求</font>
+ `<font style="color:rgb(28, 30, 33);">/stream_log</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">端点，用于流式传输链/代理的所有（或部分）中间步骤</font>
+ **<font style="color:rgb(28, 30, 33);">新功能</font>**<font style="color:rgb(28, 30, 33);"> 自 0.0.40 版本起，支持 </font>`<font style="color:rgb(28, 30, 33);">/stream_events</font>`<font style="color:rgb(28, 30, 33);">，使流式传输更加简便，无需解析 </font>`<font style="color:rgb(28, 30, 33);">/stream_log</font>`<font style="color:rgb(28, 30, 33);"> 的输出。</font>
+ <font style="color:rgb(28, 30, 33);">使用经过严格测试的开源 Python 库构建，如 FastAPI、Pydantic、uvloop 和 asyncio。</font>
+ <font style="color:rgb(28, 30, 33);">使用客户端 SDK 调用 LangServe 服务器，就像本地运行可运行对象一样（或直接调用 HTTP API）</font>

## <font style="color:rgb(28, 30, 33);">限制</font>
+ <font style="color:rgb(28, 30, 33);">目前不支持服务器发起的事件的客户端回调</font>
+ <font style="color:rgb(28, 30, 33);">当使用 Pydantic V2 时，将不会生成 OpenAPI 文档。FastAPI 不支持</font>[<font style="color:rgb(28, 30, 33);">混合使用 pydantic v1 和 v2 命名空间</font>](https://github.com/tiangolo/fastapi/issues/10360)<font style="color:rgb(28, 30, 33);">。更多细节请参见下面的章节。</font>

## <font style="color:rgb(28, 30, 33);">安装</font>
<font style="color:rgb(28, 30, 33);">对于客户端和服务器：</font>

```bash
pip install --upgrade "langserve[all]" 
```

<font style="color:rgb(28, 30, 33);">或者对于客户端代码，</font>`<font style="color:rgb(28, 30, 33);">pip install "langserve[client]"</font>`<font style="color:rgb(28, 30, 33);">，对于服务器代码，</font>`<font style="color:rgb(28, 30, 33);">pip install "langserve[server]"</font>`<font style="color:rgb(28, 30, 33);">。</font>

## <font style="color:rgb(28, 30, 33);">LangChain CLI </font><font style="color:rgb(28, 30, 33);">🛠️</font>
<font style="color:rgb(28, 30, 33);">使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">LangChain</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">CLI 快速启动</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">LangServe</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">项目。</font>

<font style="color:rgb(28, 30, 33);">要使用 langchain CLI，请确保已安装最新版本的 </font>`<font style="color:rgb(28, 30, 33);">langchain-cli</font>`<font style="color:rgb(28, 30, 33);">。您可以使用 </font>`<font style="color:rgb(28, 30, 33);">pip install -U langchain-cli</font>`<font style="color:rgb(28, 30, 33);"> 进行安装。</font>

## <font style="color:rgb(28, 30, 33);">设置</font>
**<font style="color:rgb(28, 30, 33);">注意</font>**<font style="color:rgb(28, 30, 33);">：我们使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">poetry</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">进行依赖管理。请参阅 poetry</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">文档</font>](https://python-poetry.org/docs/)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">了解更多信息。</font>

### <font style="color:rgb(28, 30, 33);">1. 使用 langchain cli 命令创建新应用</font>
```plain
langchain app new my-app
```

### <font style="color:rgb(28, 30, 33);">2. 在 add_routes 中定义可运行对象。转到 server.py 并编辑</font>
```plain
add_routes(app. NotImplemented)
```

### <font style="color:rgb(28, 30, 33);">3. 使用 </font>`<font style="color:rgb(28, 30, 33);">poetry</font>`<font style="color:rgb(28, 30, 33);"> 添加第三方包（例如 langchain-openai、langchain-anthropic、langchain-mistral 等）</font>
```powershell
#安装pipx，参考：https://pipx.pypa.io/stable/installation/
pip install pipx 
#加入到环境变量，需要重启PyCharm 
pipx ensurepath

# 安装poetry，参考：https://python-poetry.org/docs/
pipx install poetry


#安装 langchain-openai 库，例如：poetry add [package-name]
poetry add langchain
poetry add langchain-openai 
```

### <font style="color:rgb(28, 30, 33);">4. 设置相关环境变量。例如，</font>
```plain
export OPENAI_API_KEY="sk-..."
```

### <font style="color:rgb(28, 30, 33);">5. 启动您的应用</font>
```plain
poetry run langchain serve --port=8000
```

## <font style="color:rgb(28, 30, 33);">示例应用</font>
## <font style="color:rgb(28, 30, 33);">服务器</font>
<font style="color:rgb(28, 30, 33);">以下是一个部署 OpenAI 聊天模型，讲述有关特定主题笑话的链的服务器。</font>

```python
#!/usr/bin/env python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes
app = FastAPI(
    title="LangChain 服务器",
    version="1.0",
    description="使用 Langchain 的 Runnable 接口的简单 API 服务器",
)
add_routes(
    app,
    ChatOpenAI(model="gpt-4"),
    path="/openai",
)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

<font style="color:rgb(28, 30, 33);">如果您打算从浏览器调用您的端点，您还需要设置 CORS 头。</font>

<font style="color:rgb(28, 30, 33);">您可以使用 FastAPI 的内置中间件来实现：</font>

```python
from fastapi.middleware.cors import CORSMiddleware
# 设置所有启用 CORS 的来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
```

## <font style="color:rgb(28, 30, 33);">文档</font>
<font style="color:rgb(28, 30, 33);">如果您已部署上述服务器，可以使用以下命令查看生成的 OpenAPI 文档：</font>

文档地址：[http://localhost:8000/docs](http://localhost:8000/docs)

```plain
curl localhost:8000/docs
```

<font style="color:rgb(28, 30, 33);">请确保</font>**<font style="color:rgb(28, 30, 33);">添加</font>**<font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">/docs</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">后缀。</font>

⚠️ 首页 `/` 没有被**设计**定义，因此 `curl localhost:8000` 或访问该 URL

将返回 404。如果您想在 `/` 上有内容，请定义一个端点 `@app.get("/")`。

## <font style="color:rgb(28, 30, 33);">客户端</font>
<font style="color:rgb(28, 30, 33);">Python SDK</font>

```python
from langchain.schema.runnable import RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

openai = RemoteRunnable("http://localhost:8000/openai/")
prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一个喜欢写故事的助手"), ("system", "写一个故事，主题是： {topic}")]
)
# 可以定义自定义链
chain = prompt | RunnableMap({
    "openai": openai
})
response = chain.batch([{"topic": "猫"}])
print(response)
#[{'openai': AIMessage(content='从前，有一个叫做肖恩的男孩，他在一个宁静的乡村里生活。一天，他在家的后院发现了一个小小的，萌萌的猫咪。这只猫咪有一双大大的蓝色眼睛，毛色如同朝霞般的粉色，看起来非常可爱。\n\n肖恩把这只猫咪带回了家，他给她取名为“樱花”，因为她的毛色让他联想到春天盛开的樱花。肖恩非常喜欢樱花，他用心照顾她，每天都会为她准备新鲜的食物和清水，还会陪她玩耍，带她去散步。\n\n樱花也非常喜欢肖恩，她会在肖恩读书的时候躺在他的脚边，会在他伤心的时候安慰他，每当肖恩回家的时候，她总是第一个跑出来迎接他。可是，樱花有一个秘密，她其实是一只会说人话的猫。\n\n这个秘密是在一个月圆的夜晚被肖恩发现的。那天晚上，肖恩做了一个噩梦，他从梦中惊醒，发现樱花正坐在他的床边，用人的语言安慰他。肖恩一开始以为自己在做梦，但是当他清醒过来，樱花还在继续讲话，他才知道这是真的。\n\n樱花向肖恩解释，她是一只来自神秘的猫咪国度的使者，她的任务是保护和帮助那些善良和爱护动物的人。肖恩因为对她的善良和照顾，使她决定向他展现真实的自我。\n\n肖恩虽然感到惊讶，但他并没有因此而害怕或者排斥樱花。他觉得这只使得他更加喜欢樱花，觉得这是他们之间的特殊纽带。\n\n从那天开始，肖恩和樱花的关系变得更加亲密，他们像最好的朋友一样，分享彼此的秘密，一起度过快乐的时光。樱花也用她的智慧和力量，帮助肖恩解决了许多困扰他的问题。\n\n许多年过去了，肖恩长大了，他离开了乡村，去了城市上大学。但是，无论他走到哪里，都会带着樱花。他们的友情和互相的陪伴，让他们无论在哪里，都能感到家的温暖。\n\n最后，肖恩成为了一名作家，他写下了自己和樱花的故事，这个故事被人们广为传播，让更多的人知道了这个关于善良、友情和勇气的故事。而樱花，也永远陪伴在肖恩的身边，成为他生活中不可或缺的一部分。\n\n这就是肖恩和樱花的故事，一个关于男孩和他的猫的故事，充满了奇迹、温暖和爱。', response_metadata={'token_usage': {'completion_tokens': 1050, 'prompt_tokens': 33, 'total_tokens': 1083}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-c44f1624-ea75-424b-ba3d-e741baf44bda-0', usage_metadata={'input_tokens': 33, 'output_tokens': 1050, 'total_tokens': 1083})}]

```

<font style="color:rgb(28, 30, 33);">在 TypeScript 中（需要 LangChain.js 版本 0.0.166 或更高）：</font>

```typescript
import { RemoteRunnable } from "@langchain/core/runnables/remote";
const chain = new RemoteRunnable({
  url: `http://localhost:8000/openai/`,
});
const result = await chain.invoke({
  topic: "cats",
});
```

<font style="color:rgb(28, 30, 33);">使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">requests</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">的 Python 代码：</font>

```python
import requests
response = requests.post(
    "http://localhost:8000/openai",
    json={'input': {'topic': 'cats'}}
)
response.json()
```

<font style="color:rgb(28, 30, 33);">您也可以使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">curl</font>`<font style="color:rgb(28, 30, 33);">：</font>

```powershell
curl --location --request POST 'http://localhost:8000/openai/stream' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "input": {
            "topic": "狗"
        }
    }'
```

## <font style="color:rgb(28, 30, 33);">端点</font>
<font style="color:rgb(28, 30, 33);">以下代码：</font>

```python
...
add_routes(
    app,
    runnable,
    path="/my_runnable",
)
```

<font style="color:rgb(28, 30, 33);">将以下端点添加到服务器：</font>

+ `<font style="color:rgb(28, 30, 33);">POST /my_runnable/invoke</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 对单个输入调用可运行项</font>
+ `<font style="color:rgb(28, 30, 33);">POST /my_runnable/batch</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 对一批输入调用可运行项</font>
+ `<font style="color:rgb(28, 30, 33);">POST /my_runnable/stream</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 对单个输入调用并流式传输输出</font>
+ `<font style="color:rgb(28, 30, 33);">POST /my_runnable/stream_log</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 对单个输入调用并流式传输输出，</font>

<font style="color:rgb(28, 30, 33);">包括生成的中间步骤的输出</font>

+ `<font style="color:rgb(28, 30, 33);">POST /my_runnable/astream_events</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 对单个输入调用并在生成时流式传输事件，</font>

<font style="color:rgb(28, 30, 33);">包括来自中间步骤的事件。</font>

+ `<font style="color:rgb(28, 30, 33);">GET /my_runnable/input_schema</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 可运行项的输入的 JSON 模式</font>
+ `<font style="color:rgb(28, 30, 33);">GET /my_runnable/output_schema</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 可运行项的输出的 JSON 模式</font>
+ `<font style="color:rgb(28, 30, 33);">GET /my_runnable/config_schema</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 可运行项的配置的 JSON 模式</font>

<font style="color:rgb(28, 30, 33);">这些端点与</font>[<font style="color:rgb(28, 30, 33);">LangChain 表达式语言接口</font>](https://python.langchain.com/docs/expression_language/interface)<font style="color:rgb(28, 30, 33);">相匹配 --</font>



# 为 Chain 添加 Message history (Memory)单行初始化 chat model
## <font style="color:rgb(28, 30, 33);">对话状态Chain传递</font>
<font style="color:rgb(28, 30, 33);">在构建聊天机器人时，将对话状态传递到链中以及从链中传出对话状态至关重要。</font>[<font style="color:rgb(28, 30, 33);">RunnableWithMessageHistory</font>](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory)<font style="color:rgb(28, 30, 33);"> 类让我们能够向某些类型的链中添加消息历史。它包装另一个 Runnable 并管理其聊天消息历史。</font>

<font style="color:rgb(28, 30, 33);">具体来说，它可用于任何接受以下之一作为输入的 Runnable：</font>

+ <font style="color:rgb(28, 30, 33);">一系列</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">BaseMessages</font>](http://www.aidoczh.com/langchain/v0.2/docs/concepts/#message-types)
+ <font style="color:rgb(28, 30, 33);">具有以序列</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">BaseMessages</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">作为值的键的字典</font>
+ <font style="color:rgb(28, 30, 33);">具有以字符串或序列</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">BaseMessages</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">作为最新消息的值的键和一个接受历史消息的单独键的字典</font>

<font style="color:rgb(28, 30, 33);">并将以下之一作为输出返回：</font>

+ <font style="color:rgb(28, 30, 33);">可视为</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">AIMessage</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">内容的字符串</font>
+ <font style="color:rgb(28, 30, 33);">一系列</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">BaseMessage</font>`
+ <font style="color:rgb(28, 30, 33);">具有包含一系列</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">BaseMessage</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">的键的字典</font>

<font style="color:rgb(28, 30, 33);">让我们看一些示例以了解其工作原理。首先，我们构建一个 Runnable（此处接受字典作为输入并返回消息作为输出）：</font>

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
runnable = prompt | model
```

```python
first=ChatPromptTemplate(input_variables=['ability', 'history', 'input'], input_types={'history': typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage, langchain_core.messages.system.SystemMessage, langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]]}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['ability'], template="You're an assistant who's good at {ability}. Respond in 20 words or fewer")), MessagesPlaceholder(variable_name='history'), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))]) last=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x0000026A478DB440>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x0000026A478FCD10>, model_name='gpt-4', openai_api_key=SecretStr('**********'), openai_proxy='')

```

<font style="color:rgb(28, 30, 33);">要管理消息历史，我们需要：</font>

1. <font style="color:rgb(28, 30, 33);">此 Runnable；</font>
2. <font style="color:rgb(28, 30, 33);">一个返回</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">BaseChatMessageHistory</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">实例的可调用对象。</font>



## <font style="color:rgb(28, 30, 33);">聊天历史存储在内存</font>
<font style="color:rgb(28, 30, 33);">下面我们展示一个简单的示例，其中聊天历史保存在内存中，此处通过全局 Python 字典实现。</font>

<font style="color:rgb(28, 30, 33);">我们构建一个名为</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">get_session_history</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">的可调用对象，引用此字典以返回</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">ChatMessageHistory</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">实例。通过在运行时向</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">RunnableWithMessageHistory</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">传递配置，可以指定可调用对象的参数。默认情况下，期望配置参数是一个字符串</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">session_id</font>`<font style="color:rgb(28, 30, 33);">。可以通过</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">history_factory_config</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">关键字参数进行调整。</font>

<font style="color:rgb(28, 30, 33);">使用单参数默认值：</font>

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

<font style="color:rgb(28, 30, 33);">请注意，我们已指定了</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">input_messages_key</font>`<font style="color:rgb(28, 30, 33);">（要视为最新输入消息的键）和</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">history_messages_key</font>`<font style="color:rgb(28, 30, 33);">（要添加历史消息的键）。</font>

<font style="color:rgb(28, 30, 33);">在调用此新 Runnable 时，我们通过配置参数指定相应的聊天历史：</font>

```python
with_message_history.invoke(
    {"ability": "math", "input": "余弦是什么意思？"},
    config={"configurable": {"session_id": "abc123"}},
)
```

```plain
content='余弦是一个数学函数，通常在三角学中使用，表示直角三角形的邻边和斜边的比例。' response_metadata={'token_usage': {'completion_tokens': 38, 'prompt_tokens': 38, 'total_tokens': 76}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-9aa23716-3959-476d-9386-6d433266e060-0' usage_metadata={'input_tokens': 38, 'output_tokens': 38, 'total_tokens': 76}
```

```python
# 记住
with_message_history.invoke(
    {"ability": "math", "input": "什么?"},
    config={"configurable": {"session_id": "abc123"}},
)
```

```python
content='余弦是一个数学术语，用于描述直角三角形中的角度关系。' response_metadata={'token_usage': {'completion_tokens': 26, 'prompt_tokens': 88, 'total_tokens': 114}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-f77baf90-6a13-4f48-991a-28e60ece84e8-0' usage_metadata={'input_tokens': 88, 'output_tokens': 26, 'total_tokens': 114}
```

```python
# 新的 session_id --> 不记得了。
with_message_history.invoke(
    {"ability": "math", "input": "什么?"},
    config={"configurable": {"session_id": "def234"}},
)
```

```python
content='对不起，我没明白你的问题。你能再详细一点吗？我很擅长数学。' response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 32, 'total_tokens': 66}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-3f69d281-a850-452f-8055-df70d4936630-0' usage_metadata={'input_tokens': 32, 'output_tokens': 34, 'total_tokens': 66}
```

## 
# 基于 LangChain 的 Chatbot: Chat History
## <font style="color:rgb(28, 30, 33);">配置会话唯一键</font>
<font style="color:rgb(28, 30, 33);">我们可以通过向 </font>`<font style="color:rgb(28, 30, 33);">history_factory_config</font>`<font style="color:rgb(28, 30, 33);"> 参数传递一个 </font>`<font style="color:rgb(28, 30, 33);">ConfigurableFieldSpec</font>`<font style="color:rgb(28, 30, 33);"> 对象列表来自定义跟踪消息历史的配置参数。下面我们使用了两个参数：</font>`<font style="color:rgb(28, 30, 33);">user_id</font>`<font style="color:rgb(28, 30, 33);"> 和 </font>`<font style="color:rgb(28, 30, 33);">conversation_id</font>`<font style="color:rgb(28, 30, 33);">。</font>

<font style="color:rgb(28, 30, 33);">配置user_id和conversation_id作为会话唯一键</font>

```python
from langchain_core.runnables import ConfigurableFieldSpec
store = {}
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]
with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户的唯一标识符。",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="对话的唯一标识符。",
            default="",
            is_shared=True,
        ),
    ],
)
with_message_history.invoke(
    {"ability": "math", "input": "余弦是什么意思？"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}},
)
```

```plain
content='对不起，你能提供一些更详细的信息吗？我会很高兴帮助你解决数学问题。' response_metadata={'token_usage': {'completion_tokens': 38, 'prompt_tokens': 32, 'total_tokens': 70}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-02030348-7bbb-4f76-8c68-61785d012c26-0' usage_metadata={'input_tokens': 32, 'output_tokens': 38, 'total_tokens': 70}
```



<font style="color:rgb(28, 30, 33);">在许多情况下，持久化对话历史是可取的。</font>`<font style="color:rgb(28, 30, 33);">RunnableWithMessageHistory</font>`<font style="color:rgb(28, 30, 33);"> 对于 </font>`<font style="color:rgb(28, 30, 33);">get_session_history</font>`<font style="color:rgb(28, 30, 33);"> 可调用如何检索其聊天消息历史是中立的。请参见这里 ，这是一个使用本地文件系统的示例。下面我们演示如何使用 Redis。请查看内存集成 页面，以获取使用其他提供程序的聊天消息历史的实现。</font>

## <font style="color:rgb(28, 30, 33);">消息持久化</font>
<font style="color:rgb(28, 30, 33);">请查看 </font>[<font style="color:rgb(28, 30, 33);">memory integrations</font>](https://integrations.langchain.com/memory)<font style="color:rgb(28, 30, 33);"> 页面，了解使用 Redis 和其他提供程序实现聊天消息历史的方法。这里我们演示使用内存中的 </font>`<font style="color:rgb(28, 30, 33);">ChatMessageHistory</font>`<font style="color:rgb(28, 30, 33);"> 以及使用 </font>`<font style="color:rgb(28, 30, 33);">RedisChatMessageHistory</font>`<font style="color:rgb(28, 30, 33);"> 进行更持久存储。</font>

### <font style="color:rgb(28, 30, 33);">配置redis环境</font>
<font style="color:rgb(28, 30, 33);">如果尚未安装 Redis，我们需要安装它：</font>

```python
%pip install --upgrade --quiet redis
```

<font style="color:rgb(28, 30, 33);">如果我们没有现有的 Redis 部署可以连接，可以启动本地 Redis Stack 服务器：</font>

```bash
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

```python
REDIS_URL = "redis://localhost:6379/0"
```



### <font style="color:rgb(28, 30, 33);">调用聊天接口，看Redis是否存储历史记录</font>
<font style="color:rgb(28, 30, 33);">更新消息历史实现只需要我们定义一个新的可调用对象，这次返回一个 </font>`<font style="color:rgb(28, 30, 33);">RedisChatMessageHistory</font>`<font style="color:rgb(28, 30, 33);"> 实例：</font>

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)
with_message_history = RunnableWithMessageHistory(
    runnable,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

<font style="color:rgb(28, 30, 33);">我们可以像以前一样调用：</font>

```python
with_message_history.invoke(
    {"ability": "math", "input": "余弦是什么意思？"},
    config={"configurable": {"session_id": "foobar"}},
)
```

```plain
content='余弦是一个三角函数，它表示直角三角形的邻边长度和斜边长度的比值。' response_metadata={'token_usage': {'completion_tokens': 33, 'prompt_tokens': 38, 'total_tokens': 71}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-2d1eba02-4709-4db5-ab6b-0fd03ab4c68a-0' usage_metadata={'input_tokens': 38, 'output_tokens': 33, 'total_tokens': 71}
```

```python
with_message_history.invoke(
    {"ability": "math", "input": "什么?"},
    config={"configurable": {"session_id": "foobar"}},
)
```

```plain
content='余弦是一个数学术语，代表在一个角度下的邻边和斜边的比例。' response_metadata={'token_usage': {'completion_tokens': 32, 'prompt_tokens': 83, 'total_tokens': 115}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-99368d03-c2ed-4dda-a32f-677c036ad676-0' usage_metadata={'input_tokens': 83, 'output_tokens': 32, 'total_tokens': 115}
```

redis历史记录查询

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722421493358-fed79174-96ba-4f32-809b-6c153c5513de.png)



# Track token usage, Cache model responses
## Track token usage(跟踪token使用情况)
<font style="color:rgb(28, 30, 33);">跟踪令牌使用情况以计算成本是将您的应用投入生产的重要部分。本指南介绍了如何从您的 LangChain 模型调用中获取此信息。</font>

### <font style="color:rgb(28, 30, 33);">使用 AIMessage.response_metadata</font>
<font style="color:rgb(28, 30, 33);">许多模型提供程序将令牌使用信息作为聊天生成响应的一部分返回。如果可用，这将包含在</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">AIMessage.response_metadata</font>](http://www.aidoczh.com/langchain/v0.2/docs/how_to/response_metadata/)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">字段中。以下是一个使用 OpenAI 的示例：</font>

```python
# !pip install -qU langchain-openai
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4-turbo")
msg = llm.invoke([("human", "最古老的楔形文字的已知例子是什么")])
msg.response_metadata
```

```python
{'token_usage': {'completion_tokens': 114, 'prompt_tokens': 25, 'total_tokens': 139}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}
```



### <font style="color:rgb(28, 30, 33);">使用回调</font>
<font style="color:rgb(28, 30, 33);">还有一些特定于 API 的回调上下文管理器，允许您跟踪多个调用中的令牌使用情况。目前仅为 OpenAI API 和 Bedrock Anthropic API 实现了此功能。</font>

<font style="color:rgb(28, 30, 33);">让我们首先看一个极其简单的示例，用于跟踪单个 Chat 模型调用的令牌使用情况。</font>

```python
# !pip install -qU langchain-community wikipedia
from langchain_community.callbacks.manager import get_openai_callback
llm = ChatOpenAI(model="gpt-4", temperature=0)
with get_openai_callback() as cb:
    result = llm.invoke("告诉我一个笑话")
    print(cb)
```

```plain
Tokens Used: 59
Prompt Tokens: 14
Completion Tokens: 45
Successful Requests: 1
Total Cost (USD): $0.0031199999999999995
----------------------------------------
使用的令牌数：59
    提示令牌：14
    完成令牌：4
成功请求次数：1
总成本（美元）：$0.0031199999999999995
```

<font style="color:rgb(28, 30, 33);">上下文管理器中的任何内容都将被跟踪。以下是在其中使用它来跟踪连续多次调用的示例。</font>

```python
with get_openai_callback() as cb:
    result = llm.invoke("告诉我一个笑话")
        result2 = llm.invoke("告诉我一个笑话")
    print(cb.total_tokens)
```

```plain
114
```

<font style="color:rgb(28, 30, 33);">如果使用具有多个步骤的链或代理，它将跟踪所有这些步骤。</font>

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent, load_tools
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "您是一个乐于助人的助手"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
tools = load_tools(["wikipedia"])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, stream_runnable=False
)
```

我们必须将 `stream_runnable=False` 设置为令牌计数才能正常工作。默认情况下，AgentExecutor 将流式传输底层代理，以便在通过 AgentExecutor.stream_events 流式传输事件时获得最精细的结果。但是，OpenAI 在流式传输模型响应时不会返回令牌计数，因此我们需要关闭底层流式传输。

```python
with get_openai_callback() as cb:
    response = agent_executor.invoke(
        {
            "input": "蜂鸟的学名是什么，哪种鸟是最快的？"
        }
    )
    print(f"总令牌数：{cb.total_tokens}")
    print(f"提示令牌：{cb.prompt_tokens}")
    print(f"完成令牌：{cb.completion_tokens}")
    print(f"总成本（美元）：${cb.total_cost}")
```

```plain
> Entering new AgentExecutor chain...

Invoking: `wikipedia` with `{'query': '蜂鸟'}`


Page: Hawick Lau
Summary: Hawick Lau Hoi-wai (Chinese: 劉愷威; born 13 October 1974) is a Hong Kong actor and singer. He was named as one of the "Five Fresh Tigers of TVB" and is best known for his performances in the series A Kindred Spirit (1995), Virtues of Harmony (2001) and My Family (2005).
He then expanded his career into mainland China, acting in several notable series. His notable appearances include Sealed with a Kiss (2011), A Clear Midsummer Night (2013), The Wife's Secret (2014), Lady & Liar (2015) and Chronicle of Life (2016).

Page: Zhang Jianing
Summary: Zhang Jianing (Chinese: 张佳宁, born 26 May 1989), also known as Karlina Zhang, is a Chinese actress. She is best known for her roles as Muyun Yanshuang in Tribes and Empires: Storm of Prophecy (2017) and Lin Beixing in Shining for One Thing (2022).

Page: Li Xirui
Summary: Li Xirui (Chinese: 李溪芮; born 30 January 1990) is a Chinese actress and singer.
Invoking: `wikipedia` with `{'query': '蜂鸟学名'}`


No good Wikipedia Search Result was found
Invoking: `wikipedia` with `{'query': 'fastest bird'}`


Page: Fastest animals
Summary: This is a list of the fastest animals in the world, by types of animal.

Page: List of birds by flight speed
Summary: This is a list of the fastest flying birds in the world. A bird's velocity is necessarily variable; a hunting bird will reach much greater speeds while diving to catch prey than when flying horizontally. The bird that can achieve the greatest airspeed is the peregrine falcon (Falco peregrinus), able to exceed 320 km/h (200 mph) in its dives. A close relative of the common swift, the white-throated needletail (Hirundapus caudacutus), is commonly reported as the fastest bird in level flight with a reported top speed of 169 km/h (105 mph). This record remains unconfirmed as the measurement methods have never been published or verified. The record for the fastest confirmed level flight by a bird is 111.5 km/h (69.3 mph) held by the common swift.

Page: Abdul Khaliq (athlete)
Summary: Subedar Abdul Khaliq (Punjabi, Urdu: عبد الخالق; 23 March 1933 – 10 March 1988), also known as Parinda-e-Asia (Urdu for The Flying Bird of Asia), was a Pakistani sprinter from 8 Medium Regiment Artillery who won 36 international gold medals, 15 international silver medals, and 12 International bronze medals while representing Pakistan. He competed in the 100m, 200m, and 4 x 100 meters relay. He participated in the 1956 Melbourne Olympics and the 1960 Rome Olympics. He also participated in the 1954 Asian Games and the 1958 Asian Games. During the 1956 Indo-Pak Meet held in Delhi, Abdul Khaliq was first referred to as "The Flying Bird of Asia" by the Prime Minister of India of the time was Jawaharlal Nehru, who was reportedly captivated by his performance during the event.蜂鸟的学名是Trochilidae。最快的鸟是游隼（Falco peregrinus），在俯冲捕食时，速度可以超过320公里/小时（200英里/小时）。在水平飞行中，最快的鸟是普通雨燕，其确认的最高速度为111.5公里/小时（69.3英里/小时）。

> Finished chain.
总令牌数：2088
提示令牌：1922
完成令牌：166
总成本（美元）：$0.06762
```



## Cache model responses
<font style="color:rgb(28, 30, 33);"> LangChain为聊天模型提供了一个可选的缓存层。这很有用，主要有两个原因：</font>

+ <font style="color:rgb(28, 30, 33);"></font><font style="color:rgb(28, 30, 33);">如果您经常多次请求相同的完成，它可以通过减少您向 LLM 提供商进行的 API 调用次数来为您节省资金。这在应用程序开发过程中特别有用。</font>
+ <font style="color:rgb(28, 30, 33);">它可以通过减少您向 LLM 提供商进行的 API 调用次数来加快您的应用程序。</font>

```bash
pip install -qU langchain-openai
```

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
```

```python
from langchain.globals import set_llm_cache
```

**API Reference: **[set_llm_cache](https://api.python.langchain.com/en/latest/globals/langchain.globals.set_llm_cache.html)

### <font style="color:rgb(28, 30, 33);">In Memory Cache 内存缓存中</font>
<font style="color:rgb(28, 30, 33);">This is an ephemeral cache that stores model calls in memory. It will be wiped when your environment restarts, and is not shared across processes.  
</font><font style="color:rgb(28, 30, 33);">这是一个临时缓存，用于在内存中存储模型调用。当您的环境重新启动时，它将被擦除，并且不会在进程之间共享。</font>

```python
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# 创建LLM实例
llm = ChatOpenAI(model="gpt-4")
set_llm_cache(InMemoryCache())

def measure_invoke_time(llm, prompt):
    # 记录开始时间
    start_wall_time = time.time()
    start_cpu_times = os.times()

    # 调用LLM
    response = llm.invoke(prompt)

    # 记录结束时间
    end_wall_time = time.time()
    end_cpu_times = os.times()

    # 计算经过的时间
    wall_time = end_wall_time - start_wall_time
    user_time = end_cpu_times.user - start_cpu_times.user
    sys_time = end_cpu_times.system - start_cpu_times.system
    total_cpu_time = user_time + sys_time
    return response, wall_time, user_time, sys_time, total_cpu_time

```

**API Reference: **[InMemoryCache](https://api.python.langchain.com/en/latest/cache/langchain_community.cache.InMemoryCache.html)



```plain
# 第一次调用
First call response: content='当然，这是一则关于数学的笑话：\n\n为什么植物恨数学？\n\n因为它给他们太多的根问题（sqrt问题）。' response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 14, 'total_tokens': 60}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-40d86131-39ad-42c9-b3b8-5a422343ba9b-0' usage_metadata={'input_tokens': 14, 'output_tokens': 46, 'total_tokens': 60}
First call CPU times: user 109 ms, sys: 31 ms, total: 141 ms
First call Wall time: 3654 ms
```

```plain
content='当然，这是一则关于数学的笑话：\n\n为什么植物恨数学？\n\n因为它给他们太多的根问题（sqrt问题）。'
```

```python
#第二次调用使用缓存，所以速度很快
response2, wall_time2, user_time2, sys_time2, total_cpu_time2 = measure_invoke_time(llm, "给我讲个笑话")
print("Second call response:", response2)
print(f"Second call CPU times: user {user_time2 * 1000:.0f} ms, sys: {sys_time2 * 1000:.0f} ms, total: {total_cpu_time2 * 1000:.0f} ms")
print(f"Second call Wall time: {wall_time2 * 1000:.0f} ms")
```

```plain
Second call response: content='当然，这是一则关于数学的笑话：\n\n为什么植物恨数学？\n\n因为它给他们太多的根问题（sqrt问题）。' response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 14, 'total_tokens': 60}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-40d86131-39ad-42c9-b3b8-5a422343ba9b-0' usage_metadata={'input_tokens': 14, 'output_tokens': 46, 'total_tokens': 60}
Second call CPU times: user 16 ms, sys: 0 ms, total: 16 ms
Second call Wall time: 1 ms
```

```plain
content='当然，这是一则关于数学的笑话：\n\n为什么植物恨数学？\n\n因为它给他们太多的根问题（sqrt问题）。'
```

### <font style="color:rgb(28, 30, 33);">SQLite Cache SQLite 缓存</font>
<font style="color:rgb(28, 30, 33);">This cache implementation uses a </font>`<font style="color:rgb(28, 30, 33);">SQLite</font>`<font style="color:rgb(28, 30, 33);"> database to store responses, and will last across process restarts.  
</font><font style="color:rgb(28, 30, 33);">此缓存实现使用 </font>`<font style="color:rgb(28, 30, 33);">SQLite</font>`<font style="color:rgb(28, 30, 33);"> 数据库来存储响应，并将在进程重启后持续进行。</font>

```python
# We can do the same thing with a SQLite cache
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

**API Reference: **[SQLiteCache](https://api.python.langchain.com/en/latest/cache/langchain_community.cache.SQLiteCache.html)

```plain
# 第一次调用
First call response: content='当然，这是一则关于数学的笑话：\n\n为什么植物恨数学？\n\n因为它给他们太多的根问题（sqrt问题）。' response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 14, 'total_tokens': 60}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-40d86131-39ad-42c9-b3b8-5a422343ba9b-0' usage_metadata={'input_tokens': 14, 'output_tokens': 46, 'total_tokens': 60}
First call CPU times: user 109 ms, sys: 31 ms, total: 141 ms
First call Wall time: 3654 ms
```

```plain
content='好的，这是一个关于电脑的笑话：\n\n为什么电脑经常感冒？\n\n因为它窗户(Window)太多了。'
```

```python
#第二次调用使用缓存，所以速度很快
response2, wall_time2, user_time2, sys_time2, total_cpu_time2 = measure_invoke_time(llm, "给我讲个笑话")
print("Second call response:", response2)
print(f"Second call CPU times: user {user_time2 * 1000:.0f} ms, sys: {sys_time2 * 1000:.0f} ms, total: {total_cpu_time2 * 1000:.0f} ms")
print(f"Second call Wall time: {wall_time2 * 1000:.0f} ms")
```

```plain
Second call response: content='当然，这是一则关于数学的笑话：\n\n为什么植物恨数学？\n\n因为它给他们太多的根问题（sqrt问题）。' response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 14, 'total_tokens': 60}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-40d86131-39ad-42c9-b3b8-5a422343ba9b-0' usage_metadata={'input_tokens': 14, 'output_tokens': 46, 'total_tokens': 60}
Second call CPU times: user 16 ms, sys: 0 ms, total: 16 ms
Second call Wall time: 1 ms
```

```plain
content='好的，这是一个关于电脑的笑话：\n\n为什么电脑经常感冒？\n\n因为它窗户(Window)太多了。'
```

