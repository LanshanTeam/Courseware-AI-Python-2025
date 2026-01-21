# Multimode 集成
## <font style="color:rgb(28, 30, 33);">多模态数据传输</font>
<font style="color:rgb(28, 30, 33);">这里我们演示如何将多模态输入直接传递给模型。我们目前期望所有输入都以与</font>[<font style="color:rgb(28, 30, 33);">OpenAI 期望的</font>](https://platform.openai.com/docs/guides/vision)<font style="color:rgb(28, 30, 33);">格式相同的格式传递。对于支持多模态输入的其他模型提供者，我们在类中添加了逻辑以转换为预期格式。</font>

<font style="color:rgb(28, 30, 33);">在这个例子中，我们将要求模型描述一幅图像。</font>

```python
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
```

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
```

**API 参考：**[HumanMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.human.HumanMessage.html) | [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

<font style="color:rgb(28, 30, 33);">最常支持的传入图像的方式是将其作为字节字符串传入。这应该适用于大多数模型集成。</font>

```python
import base64
import httpx

image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
```

```python
message = HumanMessage(
    content=[
        {"type": "text", "text": "用中文描述这张图片中的天气"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)
response = model.invoke([message])
print(response.content)
```

```plain
这张图片展示了一个晴朗的天气。天空中有一些淡淡的云，阳光明媚，照亮了图中的草地和木板路。天空呈现出明亮的蓝色，与绿色的草地形成了鲜明的对比。整体感觉是非常清新和舒适的，适合户外活动和散步。
```

<font style="color:rgb(28, 30, 33);">我们可以在“image_url”类型的内容块中直接提供图像 URL。请注意，只有部分模型提供商支持此功能。</font>

```python
message = HumanMessage(
    content=[
        {"type": "text", "text": "用中文描述这张图片中的天气"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ],
)
response = model.invoke([message])
print(response.content)
```

```plain
图片中的天气晴朗，天空中有一些稀薄的白云，整体呈现出蓝色。阳光明媚，光线充足，草地和树木显得非常绿意盎然。这种天气非常适合户外活动，比如散步或野餐。总的来说，天气非常舒适宜人。
```

<font style="color:rgb(28, 30, 33);">我们还可以传入多幅图像。</font>

```python
message = HumanMessage(
    content=[
        {"type": "text", "text": "这两张图片是一样的吗？"},
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "image_url", "image_url": {"url": image_url}},
    ],
)
response = model.invoke([message])
print(response.content)
```

```plain
这两张图片不一样。第一张是一个晴天的草地景色，有一条木板小路通向远方；第二张是一个覆盖着雪的村庄，有多栋房屋和一些红色灯笼。两张图片显示的是完全不同的场景。
```

## <font style="color:rgb(28, 30, 33);">工具</font>[<font style="color:rgb(28, 30, 33);">调用</font>](https://python.langchain.com/v0.2/docs/how_to/multimodal_inputs/#tool-calls)
<font style="color:rgb(28, 30, 33);">一些多模态模型也</font><font style="color:rgb(28, 30, 33);">支持</font>[<font style="color:rgb(28, 30, 33);">工具调用功能。要使用此类模型调用工具，只需以</font>](https://python.langchain.com/v0.2/docs/concepts/#functiontool-calling)[<font style="color:rgb(28, 30, 33);">通常的方式</font>](https://python.langchain.com/v0.2/docs/how_to/tool_calling/)<font style="color:rgb(28, 30, 33);">将工具绑定到它们，然后使用所需类型的内容块（例如，包含图像数据）调用模型。</font>

```python
from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


@tool
def weather_tool(weather: Literal["晴朗的", "多云的", "多雨的","下雪的"]) -> None:
    """Describe the weather"""
    pass


model = ChatOpenAI(model="gpt-4o")
model_with_tools = model.bind_tools([weather_tool])
image_url_1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_url_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Morning_in_China_Snow_Town.jpg/1280px-Morning_in_China_Snow_Town.jpg"

message = HumanMessage(
    content=[
        {"type": "text", "text": "用中文描述两张图片中的天气"},
        {"type": "image_url", "image_url": {"url": image_url_1}},
        {"type": "image_url", "image_url": {"url": image_url_2}},
    ],
)
response = model_with_tools.invoke([message])
print(response.tool_calls)

```

**API 参考：**[工具](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.tool.html)

```python
[{'name': 'weather_tool', 'args': {'weather': '晴朗的'}, 'id': 'call_7vbVxf7xnHvBqpO5SkVCt5xq', 'type': 'tool_call'}, {'name': 'weather_tool', 'args': {'weather': '下雪的'}, 'id': 'call_zm5zOZgSTd8R57N23aBbIfwX', 'type': 'tool_call'}]
```

# Ouput parsers: JSON, XML
## <font style="color:rgb(28, 30, 33);">如何解析 JSON 输出</font>
<font style="color:rgb(28, 30, 33);">虽然一些模型提供商支持内置的方法返回结构化输出，但并非所有都支持。我们可以使用输出解析器来帮助用户通过提示指定任意的 JSON 模式，查询符合该模式的模型输出，最后将该模式解析为 JSON。</font>

请记住，大型语言模型是有泄漏的抽象！您必须使用具有足够容量的大型语言模型来生成格式良好的 JSON。

<font style="color:rgb(28, 30, 33);">JsonOutputParser 是一个内置选项，用于提示并解析 JSON 输出。虽然它在功能上类似于 </font>[<font style="color:rgb(28, 30, 33);">PydanticOutputParser</font>](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.pydantic.PydanticOutputParser.html)<font style="color:rgb(28, 30, 33);">，但它还支持流式返回部分 JSON 对象。</font>

<font style="color:rgb(28, 30, 33);">以下是如何将其与</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">Pydantic</font>](https://docs.pydantic.dev/)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">一起使用以方便地声明预期模式的示例：</font>

```python
%pip install -qU langchain langchain-openai
```

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
model = ChatOpenAI(temperature=0)
# 定义您期望的数据结构。
class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")
# 还有一个用于提示语言模型填充数据结构的查询意图。
joke_query = "告诉我一个笑话。"
# 设置解析器 + 将指令注入提示模板。
parser = JsonOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
chain.invoke({"query": joke_query})
```

```plain
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它们有很好的防火墙！'}
```

<font style="color:#080808;background-color:#ffffff;">langchain v0.3与pydantic_v1 兼容问题，调整引用方式</font>

```python
#langchain v0.2 使用以下引用pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
#langchain v0.3 使用以下引用pydantic
from pydantic import BaseModel, Field
```

```python
sys:1: LangChainDeprecationWarning: As of langchain-core 0.3.0, LangChain uses pydantic v2 internally. The langchain_core.pydantic_v1 module was a compatibility shim for pydantic v1, and should no longer be used. Please update the code to import from Pydantic directly.

For example, replace imports like: `from langchain_core.pydantic_v1 import BaseModel`
with: `from pydantic import BaseModel`
or the v1 compatibility namespace if you are working in a code base that has not been fully upgraded to pydantic 2 yet. 	from pydantic.v1 import BaseModel

```

<font style="color:rgb(28, 30, 33);"></font>

<font style="color:rgb(28, 30, 33);">请注意，我们将解析器中的 </font>`<font style="color:rgb(28, 30, 33);">format_instructions</font>`<font style="color:rgb(28, 30, 33);"> 直接传递到提示中。您可以并且应该尝试在提示的其他部分中添加自己的格式提示，以增强或替换默认指令：</font>

```python
parser.get_format_instructions()
```

```json
'输出应格式化为符合以下 JSON 模式的 JSON 实例。\n\n例如，对于模式 {"properties": {"foo": {"title": "Foo", "description": "字符串列表", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}，对象 {"foo": ["bar", "baz"]} 是该模式的格式良好实例。对象 {"properties": {"foo": ["bar", "baz"]}} 不是格式良好的。\n\n这是输出模式：\n```\n{"properties": {"setup": {"title": "Setup", "description": "设置笑话的问题", "type": "string"}, "punchline": {"title": "Punchline", "description": "解决笑话的答案", "type": "string"}}, "required": ["setup", "punchline"]}\n```'
```

### <font style="color:rgb(28, 30, 33);">流式处理</font>
<font style="color:rgb(28, 30, 33);">如上所述，</font>`<font style="color:rgb(28, 30, 33);">JsonOutputParser</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">和</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">PydanticOutputParser</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">之间的一个关键区别是</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">JsonOutputParser</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">输出解析器支持流式处理部分块。以下是其示例：</font>

```python
for s in chain.stream({"query": joke_query}):
    print(s)
```

```plain
{}
{'setup': ''}
{'setup': '为什么'}
{'setup': '为什么计算'}
{'setup': '为什么计算机'}
{'setup': '为什么计算机不能'}
{'setup': '为什么计算机不能得'}
{'setup': '为什么计算机不能得感'}
{'setup': '为什么计算机不能得感冒'}
{'setup': '为什么计算机不能得感冒？'}
{'setup': '为什么计算机不能得感冒？', 'punchline': ''}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为'}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它'}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它们'}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它们有'}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它们有很'}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它们有很好的'}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它们有很好的防'}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它们有很好的防火'}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它们有很好的防火墙'}
{'setup': '为什么计算机不能得感冒？', 'punchline': '因为它们有很好的防火墙！'}'


```

你也可以在没有 Pydantic 的情况下使用 `JsonOutputParser`。这将提示模型返回 JSON，但不提供关于模式应该是什么的具体信息。

```python
## 没有 Pydantic
joke_query = "告诉我一个笑话。"
parser = JsonOutputParser()
prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
chain.invoke({"query": joke_query})
```



```plain
{'joke': '为什么数学书总是很难过？因为它有太多的问题！'}
```



## <font style="color:rgb(28, 30, 33);">如何解析 XML 输出</font>
<font style="color:rgb(28, 30, 33);">下面使用</font>[<font style="color:rgb(28, 30, 33);">XMLOutputParser</font>](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.xml.XMLOutputParser.html)<font style="color:rgb(28, 30, 33);">来提示模型生成XML输出，然后将该输出解析为可用的格式。</font>

<font style="color:rgb(28, 30, 33);">我们可以使用</font>`<font style="color:rgb(28, 30, 33);">XMLOutputParser</font>`<font style="color:rgb(28, 30, 33);">将默认的格式指令添加到提示中，并将输出的XML解析为字典：</font>

```python
parser = XMLOutputParser()
# 我们将在下面的提示中添加这些指令
parser.get_format_instructions()
```

```plain
输出应格式化为XML文件。
1.输出应符合以下标签。
2.如果没有标签，请自己制作。
3.记得始终打开和关闭所有标签。

例如，对于标签["foo"，"bar"，"baz"]：
1. String "<foo>
   <bar>
      <baz></baz>
   </bar>
</foo>" 是模式的一个格式良好的实例。
2. String "<foo>
   <bar>
   </foo>" 是一个格式错误的实例。
3. String "<foo>
   <tag>
   </tag>
</foo>" 是一个格式错误的实例。输出应格式化为XML文件。
```

<font style="color:rgb(28, 30, 33);"></font>

```python
from langchain_openai import ChatOpenAI
# pip install -qU langchain langchain-openai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import XMLOutputParser
# pip install defusedxml

model = ChatOpenAI(model="gpt-4o", temperature=0)

# 还有一个用于提示语言模型填充数据结构的查询意图。
actor_query = "生成周星驰的简化电影作品列表，按照最新的时间降序"
# 设置解析器 + 将指令注入提示模板。
parser = XMLOutputParser()
prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
# print(parser.get_format_instructions())
chain = prompt | model
response = chain.invoke({"query": actor_query})
xml_output = parser.parse(response.content)
print(response.content)

```

```xml
<movies>
  <movie>
    <title>美人鱼</title>
    <year>2016</year>
  </movie>
  <movie>
    <title>西游降魔篇</title>
    <year>2013</year>
  </movie>
  <movie>
    <title>长江七号</title>
    <year>2008</year>
  </movie>
  <movie>
    <title>功夫</title>
    <year>2004</year>
  </movie>
  <movie>
    <title>少林足球</title>
    <year>2001</year>
  </movie>
  <movie>
    <title>喜剧之王</title>
    <year>1999</year>
  </movie>
  <movie>
    <title>大话西游之大圣娶亲</title>
    <year>1995</year>
  </movie>
  <movie>
    <title>大话西游之月光宝盒</title>
    <year>1995</year>
  </movie>
  <movie>
    <title>唐伯虎点秋香</title>
    <year>1993</year>
  </movie>
  <movie>
    <title>逃学威龙</title>
    <year>1991</year>
  </movie>
</movies>
```



<font style="color:rgb(28, 30, 33);">我们还可以添加一些标签以根据我们的需求定制输出。您可以在提示的其他部分中尝试添加自己的格式提示，以增强或替换默认指令：</font>

```python
parser = XMLOutputParser(tags=["movies", "actor", "film", "name", "genre"])
# 我们将在下面的提示中添加这些指令
parser.get_format_instructions()
```

```plain
The output should be formatted as a XML file.
1. Output should conform to the tags below. 
2. If tags are not given, make them on your own.
3. Remember to always open and close all the tags.

As an example, for the tags ["foo", "bar", "baz"]:
1. String "<foo>
   <bar>
      <baz></baz>
   </bar>
</foo>" is a well-formatted instance of the schema. 
2. String "<foo>
   <bar>
   </foo>" is a badly-formatted instance.
3. String "<foo>
   <tag>
   </tag>
</foo>" is a badly-formatted instance.

Here are the output tags:
```
['movies', 'actor', 'film', 'name', 'genre']
```

```python
prompt = PromptTemplate(
    template="""{query}\n{format_instructions}""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
output = chain.invoke({"query": actor_query})
print(output)
```

```xml
<movies>
  <actor>
    <name>周星驰</name>
    <film>
      <name>美人鱼</name>
      <genre>喜剧, 奇幻</genre>
    </film>
    <film>
      <name>西游降魔篇</name>
      <genre>喜剧, 奇幻</genre>
    </film>
    <film>
      <name>长江七号</name>
      <genre>喜剧, 科幻</genre>
    </film>
    <film>
      <name>功夫</name>
      <genre>喜剧, 动作</genre>
    </film>
    <film>
      <name>少林足球</name>
      <genre>喜剧, 运动</genre>
    </film>
  </actor>
</movies>
```

<font style="color:rgb(28, 30, 33);">这个输出解析器还支持部分数据流的处理。以下是一个示例：</font>

```python
for s in chain.stream({"query": actor_query}):
    print(s)
```

```xml
{'movies': [{'actor': [{'name': '周星驰'}]}]}
{'movies': [{'actor': [{'film': [{'name': '美人鱼'}]}]}]}
{'movies': [{'actor': [{'film': [{'genre': '喜剧, 奇幻'}]}]}]}
{'movies': [{'actor': [{'film': [{'name': '西游·降魔篇'}]}]}]}
{'movies': [{'actor': [{'film': [{'genre': '喜剧, 奇幻'}]}]}]}
{'movies': [{'actor': [{'film': [{'name': '长江七号'}]}]}]}
{'movies': [{'actor': [{'film': [{'genre': '喜剧, 科幻'}]}]}]}
{'movies': [{'actor': [{'film': [{'name': '功夫'}]}]}]}
{'movies': [{'actor': [{'film': [{'genre': '喜剧, 动作'}]}]}]}
{'movies': [{'actor': [{'film': [{'name': '少林足球'}]}]}]}
{'movies': [{'actor': [{'film': [{'genre': '喜剧, 运动'}]}]}]}
{'movies': [{'actor': [{'film': [{'name': '喜剧之王'}]}]}]}
{'movies': [{'actor': [{'film': [{'genre': '喜剧, 剧情'}]}]}]}
{'movies': [{'actor': [{'film': [{'name': '大话西游之大圣娶亲'}]}]}]}
{'movies': [{'actor': [{'film': [{'genre': '喜剧, 奇幻'}]}]}]}
{'movies': [{'actor': [{'film': [{'name': '大话西游之月光宝盒'}]}]}]}
{'movies': [{'actor': [{'film': [{'genre': '喜剧, 奇幻'}]}]}]}
```



# 自定义 Tools, 调用 Tools集成内建 Tools
## <font style="color:rgb(28, 30, 33);">自定义 Tools</font>
<font style="color:rgb(28, 30, 33);">在构建代理时，您需要为其提供一个</font>`<font style="color:rgb(28, 30, 33);">Tool</font>`<font style="color:rgb(28, 30, 33);">列表，以便代理可以使用这些工具。除了实际调用的函数之外，</font>`<font style="color:rgb(28, 30, 33);">Tool</font>`<font style="color:rgb(28, 30, 33);">由几个组件组成：</font>

| <font style="color:rgb(28, 30, 33);">属性</font> | <font style="color:rgb(28, 30, 33);">类型</font> | <font style="color:rgb(28, 30, 33);">描述</font> |
| --- | --- | --- |
| <font style="color:rgb(28, 30, 33);">name</font> | <font style="color:rgb(28, 30, 33);">str</font> | <font style="color:rgb(28, 30, 33);">在提供给LLM或代理的工具集中必须是唯一的。</font> |
| <font style="color:rgb(28, 30, 33);">description</font> | <font style="color:rgb(28, 30, 33);">str</font> | <font style="color:rgb(28, 30, 33);">描述工具的功能。LLM或代理将使用此描述作为上下文。</font> |
| <font style="color:rgb(28, 30, 33);">args_schema</font> | <font style="color:rgb(28, 30, 33);">Pydantic BaseModel</font> | <font style="color:rgb(28, 30, 33);">可选但建议，可用于提供更多信息（例如，few-shot示例）或验证预期参数。</font> |
| <font style="color:rgb(28, 30, 33);">return_direct</font> | <font style="color:rgb(28, 30, 33);">boolean</font> | <font style="color:rgb(28, 30, 33);">仅对代理相关。当为True时，在调用给定工具后，代理将停止并将结果直接返回给用户。</font> |


<font style="color:rgb(28, 30, 33);">LangChain 提供了三种创建工具的方式：</font>

1. <font style="color:rgb(28, 30, 33);">使用</font>[<font style="color:rgb(28, 30, 33);">@tool装饰器</font>](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.tool.html#langchain_core.tools.tool)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">-- 定义自定义工具的最简单方式。</font>
2. <font style="color:rgb(28, 30, 33);">使用</font>[<font style="color:rgb(28, 30, 33);">StructuredTool.from_function</font>](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.StructuredTool.html#langchain_core.tools.StructuredTool.from_function)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">类方法 -- 这类似于</font>`<font style="color:rgb(28, 30, 33);">@tool</font>`<font style="color:rgb(28, 30, 33);">装饰器，但允许更多配置和同步和异步实现的规范。</font>
3. <font style="color:rgb(28, 30, 33);">通过子类化</font>[<font style="color:rgb(28, 30, 33);">BaseTool</font>](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.BaseTool.html)<font style="color:rgb(28, 30, 33);"> -- 这是最灵活的方法，它提供了最大程度的控制，但需要更多的工作量和代码。 </font>`<font style="color:rgb(28, 30, 33);">@tool</font>`<font style="color:rgb(28, 30, 33);">或</font>`<font style="color:rgb(28, 30, 33);">StructuredTool.from_function</font>`<font style="color:rgb(28, 30, 33);">类方法对于大多数用例应该足够了。 提示 如果工具具有精心选择的名称、描述和 JSON 模式，模型的性能会更好。 </font>

### <font style="color:rgb(28, 30, 33);">@tool 装饰器</font>
<font style="color:rgb(28, 30, 33);">这个</font>`<font style="color:rgb(28, 30, 33);">@tool</font>`<font style="color:rgb(28, 30, 33);">装饰器是定义自定义工具的最简单方式。该装饰器默认使用函数名称作为工具名称，但可以通过传递字符串作为第一个参数来覆盖。此外，装饰器将使用函数的文档字符串作为工具的描述 - 因此必须提供文档字符串。</font>

```python
#示例：tools_decorator.py
from langchain_core.tools import tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
# 让我们检查与该工具关联的一些属性。
print(multiply.name)
print(multiply.description)
print(multiply.args)
```

```plain
multiply
multiply(a: int, b: int) -> int - Multiply two numbers.
{'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
```

<font style="color:rgb(28, 30, 33);">或者创建一个</font>**<font style="color:rgb(28, 30, 33);">异步</font>**<font style="color:rgb(28, 30, 33);">实现，如下所示：</font>

```python
#示例：tools_async.py
from langchain_core.tools import tool
@tool
async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

<font style="color:rgb(28, 30, 33);">您还可以通过将它们传递给工具装饰器来自定义工具名称和 JSON 参数。</font>

```python
from pydantic import BaseModel, Field
class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")
@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
# 让我们检查与该工具关联的一些属性。
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.return_direct)
```

```plain
multiplication-tool
multiplication-tool(a: int, b: int) -> int - Multiply two numbers.
{'a': {'title': 'A', 'description': 'first number', 'type': 'integer'}, 'b': {'title': 'B', 'description': 'second number', 'type': 'integer'}}
True
```

### <font style="color:rgb(28, 30, 33);">StructuredTool</font>
`<font style="color:rgb(28, 30, 33);">StructuredTool.from_function</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">类方法提供了比</font>`<font style="color:rgb(28, 30, 33);">@tool</font>`<font style="color:rgb(28, 30, 33);">装饰器更多的可配置性，而无需太多额外的代码。</font>

```python
from langchain_core.tools import StructuredTool
import asyncio

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

async def main():
    calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)
    print(calculator.invoke({"a": 2, "b": 3}))
    print(await calculator.ainvoke({"a": 2, "b": 5}))

# 运行异步主函数
asyncio.run(main())
```

```plain
6
10
```

<font style="color:rgb(28, 30, 33);">可以配置自定义参数</font>

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import asyncio

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# 创建一个异步包装器函数
async def async_addition(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a + b
async def main():
    calculator = StructuredTool.from_function(
        func=multiply,
        name="Calculator",
        description="multiply numbers",
        args_schema=CalculatorInput,
        return_direct=True,
        #coroutine= async_addition
        # coroutine= ... <- 如果需要，也可以指定异步方法
    )
    print(calculator.invoke({"a": 2, "b": 3}))
    #print(await calculator.ainvoke({"a": 2, "b": 5}))
    print(calculator.name)
    print(calculator.description)
    print(calculator.args)


# 运行异步主函数
asyncio.run(main())
```

### <font style="color:rgb(28, 30, 33);">处理工具错误</font>
<font style="color:rgb(28, 30, 33);">如果您正在使用带有代理的工具，您可能需要一个错误处理策略，以便代理可以从错误中恢复并继续执行。 一个简单的策略是在工具内部抛出</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">ToolException</font>`<font style="color:rgb(28, 30, 33);">，并使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">handle_tool_error</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">指定一个错误处理程序。 当指定了错误处理程序时，异常将被捕获，错误处理程序将决定从工具返回哪个输出。 您可以将</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">handle_tool_error</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">设置为</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">True</font>`<font style="color:rgb(28, 30, 33);">、字符串值或函数。如果是函数，该函数应该以</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">ToolException</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">作为参数，并返回一个值。 请注意，仅仅抛出</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">ToolException</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">是不会生效的。您需要首先设置工具的</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">handle_tool_error</font>`<font style="color:rgb(28, 30, 33);">，因为其默认值是</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">False</font>`<font style="color:rgb(28, 30, 33);">。</font>

```python
from langchain_core.tools import ToolException
def get_weather(city: str) -> int:
    """获取给定城市的天气。"""
    raise ToolException(f"错误：没有名为{city}的城市。")
```

<font style="color:rgb(28, 30, 33);">下面是一个使用默认的</font>`<font style="color:rgb(28, 30, 33);">handle_tool_error=True</font>`<font style="color:rgb(28, 30, 33);">行为的示例。</font>

```python
#示例：tools_exception.py
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=True,
)
get_weather_tool.invoke({"city": "foobar"})
```

```plain
'错误：没有名为foobar的城市。'
```

<font style="color:rgb(28, 30, 33);">我们可以将</font>`<font style="color:rgb(28, 30, 33);">handle_tool_error</font>`<font style="color:rgb(28, 30, 33);">设置为一个始终返回的字符串。</font>

```python
#示例：tools_exception_handle.py
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error="没找到这个城市",
)
get_weather_tool.invoke({"city": "foobar"})
```

```plain
"没有这样的城市，但可能在那里的温度超过0K！"
```

<font style="color:rgb(28, 30, 33);">使用函数处理错误：</font>

```python
#示例：tools_exception_handle_error.py
def _handle_error(error: ToolException) -> str:
    return f"工具执行期间发生以下错误：`{error.args[0]}`"
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=_handle_error,
)
get_weather_tool.invoke({"city": "foobar"})
```

```plain
'工具执行期间发生以下错误：`错误：没有名为foobar的城市。`'
```



## 调用 Tools集成内建 Tools
### 工具
<font style="color:rgb(28, 30, 33);">工具是代理、链或聊天模型/LLM用来与世界交互的接口。 一个工具由以下组件组成：</font>

1. <font style="color:rgb(28, 30, 33);">工具的名称</font>
2. <font style="color:rgb(28, 30, 33);">工具的功能描述</font>
3. <font style="color:rgb(28, 30, 33);">工具输入的JSON模式</font>
4. <font style="color:rgb(28, 30, 33);">要调用的函数</font>
5. <font style="color:rgb(28, 30, 33);">工具的结果是否应直接返回给用户（仅对代理相关） 名称、描述和JSON模式作为上下文提供给LLM，允许LLM适当地确定如何使用工具。 给定一组可用工具和提示，LLM可以请求调用一个或多个工具，并提供适当的参数。 通常，在设计供聊天模型或LLM使用的工具时，重要的是要牢记以下几点：</font>
+ <font style="color:rgb(28, 30, 33);">经过微调以进行工具调用的聊天模型将比未经微调的模型更擅长进行工具调用。</font>
+ <font style="color:rgb(28, 30, 33);">未经微调的模型可能根本无法使用工具，特别是如果工具复杂或需要多次调用工具。</font>
+ <font style="color:rgb(28, 30, 33);">如果工具具有精心选择的名称、描述和JSON模式，则模型的性能将更好。</font>
+ <font style="color:rgb(28, 30, 33);">简单的工具通常比更复杂的工具更容易让模型使用。</font>

<font style="color:rgb(28, 30, 33);">LangChain 拥有大量第三方工具。请访问</font>[<font style="color:rgb(28, 30, 33);">工具集成</font>](http://www.aidoczh.com/langchain/v0.2/docs/integrations/tools/)<font style="color:rgb(28, 30, 33);">查看可用工具列表。</font>

[https://python.langchain.com/v0.2/docs/integrations/tools/](https://python.langchain.com/v0.2/docs/integrations/tools/)

在使用第三方工具时，请确保您了解工具的工作原理、权限情况。请阅读其文档，并检查是否需要从安全角度考虑任何事项。请查看[安全](https://python.langchain.com/v0.1/docs/security/)指南获取更多信息。

<font style="color:rgb(28, 30, 33);">让我们尝试一下</font>[<font style="color:rgb(28, 30, 33);">维基百科集成</font>](http://www.aidoczh.com/langchain/v0.2/docs/integrations/tools/wikipedia/)<font style="color:rgb(28, 30, 33);">。</font>

```python
!pip install -qU wikipedia
```

```python
#示例：tools_wikipedia.py
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
print(tool.invoke({"query": "langchain"}))
```

```plain
Page: LangChain
Summary: LangChain is a framework designed to simplify the creation of applications
```

<font style="color:rgb(28, 30, 33);">该工具具有以下默认关联项：</font>

```python
print(f"Name: {tool.name}")
print(f"Description: {tool.description}")
print(f"args schema: {tool.args}")
print(f"returns directly?: {tool.return_direct}")
```

```plain
Name: wikipedia
Description: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
args schema: {'query': {'title': 'Query', 'description': 'query to look up on wikipedia', 'type': 'string'}}
returns directly?: False
```

### 自定义默认工具
<font style="color:rgb(28, 30, 33);">我们还可以修改内置工具的名称、描述和参数的 JSON 模式。</font>

<font style="color:rgb(28, 30, 33);">在定义参数的 JSON 模式时，重要的是输入保持与函数相同，因此您不应更改它。但您可以轻松为每个输入定义自定义描述。</font>

```python
#示例：tools_custom.py
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field
class WikiInputs(BaseModel):
    """维基百科工具的输入。"""
    query: str = Field(
        description="query to look up in Wikipedia, should be 3 or less words"
    )
tool = WikipediaQueryRun(
    name="wiki-tool",
    description="look up things in wikipedia",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=True,
)
print(tool.run("langchain"))
```

```plain
Page: LangChain
Summary: LangChain is a framework designed to simplify the creation of applications 
```

```python
print(f"Name: {tool.name}")
print(f"Description: {tool.description}")
print(f"args schema: {tool.args}")
print(f"returns directly?: {tool.return_direct}")
```

```plain
Name: wiki-tool
Description: look up things in wikipedia
args schema: {'query': {'title': 'Query', 'description': 'query to look up in Wikipedia, should be 3 or less words', 'type': 'string'}}
returns directly?: True
```

### 如何使用内置工具包
<font style="color:rgb(28, 30, 33);">工具包是一组旨在一起使用以执行特定任务的工具。它们具有便捷的加载方法。</font>

<font style="color:rgb(28, 30, 33);">要获取可用的现成工具包完整列表，请访问</font>[<font style="color:rgb(28, 30, 33);">集成</font>](http://www.aidoczh.com/langchain/v0.2/docs/integrations/toolkits/)<font style="color:rgb(28, 30, 33);">。</font>

<font style="color:rgb(28, 30, 33);">所有工具包都公开了一个</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">get_tools</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">方法，该方法返回一个工具列表。</font>

<font style="color:rgb(28, 30, 33);">通常您应该这样使用它们：</font>

```python
# 初始化一个工具包
toolkit = ExampleTookit(...)
# 获取工具列表
tools = toolkit.get_tools()
```

---

例如使用：<font style="color:#080808;background-color:#ffffff;">SQLDatabase toolkit 读取langchain.db数据库表结构</font>

```python
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType

db = SQLDatabase.from_uri("sqlite:///langchain.db")
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
print(toolkit.get_tools())

agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4"),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
)
# %%
agent_executor.invoke("Describe the full_llm_cache table")

```

执行 print(toolkit.get_tools())

```python
[QuerySQLDataBaseTool(description="Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.", db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x000001D15C9749B0>), InfoSQLDatabaseTool(description='Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3', db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x000001D15C9749B0>), ListSQLDatabaseTool(db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x000001D15C9749B0>), QuerySQLCheckerTool(description='Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!', db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x000001D15C9749B0>, llm=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x000001D17B4453D0>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x000001D17B446D80>, temperature=0.0, openai_api_key=SecretStr('**********'), openai_proxy=''), llm_chain=LLMChain(prompt=PromptTemplate(input_variables=['dialect', 'query'], template='\n{query}\nDouble check the {dialect} query above for common mistakes, including:\n- Using NOT IN with NULL values\n- Using UNION when UNION ALL should have been used\n- Using BETWEEN for exclusive ranges\n- Data type mismatch in predicates\n- Properly quoting identifiers\n- Using the correct number of arguments for functions\n- Casting to the correct data type\n- Using the proper columns for joins\n\nIf there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.\n\nOutput the final SQL query only.\n\nSQL Query: '), llm=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x000001D17B4453D0>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x000001D17B446D80>, temperature=0.0, openai_api_key=SecretStr('**********'), openai_proxy='')))]
```

执行：agent_executor.invoke("Describe the full_llm_cache table")

```python
> Entering new SQL Agent Executor chain...

Invoking: `sql_db_schema` with `{'table_names': 'full_llm_cache'}`

CREATE TABLE full_llm_cache (
	prompt VARCHAR NOT NULL, 
	llm VARCHAR NOT NULL, 
	idx INTEGER NOT NULL, 
	response VARCHAR, 
	PRIMARY KEY (prompt, llm, idx)
)

/*
3 rows from full_llm_cache table:
prompt	llm	idx	response
[{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs	{"id": ["langchain", "chat_models", "openai", "ChatOpenAI"], "kwargs": {"max_retries": 2, "model_nam	0	{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "output", "ChatGeneration"], "kwargs"
*/The `full_llm_cache` table has the following structure:

- `prompt`: A VARCHAR field that is part of the primary key. It cannot be NULL.
- `llm`: A VARCHAR field that is also part of the primary key. It cannot be NULL.
- `idx`: An INTEGER field that is part of the primary key as well. It cannot be NULL.
- `response`: A VARCHAR field that can contain NULL values.

Here are some sample rows from the `full_llm_cache` table:

| prompt | llm | idx | response |
|--------|-----|-----|----------|
| [{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs | {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"], "kwargs": {"max_retries": 2, "model_nam | 0 | {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "output", "ChatGeneration"], "kwargs" |

> Finished chain.
```

# 创建和运行 Agent
<font style="color:rgb(28, 30, 33);">单独来说，语言模型无法采取行动 - 它们只能输出文本。</font>

<font style="color:rgb(28, 30, 33);">LangChain 的一个重要用例是创建</font>**<font style="color:rgb(28, 30, 33);">代理</font>**<font style="color:rgb(28, 30, 33);">。</font>

<font style="color:rgb(28, 30, 33);">代理是使用 LLM 作为推理引擎的系统，用于确定应采取哪些行动以及这些行动的输入应该是什么。</font>

<font style="color:rgb(28, 30, 33);">然后可以将这些行动的结果反馈给代理，并确定是否需要更多行动，或者是否可以结束。</font>

<font style="color:rgb(28, 30, 33);">在本次课程中，我们将构建一个可以与多种不同工具进行交互的代理：一个是本地数据库，另一个是搜索引擎。您将能够向该代理提问，观察它调用工具，并与它进行对话。</font>

下面将介绍使用 LangChain 代理进行构建。LangChain 代理适合入门，但在一定程度之后，我们可能希望拥有它们无法提供的灵活性和控制性。要使用更高级的代理，我们建议查看LangGraph

## <font style="color:rgb(28, 30, 33);">概念</font>
<font style="color:rgb(28, 30, 33);">我们将涵盖的概念包括：</font>

+ <font style="color:rgb(28, 30, 33);">使用</font>[<font style="color:rgb(28, 30, 33);">语言模型</font>](http://www.aidoczh.com/langchain/v0.2/docs/concepts/#chat-models)<font style="color:rgb(28, 30, 33);">，特别是它们的工具调用能力</font>
+ <font style="color:rgb(28, 30, 33);">创建</font>[<font style="color:rgb(28, 30, 33);">检索器</font>](http://www.aidoczh.com/langchain/v0.2/docs/concepts/#retrievers)<font style="color:rgb(28, 30, 33);">以向我们的代理公开特定信息</font>
+ <font style="color:rgb(28, 30, 33);">使用搜索</font>[<font style="color:rgb(28, 30, 33);">工具</font>](http://www.aidoczh.com/langchain/v0.2/docs/concepts/#tools)<font style="color:rgb(28, 30, 33);">在线查找信息</font>
+ [<font style="color:rgb(28, 30, 33);">聊天历史</font>](http://www.aidoczh.com/langchain/v0.2/docs/concepts/#chat-history)<font style="color:rgb(28, 30, 33);">，允许聊天机器人“记住”过去的交互，并在回答后续问题时考虑它们。</font>
+ <font style="color:rgb(28, 30, 33);">使用</font>[<font style="color:rgb(28, 30, 33);">LangSmith</font>](http://www.aidoczh.com/langchain/v0.2/docs/concepts/#langsmith)<font style="color:rgb(28, 30, 33);">调试和跟踪您的应用程序</font>

## <font style="color:rgb(28, 30, 33);">安装</font>
<font style="color:rgb(28, 30, 33);">要安装 LangChain，请运行：</font>

```bash
pip install langchain
```

### <font style="color:rgb(28, 30, 33);">LangSmith</font>
<font style="color:rgb(28, 30, 33);">使用 LangChain 构建的许多应用程序将包含多个步骤，其中会多次调用 LLM。</font>

<font style="color:rgb(28, 30, 33);">随着这些应用程序变得越来越复杂，能够检查链或代理内部发生了什么变得至关重要。</font>

<font style="color:rgb(28, 30, 33);">这样做的最佳方式是使用</font>[<font style="color:rgb(28, 30, 33);">LangSmith</font>](https://smith.langchain.com/)<font style="color:rgb(28, 30, 33);">。</font>

<font style="color:rgb(28, 30, 33);">在上面的链接注册后，请确保设置您的环境变量以开始记录跟踪：</font>

```shell
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```

## <font style="color:rgb(28, 30, 33);">定义工具</font>
<font style="color:rgb(28, 30, 33);">我们首先需要创建我们想要使用的工具。我们将使用两个工具：</font>[<font style="color:rgb(28, 30, 33);">Tavily</font>](http://www.aidoczh.com/langchain/v0.2/docs/integrations/tools/tavily_search/)<font style="color:rgb(28, 30, 33);">（用于在线搜索），然后是我们将创建的本地索引上的检索器。</font>

### [<font style="color:rgb(28, 30, 33);">Tavily</font>](http://www.aidoczh.com/langchain/v0.2/docs/integrations/tools/tavily_search/)
<font style="color:rgb(28, 30, 33);">LangChain 中有一个内置工具，可以轻松使用 Tavily 搜索引擎作为工具。</font>

<font style="color:rgb(28, 30, 33);">请注意，这需要一个 API 密钥 - 他们有一个免费的层级，但如果您没有或不想创建一个，您可以忽略这一步。</font>

<font style="color:rgb(28, 30, 33);">创建 API 密钥后，您需要将其导出为：</font>

```bash
export TAVILY_API_KEY="..."
```

```python
from langchain_community.tools.tavily_search import TavilySearchResults
```

```python
search = TavilySearchResults(max_results=2)
```

```python
print(search.invoke("今天上海天气怎么样"))
```

```python
[{'url': 'http://sh.cma.gov.cn/sh/tqyb/jrtq/', 'content': '上海今天气温度30℃～38℃，偏南风风力4-5级，有多云和雷阵雨的可能。生活气象指数显示，气温高，人体感觉不舒适，不适宜户外活动。'}]
```

### <font style="color:rgb(28, 30, 33);">Retriever</font>
<font style="color:rgb(28, 30, 33);">Retriever 是 langchain 库中的一个模块，用于检索工具。检索工具的主要用途是从大型文本集合或知识库中找到相关信息。它们通常用于问答系统、对话代理和其他需要从大量文本数据中提取信息的应用程序。</font>

<font style="color:rgb(28, 30, 33);">我们还将在自己的一些数据上创建一个Retriever。有关每个步骤的更深入解释，请参阅此教程。</font>

```python
#示例：tools_retriever.py
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    #chunk_size 参数在 RecursiveCharacterTextSplitter 中用于指定每个文档块的最大字符数。它的作用主要有以下几个方面：
    #chunk_overlap 参数用于指定每个文档块之间的重叠字符数。这意味着，当文档被拆分成较小的块时，每个块的末尾部分会与下一个块的开头部分有一定数量的重叠字符。
    #第一个块包含字符 1 到 1000。第二个块包含字符 801 到 1800。第三个块包含字符 1601 到 2600。
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
```

```python
retriever.invoke("猫的特征")[0]
```

```python
page_content='聽覺[编辑]
貓每隻耳各有32條獨立的肌肉控制耳殼轉動，因此雙耳可單獨朝向不同的音源轉動，使其向獵物移動時仍能對周遭其他音源保持直接接觸。[50] 除了蘇格蘭折耳貓這類基因突變的貓以外，貓極少有狗常見的「垂耳」，多數的貓耳向上直立。當貓忿怒或受驚時，耳朵會貼向後方，並發出咆哮與「嘶」聲。
貓與人類對低頻聲音靈敏度相若。人類中只有極少數的調音師能聽到20 kHz以上的高頻聲音（8.4度的八度音），貓則可達64kHz（10度的八度音），比人類要高1.6個八度音，甚至比狗要高1個八度；但是貓辨別音差須開最少5度，比起人類辨別音差須開最少0.5度來得粗疏。[51][47]

嗅覺[编辑]
家貓的嗅覺較人類靈敏14倍。[52]貓的鼻腔內有2億個嗅覺受器，數量甚至超過某些品種的狗（狗嗅覺細胞約1.25億～2.2億）。

味覺[编辑]
貓早期演化時由於基因突變，失去了甜的味覺，[53]但貓不光能感知酸、苦、鹹味，选择适合自己口味的食物，还能尝出水的味道，这一点是其他动物所不及的。不过总括来说猫的味觉不算完善，相比一般人類平均有9000個味蕾，貓一般平均僅有473個味蕾且不喜好低於室溫之食物。故此，貓辨認食物乃憑嗅覺多於味覺。[47]

觸覺[编辑]
貓在磨蹭時身上會散發出特別的費洛蒙，當這些獨有的費洛蒙留下時，目的就是在宣誓主權，提醒其它貓這是我的，其實這種行為算是一種標記地盤的象徵，會讓牠們有感到安心及安全感。

被毛[编辑]
主条目：貓的毛色遺傳和顏色
長度[编辑]
貓主要可以依據被毛長度分為長毛貓，短毛貓和無毛貓。' metadata={'source': 'https://zh.wikipedia.org/wiki/%E7%8C%AB', 'title': '猫 - 维基百科，自由的百科全书', 'language': 'zh'}

```

<font style="color:rgb(28, 30, 33);">现在，我们已经填充了我们将要进行Retriever的索引，我们可以轻松地将其转换为一个工具（代理程序正确使用所需的格式）。</font>

```python
from langchain.tools.retriever import create_retriever_tool
```

```python
retriever_tool = create_retriever_tool(
    retriever,
    "wiki_search",
    "搜索维基百科",
)
```

### <font style="color:rgb(28, 30, 33);">工具</font>
<font style="color:rgb(28, 30, 33);">既然我们都创建好了，我们可以创建一个工具列表，以便在下游使用。</font>

```python
tools = [search, retriever_tool]
```

## <font style="color:rgb(28, 30, 33);">使用语言模型</font>
<font style="color:rgb(28, 30, 33);">接下来，让我们学习如何使用语言模型来调用工具。LangChain支持许多可以互换使用的不同语言模型 - 选择您想要使用的语言模型！</font>

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")
```

<font style="color:rgb(28, 30, 33);">您可以通过传入消息列表来调用语言模型。默认情况下，响应是一个</font>`<font style="color:rgb(28, 30, 33);">content</font>`<font style="color:rgb(28, 30, 33);">字符串。</font>

```python
from langchain_core.messages import HumanMessage
response = model.invoke([HumanMessage(content="hi!")])
response.content
```

```plain
'Hello! How can I assist you today?'
```

<font style="color:rgb(28, 30, 33);">现在，我们可以看看如何使这个模型能够调用工具。为了使其具备这种能力，我们使用</font>`<font style="color:rgb(28, 30, 33);">.bind_tools</font>`<font style="color:rgb(28, 30, 33);">来让语言模型了解这些工具。</font>

```python
model_with_tools = model.bind_tools(tools)
```

<font style="color:rgb(28, 30, 33);">现在我们可以调用模型了。让我们首先用一个普通的消息来调用它，看看它的响应。我们可以查看</font>`<font style="color:rgb(28, 30, 33);">content</font>`<font style="color:rgb(28, 30, 33);">字段和</font>`<font style="color:rgb(28, 30, 33);">tool_calls</font>`<font style="color:rgb(28, 30, 33);">字段。</font>

```python
response = model_with_tools.invoke([HumanMessage(content="你好")])
print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
```

```plain
ContentString: 你好！有什么可以帮助你的吗？
ToolCalls: []
```

<font style="color:rgb(28, 30, 33);">现在，让我们尝试使用一些期望调用工具的输入来调用它。</font>

```python
response = model_with_tools.invoke([HumanMessage(content="今天上海天气怎么样")])
print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
```

```plain
ContentString: 
ToolCalls: [{'name': 'tavily_search_results_json', 'args': {'query': '今天上海天气'}, 'id': 'call_EOxYscVIVjttlbztWoR1CvTm', 'type': 'tool_call'}]
```

<font style="color:rgb(28, 30, 33);">我们可以看到现在没有内容，但有一个工具调用！它要求我们调用Tavily Search工具。</font>

<font style="color:rgb(28, 30, 33);">这并不是在调用该工具 - 它只是告诉我们要调用。为了实际调用它，我们将创建我们的代理程序。</font>

## <font style="color:rgb(28, 30, 33);">创建代理程序</font>
<font style="color:rgb(28, 30, 33);">既然我们已经定义了工具和LLM，我们可以创建代理程序。我们将使用一个工具调用代理程序 - 有关此类代理程序以及其他选项的更多信息，请参阅</font>[<font style="color:rgb(28, 30, 33);">此指南</font>](http://www.aidoczh.com/langchain/v0.2/docs/concepts/#agent_types/)<font style="color:rgb(28, 30, 33);">。</font>

<font style="color:rgb(28, 30, 33);">我们可以首先选择要用来指导代理程序的提示。</font>

<font style="color:rgb(28, 30, 33);">如果您想查看此提示的内容并访问LangSmith，您可以转到：</font>

[<font style="color:rgb(28, 30, 33);">https://smith.langchain.com/hub/hwchase17/openai-functions-agent</font>](https://smith.langchain.com/hub/hwchase17/openai-functions-agent)

```python
#示例：agent_tools_create.py
from langchain import hub
# 获取要使用的提示 - 您可以修改这个！
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages
```

```python
[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')), MessagesPlaceholder(variable_name='chat_history', optional=True), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')), MessagesPlaceholder(variable_name='agent_scratchpad')]

```

<font style="color:rgb(28, 30, 33);">现在，我们可以使用LLM、提示和工具初始化代理。代理负责接收输入并决定采取什么行动。关键的是，代理不执行这些操作 - 这是由AgentExecutor（下一步）完成的。</font>

<font style="color:rgb(28, 30, 33);">请注意，我们传递的是</font>`<font style="color:rgb(28, 30, 33);">model</font>`<font style="color:rgb(28, 30, 33);">，而不是</font>`<font style="color:rgb(28, 30, 33);">model_with_tools</font>`<font style="color:rgb(28, 30, 33);">。这是因为</font>`<font style="color:rgb(28, 30, 33);">create_tool_calling_agent</font>`<font style="color:rgb(28, 30, 33);">会在幕后调用</font>`<font style="color:rgb(28, 30, 33);">.bind_tools</font>`<font style="color:rgb(28, 30, 33);">。</font>

```python
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(model, tools, prompt)
```

<font style="color:rgb(28, 30, 33);">最后，我们将代理（大脑）与AgentExecutor中的工具结合起来（AgentExecutor将重复调用代理并执行工具）。</font>

```python
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

## <font style="color:rgb(28, 30, 33);">运行代理</font>
<font style="color:rgb(28, 30, 33);">现在，我们可以在几个查询上运行代理！请注意，目前这些都是</font>**<font style="color:rgb(28, 30, 33);">无状态</font>**<font style="color:rgb(28, 30, 33);">查询（它不会记住先前的交互）。</font>

<font style="color:rgb(28, 30, 33);">首先，让我们看看当不需要调用工具时它如何回应：</font>

```python
#示例：agent_tools_run.py
print(agent_executor.invoke({"input": "你好"}))
```

```plain
{'input': '你好', 'output': '你好！有什么我可以帮助你的吗？'}
```

<font style="color:rgb(28, 30, 33);">为了确切了解底层发生了什么（并确保它没有调用工具），我们可以查看</font>[<font style="color:rgb(28, 30, 33);">LangSmith跟踪</font>](https://smith.langchain.com/public/8441812b-94ce-4832-93ec-e1114214553a/r)

<font style="color:rgb(28, 30, 33);">现在让我们尝试一个应该调用检索器的示例：</font>

```python
print(agent_executor.invoke({"input": "猫的特征"}))
```

```plain
{'input': '猫的特征', 'output': '猫有许多显著的特征，包括以下几点：\n\n**听觉**：猫每只耳朵都有32条独立的肌肉控制耳壳转动。它们可以单独朝向不同的音源转动，使得在向猎物移动时仍能对周围其他音源保持直接接触。猫的听觉比人类和狗更敏锐，能听到更高的频率。\n\n**嗅觉**：家猫的嗅觉比人类灵敏14倍，鼻腔内有2亿个嗅觉受器，数量甚至超过某些品种的狗。\n\n**味觉**：由于早期的演化，猫失去了甜的味觉，但它们能感知酸、苦、咸味，并选择适合自己口味的食物。不过总的来说，猫的味觉并不算完善，相比一般人类平均有9000个味蕾，猫一般平均只有473个味蕾，且不喜欢低于室温的食物。\n\n**触觉**：猫在磨蹭时身上会散发出特别的费洛蒙，当这些独有的费洛蒙留下时，目的就是在宣誓主权，提醒其他猫这是我的。\n\n**被毛**：猫的被毛长度可以根据成为长毛猫，短毛猫和无毛猫。\n\n**视觉**：猫的夜视能力和追踪视觉活动物体相当出色，夜视能力是人类的六倍。猫的眼睛具有微光观察能力，即使只有微弱的月光也能分辨物体。\n\n**骨骼**：猫的骨骼共有230块，其中脊椎骨占了30块。\n\n**爪子**：猫的爪子尖锐且具有伸缩作用，能向外张开或向内收缩藏起来。\n\n**地域性攻击**：猫是很有地域性的动物，会在自己的地盤留下气味，利用下巴、耳朵及尾部的皮脂腺磨蹭物体以标记领地。\n\n**与狗的关系**：一般认为猫和狗互相厌恶，但经过训练和适应，猫和狗可能理解同一种“语言”并和睦相处。\n\n以上是猫的一些主要特征，但每只猫都有其个性和独特之处。'}
```

<font style="color:rgb(28, 30, 33);">让我们查看</font>[<font style="color:rgb(28, 30, 33);">LangSmith跟踪</font>](https://smith.langchain.com/public/762153f6-14d4-4c98-8659-82650f860c62/r)<font style="color:rgb(28, 30, 33);">以确保它实际上在调用该工具。</font>

<font style="color:rgb(28, 30, 33);">现在让我们尝试一个需要调用搜索工具的示例：</font>

```python
print(agent_executor.invoke({"input": "今天上海天气怎么样"}))
```

```plain
{'input': '今天上海天气怎么样', 'output': '很抱歉，我无法获取实时的天气信息。你可以访问上海市气象局的网站[这里](http://sh.cma.gov.cn/sh/tqyb/jrtq/)来查询今天的气象状况、预警信息、生活指数等。同时，该网站还提供了未来几天的天气趋势和空气质量状况，以及气象科普、气象视频、气候变化等其他相关服务和信息。'}
```

<font style="color:rgb(28, 30, 33);">我们可以查看</font>[<font style="color:rgb(28, 30, 33);">LangSmith跟踪</font>](https://smith.langchain.com/public/36df5b1a-9a0b-4185-bae2-964e1d53c665/r)<font style="color:rgb(28, 30, 33);">以确保它有效地调用了搜索工具。</font>

## <font style="color:rgb(28, 30, 33);">添加记忆</font>
<font style="color:rgb(28, 30, 33);">如前所述，此代理是无状态的。这意味着它不会记住先前的交互。要给它记忆，我们需要传递先前的</font>`<font style="color:rgb(28, 30, 33);">chat_history</font>`<font style="color:rgb(28, 30, 33);">。注意：由于我们使用的提示，它需要被称为</font>`<font style="color:rgb(28, 30, 33);">chat_history</font>`<font style="color:rgb(28, 30, 33);">。如果我们使用不同的提示，我们可以更改变量名</font>

```python
#示例：agent_tools_memory.py

# 这里我们为chat_history传入了一个空消息列表，因为这是对话中的第一条消息
agent_executor.invoke({"input": "hi! my name is bob", "chat_history": []})
```

```plain
{'input': '你好，我的名字是Cyber', 'chat_history': [], 'output': '你好，Cyber，很高兴见到你！有什么我可以帮助你的吗？'}
```

```python
from langchain_core.messages import AIMessage, HumanMessage
```

```python
agent_executor.invoke(
    {
        "chat_history": [
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="你好Bob！我今天能帮你什么？"),
        ],
        "input": "我的名字是什么?",
    }
)
```

<font style="color:rgb(28, 30, 33);">如果我们想要自动跟踪这些消息，我们可以将其包装在一个RunnableWithMessageHistory中</font>

```python
#示例：agent_tools_memory_store.py
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:

    if session_id not in store:

        store[session_id] = ChatMessageHistory()

    return store[session_id]
```

<font style="color:rgb(28, 30, 33);"></font>

<font style="color:rgb(28, 30, 33);">因为我们有多个输入，我们需要指定两个事项：</font>

+ `<font style="color:rgb(28, 30, 33);">input_messages_key</font>`<font style="color:rgb(28, 30, 33);">：用于将输入添加到对话历史记录中的键。</font>
+ `<font style="color:rgb(28, 30, 33);">history_messages_key</font>`<font style="color:rgb(28, 30, 33);">：用于将加载的消息添加到其中的键。</font>

```python
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
```

```python
response = agent_with_chat_history.invoke(
    {"input": "Hi，我的名字是Cyber"},
    config={"configurable": {"session_id": "123"}},
)
```

```plain
{'input': 'Hi，我的名字是Cyber', 'chat_history': [], 'output': '你好，Cyber！很高兴认识你。有什么我可以帮助你的吗？'}
```

```python
response = agent_with_chat_history.invoke(
    {"input": "我叫什么名字?"},
    config={"configurable": {"session_id": "123"}},
)
```

```plain
{'input': '我叫什么名字?', 'chat_history': [HumanMessage(content='Hi，我的名字是Cyber'), AIMessage(content='你好，Cyber！很高兴认识你。有什么我可以帮助你的吗？')], 'output': '你的名字是Cyber。'}

```

<font style="color:rgb(28, 30, 33);">LangSmith示例跟踪：</font>[https://smith.langchain.com/public/98c8d162-60ae-4493-aa9f-992d87bd0429/r](https://smith.langchain.com/public/98c8d162-60ae-4493-aa9f-992d87bd0429/r)



