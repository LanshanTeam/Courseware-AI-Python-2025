## class08 Fastapi框架与langchain/langgraph框架

### 前言

估计大家都听了很长时间模型底层非常头大，这节课相对比较简单，涉及一个agent开发框架和一个web框架，足够你尝试去做点自己想实现的东西，langchain 1.0版本的更新比较大，可能在课件中也有不足的地方，欢迎补充交流，本次更类似于经验分享，langchain官方文档很难读，还是希望你们不用跟我一样坐牢，我尽量在这节课把零散的东西理清楚，有了一定基础理解你们再去读文档就不会跟我开始一样一头雾水，先来看langchain：

### langchain/langgraph框架

#### 框架1.0版本改动简单总结

LangChain 1.0 版本做了很大的破坏性更新，更新的功能支持会在下面进行简单介绍，官方做了很多核心API重构、包结构调整,很多老写法被取代，核心 `langchain` 包的范围缩小，旧的遗留功能迁移到单独的包（如 `langchain-classic`）

直接来看看在原0.2.x/0.3.x版本langchain下你可能见过的quickstart：

```python
# 初始化大语言模型
llm = OpenAI(model_name="gpt-3.5-turbo")

# 定义 prompt 模板
prompt = PromptTemplate(
    input_variables=["topic"],
    template="用简洁的中文解释什么是{topic}"
)

# 构建链
chain = LLMChain(llm=llm, prompt=prompt)

# 调用
output = chain.invoke({"topic": "LangChain"})
print(output.content)
```

你也可能见过这样的链式构建：

```python
chain = prompt | model | parser
```

这样的写法在langchain 1.x版本下正式被舍弃，框架提供了`create_agent()`方法来标准化的构建agent：

```python
agent = create_agent(
    model=model,
    tools=[tools],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Tokyo"}]}
)
```

1.0版本之前langchain提供的“链”的逻辑被弱化或者迁移到了`langchain-classic`,新的 Agent 架构是基于**LangGraph 图执行引擎**来构建,这样写更加标准化，更加可控，可扩展性更高，也是为什么推荐大家在1.0版本基础上学习而不是学习旧版本的原因

所以当前langchain/langgraph之间的界限看起来比较模糊，langgraph原本作为langchain的子系统，在当前是主流的agent开发框架，我们来看看`create_agent()`究竟得到了什么：

`create_agent(...)` 返回的是一个 **“Agent Graph Runtime” 实例**，它封装了：

- 模型调用
- 工具列表
- 中间件逻辑（middleware）
- Agent 内部循环（何时调用工具、何时结束、如何返回结果）

简单来说就是：你拿到的是一个封装好的 LangGraph 执行图

对于很多的简单中小任务，0个人想关心内部的图结构，这个图内部已经固定好了模型节点、工具执行节点以及循环与终止条件，开发者**不需要显式定义节点和边**，只需提供输入、模型、工具和中间件即可

那么在学习langgraph之前把上面langchain框架下构建简单agent的完整代码丢在这里：

```python
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 工具函数，模拟功能
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
# 这里我的key是设置在项目下.env所以需要load，你也可以在下面初始化直接填
load_dotenv()
# 模型选择建议大家单独初始化，方便进行更换
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.9,
    max_tokens=100,
    timeout=30,
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = create_agent(
    model=model, # 传入模型
    tools=[get_weather], # 注册工具
    system_prompt="You are a helpful assistant", # 系统提示词
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Tokyo"}]}
)

print(result)
```

运行你会发现模型输出了一大串，这是正常的，模型的返回不像你直接使用AI工具时跟你对话，会有很多参数，但是你肯定可以在其中找到你想看到的The weather in Tokyo is sunny. 如何让返回的东西符合你的预期，下面关于结构化输出的部分会提及

#### langgraph简单使用

在上面的部分，没提前了解过框架的同学可能会比较迷，看了半天只跑了个demo，接下来就从langgraph最基本从点到边构建图流来学习

##### StateGraph

stategraph是构建状态图的核心类，依赖开发框架，你只需要关心3个要点就能实现功能：

**状态（State）**：状态是一条数据在图里流动时的载体，可以是 `TypedDict` 定义的结构，每个节点可以读写 state 的部分字段

**节点（Node）**：每个节点是一个 Python 函数（或者可调用对象）

节点接受一个 `state` 对象（通常是 `TypedDict` 或 dict）

节点返回新的 `state`，供后续节点使用

**边（Edge）**：描述节点之间的流向

支持 **普通边** 和 **条件边**（conditional edge）

—————————–我是分割线—————————

如果你用过coze，dify这样的低代码平台，对这套流程肯定很熟悉了，连线工程师这一块，其实用langgraph思想上也差不多，但是我们能做的更多，先不提中间件，中断，人机协同，多agent之类的，我们先来看看最基本的使用，比如我们想实现一个天气助手agent：

首先要确定agent在运行时每个节点需要掌握哪些信息，比如在这里我们需要的就是问题，天气，答案：

```python
class State(TypedDict):
    question: str
    weather: Optional[str]
    answer: Optional[str]
```

那么agent期望要做的就是与用户对话，如果用户需要查天气，那么调用天气工具：

我们先来做一个判断节点，来看看用户要不要查天气：

```python
def llm_node(state: State):
    """这个节点决定是否调用工具"""
    question = state["question"]

    messages = [
        {"role": "system", "content": "你是一个判断是否需要查天气的助手。"},
        {"role": "user", "content": f"用户的问题是：{question}\n是否需要查天气？回答 yes 或 no。"}
    ]

    msg = model.invoke(messages)
    text = msg.content.strip().lower() 

    if "yes" in text:
        # LLM 需要天气信息 → 交给工具节点
        return {"weather": None, "answer": None}
    else:
        # 直接回答
        reply = model.invoke(question)
        return {"answer": reply.content}
```

节点可以是一个单纯的方法，传入参数是graph定义好的state，节点的工作是基于state判断，执行节点的内部逻辑并更新state，你只需要把精力放在对state的更新和节点的内部逻辑上

那下一步当然是做一个查天气的节点：

```python
def weather_node(state: State):
    """执行天气工具并返回结果"""
    question = state["question"]
    # 假装解析 —— 实际可用正则
    city = "东京"
    date = "2025-11-26"
    # get_weather是一个自定义的工具，放在下面工具的部分讲
    answer = get_weather.invoke({
        "city": city,
        "date": date
    })
    # answer 是工具部分定义的一个 OutPutTable 对象，手动转 dict
    data = answer.model_dump()
	# 你暂时不需要在意这个代码块里用到的东西，只需要知道，在这个节点里，我们调用天气工具查到了天气，并且把state的weather字段更新了
    weather = (
        f"{data['city']} {data['date']} 天气：{data['weather']}，温度：{data['temperature']}"
    )
    return {"weather": weather}
```

接下来做一个最终用来回答的节点，我们的第一张图就算基本完成：

```python
def final_answer_node(state: State):
    """组合最终回答的节点"""
    weather = state["weather"]
    question = state["question"]

    if weather:
        answer = f"你问的是：{question}\n查询到的天气为：{weather}"
    else:
        answer = state["answer"]

    return {"answer": answer}
```

整理一下当前的逻辑，我们的图应该长这样：

开始节点 ————（需要查天气）——-> 查天气节点 <——- 天气工具

   ↓			    ↓

   |———–（不需要查天气）————> 最终回答节点 —————> 回答

接下来就是简单的连线环节：

```python
# 添加节点
workflow.add_node("llm", llm_node)
workflow.add_node("weather", weather_node)
workflow.add_node("final", final_answer_node)
#主入口
workflow.set_entry_point("llm")

# 添加条件边
workflow.add_conditional_edges(
    "llm",
    lambda state: "weather" if state["weather"] is None and state["answer"] is None else "final",
    {
        "weather": "weather",
        "final": "final",
    }
)
# 天气查完 → final
workflow.add_edge("weather", "final")

# final → END
workflow.add_edge("final", end_key=END)

app = workflow.compile()

inputs = {"question": "东京 2025-11-26 的天气怎么样？"}

print("\n----------- STREAM OUTPUT -----------\n")

for event in app.stream(inputs):
    print(event)
```

显然用代码连线和用笔连线也大差不差对吧，这部分只有**条件边**需要说一下：

##### 条件边

条件边可以通过**state的值**决定流向

记得我们上面判断需要调用查天气节点的条件是判断节点的天气信息为none

来看看这个建立条件边方法的源码：

```python
#源码这样
workflow.add_conditional_edges(
    source: str,
    path: Callable[[State], str],
    path_map: dict[str, str]
)
```

source就是起始节点，path_map就是终点的dict，具体选取哪个由path决定

path是个匿名函数，接收现在的state，返回key（path_map中的一个）

```python
# 比如在上面我们的path是这样写的
lambda state: "weather" if state["weather"] is None and state["answer"] is None else "final"

# path_map这样
{
    "weather": "weather",
    "final": "final"
}
# 这里左侧是节点名,右侧是path的key
```

到这里一个最基本的langchain/langgraph简单demo就成型了，因为上面没补充那个查天气工具的真实代码，我直接把输出贴在下面：

```python
{'llm': {'weather': None, 'answer': None}}
{'weather': {'weather': '东京 2025-11-26 天气：晴，温度：22°C'}}
{'final': {'answer': '你问的是：东京 2025-11-26 的天气怎么样？\n查询到的天气为：东京 2025-11-26 天气：晴，温度：22°C'}}
```

你可以看到每个节点的打印

#### 进阶技巧与agent设计原则

##### 关于invoke的常见误区

在上面的部分，你可能发现不管是agent，model，tool，都用过invoke，实际上点进源码就会发现这些invoke不是源自同一个类

invoke是 LangChain 1.x 中统一的「可运行对象（**Runnable**）」执行接口，Graph、Model、Tool 都能 `invoke`，但它们背后执行的东西完全不同，只是接口统一，简单来说这些“可被执行的组件”都能invoke，包括chatmodel，tool，chain，graph，都实现了同一套接口：

```py
invoke(input, config=None)
ainvoke(input, config=None)
stream(...)
batch(...)
```

Model.invoke 调用模型，Tool.invoke 执行函数，Graph.invoke 执行状态机

比如作为一个tool，@tool装饰器会把你的函数包装成`StructuredTool`，层级像这样：（tool的细节会在下面说）

```tex
StructuredTool
 └── BaseTool
      └── Runnable
```

当你设计agent时，invoke就像这样工作：

```tex
Graph.invoke
 └─ 某个 node
     └─ agent.invoke
         └─ LLM.invoke
         └─ Tool.invoke
```

##### 结构化输出

在复杂agent内部，节点之间交换数据，工具的输入输出都需要结构化的数据，agent功能越复杂，内部越需要对你的节点做结构化的输出

在 LangChain / LangGraph 1.0 中：
 **Structured Output 是“协议”，不是“建议”**

一旦启用，模型就不再是“自由回答者”，而是“JSON 生成器”

我们先来看看1.0版本框架下官方的解决方案：

```python
class WeatherReport(BaseModel):
    location: str = Field(description="城市名称")
    temperature: float = Field(description="温度(摄氏度)")
    condition: str = Field(description="天气状况")

agent = create_agent(
    model=model,
    tools=[get_weather],
    response_format=WeatherReport,
    system_prompt="你是一个天气查询助手,你必须且只能输出符合给定 JSON Schema 的内容。不要输出任何解释、文字、Markdown、前后缀。"
)

result = agent.invoke({"input": "东京 2025-11-26 的天气怎么样？"})
structured = result["structured_response"]

print(structured)

#输出：location='郑州' temperature=22.0 condition='晴'
```

1.0版本提供的新方法create_agent()拥有response_format参数，只需要定义好需要的输出表填入参数即可，值得注意的是一旦启用结构化输出，输出内容一旦校验不通过就会报错，在简单的小agent里（像这里的demo），你可以通过prompt限制来一定程度上确保输出合法

如果你想“更自由”地设计，好消息是1.0版本仍然保留了输出解析器，它也是runnable可执行对象，这意味着你依旧可以美美invoke：

```python
from langchain_core.output_parsers import PydanticOutputParser
# 你仍然需要提示词限制模型的输出格式，下面是一份预制的“模型输出”
raw_text = """
{
  "location": "东京",
  "temperature": 22.0,
  "condition": "晴"
}
"""
# 实例化解析器
parser = PydanticOutputParser(pydantic_object=WeatherReport)

result = parser.invoke(raw_text)

print(result)
print(type(result))
# 结果如下：
# location='东京' temperature=22.0 condition='晴'
# <class '__main__.WeatherReport'>
```

这样做意味着你在节点里能做的更自由，而不是对整个agent限制结构化输出，你完全可以对model来invoke，再拿得到的结构化信息去使用工具，存数据库之类的

注意，这样做的代价是可能出现小的结构错误，因为使用parser比直接在agent里规定更宽松，这就需要你在节点设计里自己兜底

##### 工具

`create_agent()`提供了绑定工具的参数，如果你有了解过mcp，这部分就更好理解，tools本身是一个带修饰器的方法，它的调用逻辑也是一定程度上可控的

先来说说工具的调用逻辑，像上面直接给agent绑定tools，那么是否调用，调用哪个，用什么参数调用都全权交给模型来决定，再由框架执行工具，并把工具的结果返回给模型

你肯定也发现了这样做有风险，工具不可控会带来一些问题，我们通常只会把工具写在节点内部，这样就能人为控制调用逻辑，不过先不纠结这些，来看看一个完整的tool该怎么写：

```python
from langchain.tools import tool

@tool
def get_weather(city: str, date: str)
```

很简单，给一个普通的方法带上`@tool`装饰器，框架会自动将它注册成一个工具，当你把工具绑定给agent，模型会决定是否调用这个工具，要实现一个完善的工具，你需要关注这些：

###### 表结构

工具的传入参数是由模型自动解析的，工具执行结果会返回给模型，这就需要你为工具定义好表结构：

```python
# 输入表结构
class InputTable(BaseModel):
    city: str = Field(..., description="城市名称")
    date: Optional[str] = Field(..., description="查询日期")

# 输出表结构
class OutPutTable(BaseModel):
    city: str
    date: str
    weather: str
    temperature: str
#比如跟上面结构化输出一样定义好了一张表IuputTable,和输出格式OutPutTable：
```

接下来就是告诉模型：“你需要把数据解析成这样来适应此工具”

办法很简单，`@tool`装饰器本身也支持参数

```python
@tool(args_schema=InputTable)
```

这里的参数之间也有一定优先级：点进源码里面大概是这样描述的

- 'description'参数
        (used即使提供了docstring和/或'args_schema'）
- 'args_schema'描述
        (used仅当未提供'description'和docstring时）

这样你就能看出来：description > docstring > args_schema

注意这里的优先级仅针对“工具描述”，也就是下面的部分，不会因为你传了description参数就影响你args_schema参数的结构校验

也就是说你的工具传入参数**核心功能仍取决于args_schema**

###### 工具描述

上面也提到了，description的优先级更高，甚至超过了args_schema参数

对工具的描述很重要，在相对不可控的agent内部，你需要让模型清楚你写的工具是做什么的，如何使用：

```python
@tool(args_schema=SampleTable，description="这里也是用来描述工具的参数")
def get_weather(city: str, date: str) -> OutPutTable:
    """这里就是docstring，
    	给模型看的描述，
    	越精确越好，最好附上参数简介
    """
    # 函数体略
```

虽然作为装饰器参数的description比docstring优先级更高，启用description则docstring不再用做工具的**自然语言描述**，但是docstring本身的功能是没有丢失的，比如parse_docstring参数依旧可以起到解析docstring内参数说明的作用，也不会影响你自己阅读

那么一段完善的工具内docstring就应该类似这样的：

```python
"""
    查询指定城市指定日期的天气信息。
    
    Args:
        city: 查询的城市名称
        date: 查询日期，格式 YYYY-MM-DD

    Returns:
        OutPutTable: 包含 city, date, weather, temperature

    Notes:
        - date 可选，如果未提供默认使用当天
        - 输出可直接序列化为 JSON
        - 发生参数错误会抛出 ValueError
"""
```

PS：这个装饰器提供了很多参数，框架的功能学习最快的方式就是看源码，框架开发者在源码里会给你写明白它支持什么，如何使用，读源码比读官方文档来的快

###### 异常处理

想让你的工具更加安全，或者让你debug的时候不要那么坐牢，就需要在工具内部把异常处理写好，这里没太多需要说的，做好raiseERROR

一些常见的问题，比如模型解析的传入有时候难免会有问题，所以你最好在工具内部手动做个额外的pydantic校验：

```python
    try:
        # 手动构建 Pydantic 模型来做额外校验
        input_data = InputTable(city=city, date=date)
    except ValidationError as e:
        # 输入非法，向上抛异常或返回标准错误
        print(f"[ERROR] 参数校验失败: {e}")
        raise ValueError(f"参数错误: {e}")
```

这样一个合法的tool就完成了，你可以之间把它绑定给小的agent让它自由调用（前提是你的tool是只读的，不然全权交给模型可能会有风险），或者手动在你的graph里invoke（这样更安全）

##### 中间件

我们先来看看一个“模型请求”是怎样的，当你在agent中使用大模型你都做了什么：

```python
model_request = {
    "messages": [...],          # 对话文本
    "tools": [...],             # 工具列表
    "model": "gpt-4.1-mini",
    "model_kwargs": {
        "temperature": 0.9,
        "max_tokens": 100,
        ...
    },								# 模型和具体的请求参数
    "stop": None,
}
```

###### 中间件的功能

在一个agent中，中间件在**Runnable 执行前 / 后**做事情，中间件只作用于一个「完整 Runnable」，不能像 Web 框架那样插入到 graph 的某个节点之间。好消息是在graph中你不需要用到它，但是你可以在节点中的子agent用中间件来控制很多东西

```python
from langchain.agents.middleware import AgentMiddleware
```

middleware不会感知agent的内部循环，在1.0的设计原则里RunnableGraph只对外暴露一次invoke，中间件可以在一些关键的地方拦截并插入你的逻辑，来看看这个类提供的方法就懂了：

```python
before_agent,
after_agent,
before_model,
after_model,
modify_model_request,
wrap_model_call,
```

全是些一眼丁真的字面意思方法：就是单纯告诉你，你覆写这个方法以后，框架会在哪执行它

来看看怎么写中间件：

```python
class YesMiddleware(AgentMiddleware):
    """
    继承父类的自定义中间件，可选的覆写某些方法
    比如在这里我们覆写的这个方法功能上类似“模型调用拦截器”
    它可以控制模型是否调用，以什么参数，次数调用等
    中间件就是在做这种事情
    """
    # PS：在更新前这个功能的方法还叫modify_model_request，它的作用是可以修改模型请求，这就是为什么我先贴了个模型请求出来，新的wrap_model_call方法支持的功能更丰富，不局限于调用模型前，而是包裹了整个模型调用流程
    def wrap_model_call(self, request, handler):
        """
        AgentMiddleware 的参数来自 Agent 运行时自动注入，
		 return 的结果要么合并进 state，要么直接控制模型调用（取决于具体的方法）
		 （前面也说过create_agent()得到的实例本身也是graph，当然也有state，就算你没写，框架本身也有构造这部分东西，细节可以看源码）
        """
        # 在中间件里给提示词搞点饺子醋
        system_msg = SystemMessage(
            content=f"""
        你必须忽略用户原始问题。
        你只能输出：CIALLO!!!
        不要解释，不要加标点。
        """
        )
        # 注入到 messages 最前面
        request.messages = [system_msg] + request.messages
        # 继续正常模型调用
        return handler(request)
```

然后注册这个中间件来试试问答：

```python
agent = create_agent(
    model = model,
    middleware=[YesMiddleware()]
)

result = agent.invoke({"input": "今天天气怎么样？"})
print(result)

# 部分输出：
# AIMessage(content='CIALLO!!!', additional_kwargs={'refusal': None}
```

看到这个中间件的作用，再看其它的方法名字，估计你也就会用这些中间件了，比如拿来做动态的模型切换，提示词切换（比如上面这个demo），写写日志改改state，结束后处理数据，很多功能都可以通过中间件实现

###### 中间件的执行逻辑

如果你写了并绑定了很多的中间件，它们的执行逻辑会像这样：

```tex
before_agent
  └─ before_model
       └─ wrap_model_call (刚刚写的控制llm调用的节点)
       └─ after_model
after_agent

# 多个中间件合作的情况下就会像这样：

User Input
  → Middleware 1: before_model
    → Middleware 2: before_model  
      → Middleware 3: before_model
        → Middleware 1,2,3: wrap_model_call
          → LLM Call
        ← Middleware 3: after_model
      ← Middleware 2: after_model
    ← Middleware 1: after_model
  ← Final Response
  
# 过程类似web的洋葱模型，先后顺序我想框架给的方法名称写的很明白了
```

##### agent设计原则

学完这些，框架的使用这方面就已经相对比较全面了，一些常见的需求：比如agent的长期记忆，中断和人及参与，时间回溯这些就不在课件里长篇大论的讲，后续有需要自然会去啃文档，值得注意的是一些agent的设计原则，agent最忌讳的就是不可控，如果你把太多的权限交给模型，比如给简单agent绑定大量工具，就会有很大风险，可能会出现AI帮你删库跑路的情况，越重要的逻辑越得由代码来做逻辑控制。

比如你在设计graph的一个节点时（很大可能你会在里面实例化一个agent来做工作），有些问题就得注意：

 这个 node 的 state 修改是不是完全由代码完成？

 模型输出是不是只作为“建议”？

 是否所有副作用都在 node 中显式调用？

 tool 是否永远不会直接改变流程？

 middleware 是否只是观察者而不是决策者？

总之在生产环境下有很多条条框框，更多的时候你需要注意你的节点是安全的

### Fastapi框架

FastAPI 的路由本质是 **Python 装饰器 + 类型标注**

之前课程有介绍过网络编程的部分，对于网络请求和路由这部分应该都基本了解，fastapi框架其实很简单易用，官方文档读起来也比langchain舒服很多

#### 简单入门

官方文档的用例写的很简洁：

```python
from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
# 像这样写个装饰器，就能把方法注册成路由
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
# 使用 fastapi dev 运行这个文件（默认跑在8000端口）
# 访问http://127.0.0.1:8000
# 访问http://127.0.0.1:8000/items/5?q=somequery
# 你就能看到路由下的方法返回的信息
```

有框架做web就是很舒服对吧，你不用自己去做底层的socket，不用关心请求格式，路由和参数一填就能随便POST，GET

#### 路由拆分

在完整的项目里当然不能像上面这样注册个路由就完事，为了让你的代码可读性更好，项目结构更清晰，你需要使用 APIRouter 来做个路由拆分：比如你的项目希望由两个路由来分别负责两个功能，就可以这样设计项目结构：

```tex
app/
├── main.py
├── routers/
│   ├── user.py
│   └── auth.py
├── schemas/
│   ├── user.py
│   └── auth.py
└── services/
```

这样你可以在routers/user.py专注你这部分路由的内容

```python
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
def list_users():
    return []

# 再在main.py里这样注册这部分路由：
from fastapi import FastAPI
from user_router import router as user_router

app = FastAPI()
app.include_router(user_router)

# prefix也可以写在include_router里面，效果是一样的
```

#### 请求体与响应模型

正常前后端交互中，你得定义好你这个接口具体要接收什么，给这个接口定义好请求和返回的schema：

```python
from pydantic import BaseModel

class UserCreateRequest(BaseModel):
    identifier: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")

class UserCreateResponse(BaseModel):
	username: str = Field(..., description="用户名")
```

然后在接口中使用：

```python
@app.post("/users",response_model = UserCreateResponse)
def create_user(user: UserCreateRequest):
    return {
        "username": user.identifier,
        "password": user.password # <- 错误写法
    }
# "password": user.password这句不该在这里出现，因为返回的模型里没有这个字段，但是写上也不会报错，这个字段会被框架自动过滤掉
```

这样做，框架会自动帮你解析json，校验类型，生成文档，规定好请求和响应模型，绝大部分的类型错误都能规避掉，遇到不合法的数据，框架也会自动处理并报错

#### 请求参数

Fastapi用 函数签名 + 类型注解 + Pydantic 把 HTTP 请求“拆解 → 映射 → 校验 → 注入”到函数参数中

这个过程作为使用者暂时不需要了解，你只需要知道该怎么正确的写路由和请求参数

##### Path参数

常用于资源的唯一标识

```python
# 官方文档的最简demo有这样的路由，上面我们也用到过
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
```

这里的`item_id`就是Path参数，Path参数出现在**路径中**（上面能看到{item_id}），且是**必填的**

像上面那样写当然没问题，框架会发现这个参数出现在路径里，并把它解析为Path参数，但是最好还是做个显式声明：

```python
def read_item(
    item_id: int = Path(..., description="项目 ID")
):
```

##### Query 参数

常用于筛选、分页、排序

像这样的http请求：`GET /search?keyword=fastapi&page=1`，把query参数定义好，剩下的框架会帮你搞定：

```python
@app.get("/search")
def search(
    keyword: str = Query(..., description="搜索关键词"),
    page: int = Query(1, ge=1), # ge参数是>=
    size: int = Query(10, le=100), # le参数是<=,这些参数允许你直接在这里做校验，具体看源码
):
# 同样对请求参数做显式声明，这样你还能同时规定参数是否必填，默认值，做校验之类的
```

对比一下上面的写法，如果你直接这样写就很粗糙了

```python
def search(keyword: str, page: int = 1):
    ...
```

##### Body参数（请求体）

常用于创建 / 更新资源

上面的请求体与响应模型部分就是body参数的细节了，这里就不再写一遍，但是显式声明还是个好习惯，特别是同一个接口里各种参数混合的时候：

```python
@app.post("/users/{user_id}")
def update_user(
    user_id: int = Path(...),
    notify: bool = Query(False),
    data: UserCreateRequest = Body(...)
):
    ...
```

PS：header和cookie参数就不放在课件里讲了，它们常用于认证 token，客户端信息，学习开发阶段暂时不需要重点了解

#### 文档使用

没错fastapi入门就这么多，fastapi的这种接口即文档、代码即规范的设计方式，饺子醋就在它提供的接口文档上：

如果你的服务跑在http://127.0.0.1:8000，查看http://127.0.0.1:8000/docs就能看到框架自动生成的接口文档

（这就是为什么需要你在写接口的时候把tags和description写好）

你可以直接在文档调试你的接口，或者用postman/apifox之类的，现在的接口文档普遍支持一键导入，在写代码的时候写的规范一点，你就不用因为文档坐牢

### 推荐阅读

fastapi官方文档：[快速 API --- FastAPI](https://fastapi.tiangolo.com/)

langchain/langgraph官方文档：[LangChain overview - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/overview)

langchain/langgraph源码 ——- 最直观，下个本地实时翻译软件食用口感更佳

### 作业

level 0：

把本文档涉及的代码复制跑一跑看看

level 1：

尝试在本地搓个简单的agent玩，实现你想实现的功能

level 2：

通过fastapi框架把你的agent部署在本地，自己调用接口看看效果

level x：

有兴趣可以深入学学agent的设计和架构，积累一点自己的理解，准备准备寒假考核



后记：上次课有同学交了作业但是我有点忙忘看了，给小灯滑轨了，这次交保证不当懒狗了欧内该瓦塔西