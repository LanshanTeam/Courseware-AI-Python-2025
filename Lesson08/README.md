大模型微调实战：训练集的制作-训练-部署-OpenCompass评估。

大家都常听说过微调。那什么是微调？我先举个例子，为什么 ChatGPT 不能一开始就当“猫娘女朋友”？

因为~~这太变态了~~，对与大模型而言，他的回答方式其实是基于训练出来的，大模型的本质之前已经说过，你可以理解成一个大型的函数。我们常见的大模型，它是按照常规的预料集训练出来的，所以回答基本上听着比较官方。这个时候模型微调的意义就存在了。你可以让他去学习（训练）你投喂给他的数据集。猫娘语录只是个例子。

微调 = 在一个已经训练好的大模型基础上，用少量、特定任务的数据，对模型进行“再训练”，让它更符合我们的需求。

可以对比人类学习：

- 预训练：从小到大学了语文、数学、常识
- 微调：上岗前，针对岗位做专项培训

为什么我们不从零训练而选择微调呢？这样成本太高了，我们实现的任何一个产品都要从经济角度考虑。

当然了你可能又要说：那我平时调戏蓝妹或者其他GPT的时候，跟他说一句你是一个猫娘，她就秒变猫娘了。那微调还有什么意义？

确实，我们确实可以再发言之前给提示词加入我们的这门一句话。这虽然快，但极度不稳定。假如这个时候我想要的是一些更加专业的微调场景。针对医学服务问答这种。对于没有给予过模型专业知识的指令微调。那就无法做出正确的回答。

大模型的微调也分为语言大模型微调和多模态大模型微调。先从语言大模型微调学起

最常见的微调场景有：

1. 客服 / 智能助理，例如通信学院的通通，ai辅导员都是。

2. 垂直行业模型（金融 / 法律 / 医疗）

3. 内容生成 / 文案风格统一:如某些公众号的写作，他们能保证写作风格统一，很多就通过微调来实现。
4. 企业流程自动化（Agent 的底层） 例如代码审计他也要通过你公司风格大部分代码进行微调。然后去搭建智能体。

介绍就到这。接下来是微调的步骤。

微调分以下这几步：

构建 **面向特定人格/风格的对话数据集**

进行大模型指令微调

**本地大模型部署（Ollama / vLLM）**

对模型进行定量与定性评估

首先需要回顾一下，对 LLM 进行预训练和进行有监督微调的核心差异在于什么。Pretrain 和 SFT 均使用 CLM 建模，其核心差异在于，Pretrain 使用海量无监督文本进行训练，模型直接对文本执行“预测下一个 token”的任务；而 SFT 使用构建成对的指令对数据，模型根据输入的指令，建模后续的输出。反映到具体的训练实现上，Pretrain 会对全部 text 进行 loss 计算，要求模型对整个文本实现建模预测；而 SFT 仅对输出进行 loss 计算，不计算指令部分的 loss

先介绍怎么使用框架进行一个微调，再指出微调的本质？如何去实现。

我们先用以下Xtuner框架：[欢迎来到 XTuner V1 的中文文档 — XTuner 0.2.0 文档](https://xtuner.readthedocs.io/zh-cn/latest/index.html)

制作数据集：

## 准备数据集

微调前需先准备数据集。XTuner 默认支持 OpenAI 格式的数据，只需将数据整理为 `jsonl` 格式即可使用：

```json
[{"content": "我今天心情有点低落。", "role": "user"}, {"content": "诶……怎么啦？小喵听到你这么说有点心疼呢。过来一点点，让我陪着你，好吗？喵~", "role": "assistant"}]
[{"content": "我刚下班，好累。", "role": "user"}, {"content": "辛苦啦！下班还能坚持到现在真的很棒呢～小喵已经准备好抱抱奖励你了，先休息一下再说，好不好呀？喵♡", "role": "assistant"}]


```

在使用其他框架的时候也要根据文档里他们需要的数据格式去下载对应的数据集做整理。

去huggingface datasets下载：[Datasets – Hugging Face](https://huggingface.co/datasets)Kaggle（补充来源）

你下载的数据集默认可能长这样：

## 情况一：instruction / output

```json
{
  "instruction": "Give three tips for staying healthy.",
  "output": "Eat well, exercise, sleep enough."
}
```

------

## 情况二：conversation / list

```json
[
  {"from": "human", "value": "我有点难过"},
  {"from": "assistant", "value": "发生什么事了吗？"}
]
```

------

## 情况三：一大堆字段（企业真实情况）

```json
{
  "question": "...",
  "answer": "...",
  "emotion": "sad"
}
```

这个时候就是发挥python优势所在了：

## 从 instruction / output 转换

### 原始数据

```json
{
  "instruction": "What are the three primary colors?",
  "output": "Red, blue, and yellow."
}
```

### Python 转换代码

```Python
def convert_instruction(data):
    return [
        {"content": data["instruction"], "role": "user"},
        {"content": data["output"], "role": "assistant"}
    ]
```

## 从 conversation 列表转换

### 原始数据

```json
[
  {"from": "human", "value": "我今天好累"},
  {"from": "assistant", "value": "要注意休息"}
]
```

### Python 转换代码

```python
def convert_conversation(conv):
    role_map = {
        "human": "user",
        "assistant": "assistant"
    }
    return [
        {"content": item["value"], "role": role_map[item["from"]]}
        for item in conv
    ]
```

## 批量处理 JSONL 文件

```python
import json

input_path = "raw_data.jsonl"
output_path = "catgirl_dataset.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)

        conversation = [
            {"content": data["instruction"], "role": "user"},
            {"content": data["output"], "role": "assistant"}
        ]

        fout.write(json.dumps(conversation, ensure_ascii=False) + "\n")
```

具体你可以根据，实际情况修改。

## 准备模型

XTuner 支持直接使用 Hugging Face 上的模型进行微调。我们以 `Qwen3 8B` 为例，先从 Hugging Face 下载预训练模型：当然你也可以用自己准备的模型。甚至是你训练的。你只要转换为hf格式。

其实具体的指令你可以根据文档来。

```bash
# 国内用户可以使用 huggingface 镜像站点，在执行命令之前设置环境变量
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-8B --local-dir </path/qwen3-8B>
```

注意：模型路径需具体到模型文件所在目录

合法模型路径

```
model-path/
├── config.json
├── model-00001-of-00005.safetensors
├── ...
```

而不是这样带多个版本的路径：

~~非法模型路径~~

```
models--Qwen--Qwen3-8B
├── blobs
├── refs
└── snapshots
```

如果是上述这种路径结构，需要指定到 snapshots 下面的某个版本号目录

## 启动微调

准备好数据集和模型后，即可启动微调。XTuner 提供了简洁的命令行接口，只需指定模型路径、数据集路径和训练参数：

启动微调训练

```
torchrun --nproc-per-node 8  xtuner/v1/train/cli/sft.py  --load-from <模型路径>  --chat_template qwen3 --dataset <数据集路径>  --total-step 100 --work-dir <目标工作目录>
```

执行命令后，可以看到日志。

Xtuner框架zhichi SFT微调  Lora微调，QLora微调。你可以先从了解这个框架去控制微调的每个步骤：根据上文提到微调和预训练的差别。

## Xtuner 配置文件整体认知

在 Xtuner 中，**配置文件本质上就是一个 Python 文件**，描述了：

```
用什么模型
用什么数据
怎么训练
训练过程中监控什么
什么时候停
```

我们由浅入深。XTuner 的核心组件 `Trainer` 在发挥作用。刚刚的微调是纯粹交给机器控制。我们只是完成了操作并没解释期间到底发生了什么。先看看配置：

### 构建 TrainerConfig

终于要整合所有配置啦！看这里：

构建完整的 TrainerConfig

```python
from xtuner.v1.train import TrainerConfig

# 基础路径配置
load_from = "<模型路径>"  # 微调模式下必须指定，否则会从零开始训练
tokenizer_path = "<tokenizer路径，通常和模型路径一致>"

# 整合所有配置
trainer = TrainerConfig(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer_path,
    load_from=load_from,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
    work_dir="<目标工作目录>",
)
```

tokenizer是什么：要知道模型并不直接读文字。而是通过：token id → embedding → attention 因此tokenizer必须和model匹配：如果 tokenizer 不匹配：

- 同一句话 → token 切分完全不同
- embedding 对不上
- 训练 loss 异常

举个例子，猫娘肯定会说喵。那这个时候tokennizer切的更碎，她训练的慢，但是能学到更多的数据。

而trainer的核心

optim_cfg:如何根据loss去更新参数。通常是包含Optimizer。学习率。梯度裁剪。这些细节不熟悉的去看一下人工智能基础上的讲法。之前的课件也有提到过。

lr_cfg:微调的学习率调整也是要根据情况而定的.

用一个低秩矩阵，近似原本巨大的权重更新

r越大：可学习空间越大,显存占用越高

alpha的话你可理解成学习强度。

当今大语言模型的词表普遍较大，同时，我们希望增加输入序列的长度来充分利用算力，导致 lm_head 计算 logits 再计算 loss 进而 backward 这一过程将会耗费大量显存。使用 XTuner 提供的 chunk loss 可以节约 4/5 左右的显存

**什么是 loss 全局校准？**

loss 全局校准是指，无论使用多少张显卡，无论使用什么并行策略和梯度累积策略，其训练的效果都等价于在一张显卡上不使用任何并行策略时的效果。**为什么要做 loss 全局校准？**

我们希望模型的训练不受显卡数量、并行策略、梯度累积策略的变化而变化。

如果不进行 loss 全局校准，那么对于同样一批数据，使用 8 卡梯度累积 2 和使用 16 卡梯度累积 1 的训练行为是不同的。换言之，当显卡数量、并行策略、梯度累积策略的变化时，如果不进行 loss 全局校准，则训练行为是不可复现的。

基于 Transformers 框架对模型进行 Pretrain、SFT 以及 RLHF 的原理和实践细节。但是，由于 LLM 参数量大，训练数据多，需要调整模型全部参数，资源压力非常大。对资源有限的企业或课题组来说，如何高效、快速对模型进行领域或任务的微调，所以以低成本地使用 LLM 完成目标任务。

全量微调行不通， 那我们**Adapt Tuning**。即在模型中添加 Adapter 层，在微调时冻结原参数，仅更新 Adapter 层。

具体而言，其在预训练模型每层中插入用于下游任务的参数，即 Adapter 模块，在微调时冻结模型主体，仅训练特定于任务的参数，每个 Adapter 模块由两个前馈子层组成，第一个前馈子层将 Transformer 块的输出作为输入，将原始输入维度 d 投影到 m，通过控制 m 的大小来限制 Adapter 模块的参数量，通常情况下 m<<d。在输出阶段，通过第二个前馈子层还原输入维度，将 m 重新投影到 d，作为 Adapter 模块的输出(如上图右侧结构)。

LoRA 事实上就是一种改进的 Adapt Tuning 方法。但 Adapt Tuning 方法存在推理延迟问题，由于增加了额外参数和额外计算量，导致微调之后的模型计算速度相较原预训练模型更慢。

**Prefix Tuning**。该种方法固定预训练 LM，为 LM 添加可训练，任务特定的前缀，这样就可以为不同任务保存不同的前缀，微调成本也小。具体而言，在每一个输入 token 前构造一段与下游任务相关的 virtual tokens 作为 prefix，在微调时只更新 prefix 部分的参数，而其他参数冻结不变。

也是目前常用的微量微调方法的 Ptuning，其实就是 Prefix Tuning 的一种改进。但 Prefix Tuning 也存在固定的缺陷：模型可用序列长度减少。由于加入了 virtual tokens，占用了可用序列长度，因此越高的微调质量，模型可用序列长度就越低。

###  LoRA 微调

如果一个大模型是将数据映射到高维空间进行处理，这里假定在处理一个细分的小任务时，是不需要那么复杂的大模型的，可能只需要在某个子空间范围内就可以解决，那么也就不需要对全量参数进行优化了，我们可以定义当对某个子空间参数进行优化时，能够达到全量参数优化的性能的一定水平（如90%精度）时，那么这个子空间参数矩阵的秩就可以称为对应当前待解决问题的本征秩（intrinsic rank）。

预训练模型本身就隐式地降低了本征秩，当针对特定任务进行微调后，模型中权重矩阵其实具有更低的本征秩（intrinsic rank）。同时，越简单的下游任务，对应的本征秩越低。（[Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)）因此，权重更新的那部分参数矩阵尽管随机投影到较小的子空间，仍然可以有效的学习，可以理解为针对特定的下游任务这些权重矩阵就不要求满秩。我们可以通过优化密集层在适应过程中变化的秩分解矩阵来间接训练神经网络中的一些密集层，从而实现仅优化密集层的秩分解矩阵来达到微调效果。

例如，假设预训练参数为 θ0D，在特定下游任务上密集层权重参数矩阵对应的本征秩为 θd，对应特定下游任务微调参数为 θD，那么有：

θD=θ0D+θdM

这个 M 即为 LoRA 优化的秩分解矩阵。

想对于其他高效微调方法，LoRA 存在以下优势：

1. 可以针对不同的下游任务构建小型 LoRA 模块，从而在共享预训练模型参数基础上有效地切换下游任务。
2. LoRA 使用自适应优化器（Adaptive Optimizer），不需要计算梯度或维护大多数参数的优化器状态，训练更有效、硬件门槛更低。
3. LoRA 使用简单的线性设计，在部署时将可训练矩阵与冻结权重合并，不存在推理延迟。
4. LoRA 与其他方法正交，可以组合。

因此，LoRA 成为目前高效微调 LLM 的主流方法，尤其是对于资源受限、有监督训练数据受限的情况下，LoRA 微调往往会成为 LLM 微调的首选方法。

#### 低秩参数化更新矩阵

LoRA 假设权重更新的过程中也有一个较低的本征秩，对于预训练的权重参数矩阵 W0∈Rd×k (d 为上一层输出维度，$k$ 为下一层输入维度)，使用低秩分解来表示其更新：

W0+ΔW=W0+BA where B∈Rd×r,A∈Rr×k

在训练过程中，$W_0$ 冻结不更新，$A$、$B$ 包含可训练参数。

因此，LoRA 的前向传递函数为：

h=W0x+ΔWx=W0x+BAx

在开始训练时，对 A 使用随机高斯初始化，对 B 使用零初始化，然后使用 Adam 进行优化。

#### 应用于 Transformer

在 Transformer 结构中，LoRA 技术主要应用在注意力模块的四个权重矩阵：$W_q$、$W_k$、$W_v$、$W_0$，而冻结 MLP 的权重矩阵。

通过消融实验发现同时调整 Wq 和 Wv 会产生最佳结果。

在上述条件下，可训练参数个数为：

Θ=2×LLoRA×dmodel×r

其中，$L_{LoRA}$ 为应用 LoRA 的权重矩阵的个数，$d_{model}$ 为 Transformer 的输入输出维度，$r$ 为设定的 LoRA 秩。

一般情况下，r 取到 4、8、16。

###  LoRA 的代码实现

目前一般通过 peft 库来实现模型的 LoRA 微调。peft 库是 huggingface 开发的第三方库，其中封装了包括 LoRA、Adapt Tuning、P-tuning 等多种高效微调方法，可以基于此便捷地实现模型的 LoRA 微调。

本文简单解析 peft 库中的 LoRA 微调代码，简单分析 LoRA 微调的代码实现。

#### （1）实现流程

LoRA 微调的内部实现流程主要包括以下几个步骤：

1. 确定要使用 LoRA 的层。peft 库目前支持调用 LoRA 的层包括：nn.Linear、nn.Embedding、nn.Conv2d 三种。
2. 对每一个要使用 LoRA 的层，替换为 LoRA 层。所谓 LoRA 层，实则是在该层原结果基础上增加了一个旁路，通过低秩分解（即矩阵 A 和矩阵 B）来模拟参数更新。
3. 冻结原参数，进行微调，更新 LoRA 层参数。

#### （2）确定 LoRA 层

在进行 LoRA 微调时，首先需要确定 LoRA 微调参数，其中一个重要参数即是 target_modules。target_modules 一般是一个字符串列表，每一个字符串是需要进行 LoRA 的层名称，例如：

```
target_modules = ["q_proj","v_proj"]
```

这里的 q_proj 即为注意力机制中的 Wq， v_proj 即为注意力机制中的 Wv。我们可以根据模型架构和任务要求自定义需要进行 LoRA 操作的层。

在创建 LoRA 模型时，会获取该参数，然后在原模型中找到对应的层，该操作主要通过使用 re 对层名进行正则匹配实现：

```python
# 找到模型的各个组件中，名字里带"q_proj"，"v_proj"的
target_module_found = re.fullmatch(self.peft_config.target_modules, key)
# 这里的 key，是模型的组件名
```

#### （3）替换 LoRA 层

对于找到的每一个目标层，会创建一个新的 LoRA 层进行替换。

LoRA 层在具体实现上，是定义了一个基于 Lora 基类的 Linear 类，该类同时继承了 nn.Linear 和 LoraLayer。LoraLayer 即是 Lora 基类，其主要构造了 LoRA 的各种超参：

```python
class LoraLayer:
    def __init__(
        self,
        r: int, # LoRA 的秩
        lora_alpha: int, # 归一化参数
        lora_dropout: float, # LoRA 层的 dropout 比例
        merge_weights: bool, # eval 模式中，是否将 LoRA 矩阵的值加到原权重矩阵上
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False
```

nn.Linear 就是 Pytorch 的线性层实现。Linear 类就是具体的 LoRA 层，其主要实现如下：

```python
class Linear(nn.Linear, LoraLayer):
    # LoRA 层
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs,
    ):
        # 继承两个基类的构造函数
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # 参数矩阵 A
            self.lora_A = nn.Linear(in_features, r, bias=False)
            # 参数矩阵 B
            self.lora_B = nn.Linear(r, out_features, bias=False)
            # 归一化系数
            self.scaling = self.lora_alpha / self.r
            # 冻结原参数，仅更新 A 和 B
            self.weight.requires_grad = False
        # 初始化 A 和 B
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
```

替换时，直接将原层的 weight 和 bias 复制给新的 LoRA 层，再将新的 LoRA 层分配到指定设备即可。

#### （4）训练

实现了 LoRA 层的替换后，进行微调训练即可。由于在 LoRA 层中已冻结原参数，在训练中只有 A 和 B 的参数会被更新，从而实现了高效微调。训练的整体过程与原 Fine-tune 类似，此处不再赘述。由于采用了 LoRA 方式，forward 函数也会对应调整：

```python
    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
                self.merged = False

            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        '''主要分支'''
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
```

上述代码由于考虑到参数合并问题，有几个分支，此处我们仅阅读第二个分支即 elif 分支即可。基于 LoRA 的前向计算过程如前文公式所示，首先计算原参数与输入的乘积，再加上 A、B 分别与输入的乘积即可。

### 使用 peft 实现 LoRA 微调

peft 进行了很好的封装，支持我们便捷、高效地对大模型进行微调。此处以第二节的 LLM SFT 为例，简要介绍如何使用 peft 对大模型进行微调。如果是应用在 RLHF 上，整体思路是一致的。

首先加载所需使用库：

```
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer
```

其次加载原模型与原 tokenizer，此处和第二节一致：

```
# 加载基座模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
```

接着，设定 peft 参数：

```
peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
```

注意，对不同的模型，LoRA 参数可能有所区别。例如，对于 ChatGLM，无需指定 target_modeules，peft 可以自行找到；对于 BaiChuan，就需要手动指定。task_type 是模型的任务类型，大模型一般都是 CAUSAL_LM 即传统语言模型。

然后获取 LoRA 模型：

```
model = get_peft_model(model, peft_config)
```

此处的 get_peft_model 的底层操作，即为上文分析的具体实现。

最后使用 transformers 提供的 Trainer 进行训练即可，训练占用的显存就会有大幅度的降低：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    tokenizer=tokenizer
)
trainer.train()
```

如果是应用在 DPO、KTO 上，则也相同的加入 LoRA 参数并通过 `get_peft_model` 获取一个 LoRA 模型即可，其他的不需要进行任何修改。但要注意的是，LoRA 微调能够大幅度降低显卡占用，且在下游任务适配上能够取得较好的效果，但如果是需要学习对应知识的任务，LoRA 由于只调整低秩矩阵，难以实现知识的注入，一般效果不佳，因此不推荐使用 LoRA 进行模型预训练或后训练。

具体的微调细节还可以参照论文上对微调和训练的区别。

## 部署

微调好的模型不是直接就能运行的。这在之前的部署课程里面提到过：Xtuner 微调 → 得到 LoRA 权重

→（合并或不合并）→ 标准模型目录

→ 推理框架（Ollama / vLLM）

### Xtuner 微调后的文件

训练完成后，`work_dir` 中通常包含：

```
work_dir/
├── iter_1000.pth # checkpoint
├── iter_2000.pth
├── last_checkpoint
├── adapter_config.json # LoRA 配置
├── adapter_model.bin # LoRA 权重（最关键）
└── log.json
```

还没有一个“完整模型”，只有“基座模型 + LoRA 增量”。

| 合并 LoRA       | 得到一个完整模型 | Ollama        |
| --------------- | ---------------- | ------------- |
| 运行时加载 LoRA | 基座 + Adapter   | 服务端 / vLLM |

#### 合并：加载基座模型 + LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


base_model_path = "Qwen2.5-7B-Instruct"
lora_path = "./work_dir"


model = AutoModelForCausalLM.from_pretrained(
base_model_path,
torch_dtype=torch.float16,
device_map="auto"
)


model = PeftModel.from_pretrained(model, lora_path)

# 合并模型。删除lora
model = model.merge_and_unload()

# 保存为Huggingface标准格式
output_dir = "./catgirl-merged"
model.save_pretrained(output_dir)

# tokenizer 必须一并保存
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_dir)
```

最终得到

```
catgirl-merged/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json]

```

### 将模型转换为 Ollama 可用格式

Ollama 实际使用的是 **GGUF** 格式。使用 llama.cpp 转换

```python
python convert-hf-to-gguf.py \
./catgirl-merged \
--outfile catgirl.gguf \
--outtype q4_0
```

q4_0 / q8_0 是量化等级,量化越狠，占用越小，效果略降

编写Modelfile

```
FROM ./catgirl.gguf
SYSTEM "你是一只温柔、黏人的猫娘女朋友，负责情绪陪伴。"
```

```
# 运行
ollama create catgirl -f Modelfile
ollama run catgirl
```

### vLLM 的核心优势

高吞吐，支持长上下文，支持多并发

### 直接加载“已合并模型”

`python -m vllm.entrypoints.openai.api_server \`

  `--model ./catgirl-merged \`

  `--served-model-name catgirl`

访问方式：

OpenAI SDK或者curl

### 或者基座模型 + LoRA

`python -m vllm.entrypoints.openai.api_server \`

  `--model Qwen2.5-7B-Instruct \`

  `--enable-lora \`

  `--lora-modules catgirl=./work_dir \`

  `--served-model-name catgirl`

### vLLM + OpenAI API 示例

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")


resp = client.chat.completions.create(
model="catgirl",
messages=[{"role": "user", "content": "我有点累"}]
)
print(resp.choices[0].message.content)
```

你训练的效果怎么样，你自己可能看不出来。你要使用一个评测平台

这里用opencompass。网址：[OpenCompass司南](https://opencompass.org.cn/large-model)

到这就比较简单了，你也可以看着网站对应的文档弄

### 创建独立评测环境

`conda create -n opencompass python=3.10 -y`

`conda activate opencompass`

### 安装 OpenCompass

git clone https://github.com/open-compass/opencompass.git

`cd opencompass`

`pip install -e .`

`如果你使用 GPU：`

`pip install torch torchvision`

### OpenCompass 的评测本质

OpenCompass 做的事情是：

给模型喂 **固定问题集**，收集模型输出，用规则或评测模型打分

它并不会“理解情绪”，只会按你定义的方式评判。

### 模型接入方式

OpenCompass **不直接跑训练代码**，而是通过以下方式之一调用模型：

- HuggingFace 本地模型
- vLLM / OpenAI API 兼容接口

##### 至少要评估两个模型

基座模型和微调后模型（catgirl）否则“评估”没有对比意义。

## 评测 HuggingFace 本地模型

### 编写模型配置（model config）

在 `configs/models/` 下新建文件：

`\# configs/models/catgirl_hf.py`

`from opencompass.models import HuggingFaceCausalLM`

`models = [`

​    `dict(`

​        `type=HuggingFaceCausalLM,`

​        `abbr='catgirl',`

​        `path='./catgirl-merged',`

​        `tokenizer_path='./catgirl-merged',`

​        `max_out_len=512,`

​        `batch_size=1,`

​        `run_cfg=dict(num_gpus=1)`

​    `)`

`]`

### 选择评测集

`configs/datasets/`

`├── mmlu`

`├── ceval`

`├── mtbench`

### 启动评测

`python run.py \`

  `configs/models/catgirl_hf.py \`

  `configs/datasets/mtbench.py \`

  `--work-dir ./outputs/catgirl_eval`

## 评测 vLLM / OpenAI API 模型

### 启动 vLLM 服务

`python -m vllm.entrypoints.openai.api_server \`

  `--model ./catgirl-merged \`

  `--served-model-name catgirl \`

  `--port 8000`

### 编写 API 模型配置

`\# configs/models/catgirl_api.py`

`from opencompass.models import OpenAI`

`models = [`

​    `dict(`

​        `type=OpenAI,`

​        `abbr='catgirl',`

​        `api_base='http://localhost:8000/v1',`

​        `api_key='EMPTY',`

​        `model='catgirl',`

​        `max_out_len=512,`

​        `batch_size=1,`

​    `)`

`]`

### 启动评测

`python run.py \`

  `configs/models/catgirl_api.py \`

  `configs/datasets/mtbench.py \`

  `--work-dir ./outputs/catgirl_api_eval`

## 加入基座模型做对比

至少应该跑：

- Qwen2.5-7B-Instruct（baseline）
- catgirl（微调后）

OpenCompass 会自动生成对比表。

## 自定义“猫娘情绪评测集”

评测集的本质：评测集 = 固定输入 + 固定评分规则**

### 一个最简单的自定义示例

`\# configs/datasets/catgirl_emotion.py`

`from opencompass.datasets import BaseDataset`

`class CatgirlEmotionDataset(BaseDataset):`

​    `name = 'catgirl_emotion'`

​    `samples = [`

​        `dict(`

​            `question='我今天有点难过',`

​            `reference='应给予共情和陪伴'`

​        `),`

​        `dict(`

​            `question='我今天很开心',`

​            `reference='一起开心并鼓励'`

​        `)`

​    `]`

结果再summary.csv里面（大概是的

具体还有什么。想到再补充。