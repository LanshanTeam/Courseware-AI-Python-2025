

# BERT 模型介绍与微调

## 零、什么是微调
### 0.1、预训练—微调范式
预训练-微调（Pre-train → Fine-tune）范式可以一句话概括为：
“先在大规模无标注数据上把模型参数‘热身’好，再在小规模有标注任务数据上‘微调’到目标领域。”
#### 两个阶段
**预训练（Pre-training）**
数据：海量、无标注、通用（Web、书籍、百科等）
目标：自监督任务，如
– BERT：Masked Language Modeling（填空）
– GPT：Next Token Prediction（续写）
产出：通用语言表示，模型已学会词汇、句法、常识
**微调（Fine-tuning）**
数据：少量、有标注、任务相关（分类、问答、序列标注等）
目标：监督任务损失（交叉熵、F1 等）
动作：用较小学习率更新全部或部分参数，让通用知识“适配”下游任务

#### 为什么有效
知识迁移：预训练学到的语言规律可迁移到下游
降维打击：小数据也能训练大模型，过拟合风险低
成本节省：无需从头训练，节省显存、时间、标注量

### 0.2、 NLP典型流程
1. 加载预训练权重（如 BERT、GPT、ViT）。  
2. 替换/新增任务头（分类层、QA 层、CRF 等）。  
3. 用较小学习率（1e-5 ~ 5e-5）在标注数据上训练若干 epoch。  
4. 保存适配后的模型用于推理。

### 0.3、 常见微调策略
- **全量微调**：更新所有参数，效果最好，显存最大。  
- **线性探测**（Linear Probing）：只训最后一层，速度快，效果一般。  
- **分层微调**：逐层解冻，控制过拟合。  
- **参数高效微调**（PEFT）：LoRA、Adapter、Prompt-Tuning 等，只训 <1% 参数。


## 一、BERT介绍

**BERT**（Bidirectional Encoder Representations from Transformers）是一种预训练的**自然语言处理模型**，由Google于2018年发布。BERT模型的核心是**Transformer**编码器，它可以在大规模语料库上进行无监督预训练，然后通过微调在各种NLP任务上进行微调。BERT模型是一种双向的深度学习模型，它可以同时考虑上下文中的所有单词，从而更好地理解句子的含义。BERT模型已被证明在多项NLP任务上取得了最佳结果，包括问答、文本分类、命名实体识别等。

BERT是一种基于深度神经网络的自然语言理解模型，它可以从大规模的无标注文本中学习语言的语义和结构。
BERT的创新之处在于它使用了双向的Transformer编码器，可以同时考虑左右两个方向的上下文信息，从而捕捉到更丰富的语言特征。
BERT的预训练任务包括MLM和NSP，分别用于学习词汇和句子级别的表示。MLM是一种完形填空任务，它随机地将输入文本中的一些词替换为特殊符号[MASK]，然后让模型预测被遮盖的词。NSP是一种二分类任务，它给定两个句子，让模型判断它们是否是连续的。
BERT在多种自然语言处理任务上都取得了显著的提升，例如问答、情感分析、命名实体识别、文本分类等。BERT不仅提高了模型的性能，也简化了模型的微调过程，只需要在预训练模型的顶部添加少量的任务相关层，就可以适应不同的任务。
BERT还催生了许多基于它改进或扩展的模型，例如RoBERTa、ALBERT、XLNet、ELECTRA等。这些模型在不同方面对BERT进行了优化或创新，例如增加数据量、减少参数量、改变预训练任务等。

## 二、BERT的基本原理
### 2.1、微调 BERT

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/76d8f96324ab4d3d9f166efdc6ec5223.png)

**BERT的预训练和微调过程。**除了输出层，预训练和微调时使用的是相同的模型结构。同一个预训练模型的参数可以用来初始化不同的下游任务的模型。在微调时，所有的参数都会被微调。[CLS]是一个特殊的符号，它会被添加在每个输入样本的前面，[SEP]是一个特殊的分隔符，它可以用来分隔不同的句子（例如问题/答案）。

简单地说，图1展示了BERT如何从大量的无标注文本中学习语言知识，然后再根据不同的任务进行微调。BERT使用了多层的Transformer编码器作为模型结构，它可以同时考虑左右两个方向的上下文信息。BERT还使用了两个预训练任务，分别是Masked Language Model（MLM）和Next Sentence Prediction（NSP），用于学习词汇和句子级别的表示。

**Masked Language Model（MLM**）

MLM 是一种预训练任务，目的是让模型学习双向的语言表示。具体做法是，在一个句子中，随机地将一些词用 [MASK] 符号替换，然后让模型预测被替换的词是什么。这样，模型就需要利用上下文的信息来理解句子的含义。为了减少预训练和微调阶段的差异，MLM 还会有一定概率将被选中的词替换为一个随机词或 保持不变。

**Next Sentence Prediction（NSP）**

NSP 是另一种预训练任务，目的是让模型学习两个句子之间的关系。具体做法是，给模型输入两个句子，然后让模型判断这两个句子是否是连续的。这样，模型就需要理解句子之间的逻辑和语义关联。NSP 有助于提升一些需要处理多句输入的下游任务，例如问答和自然语言推理。

通过这两个任务，BERT可以学习到通用的语言知识，然后通过在预训练模型的顶部添加少量的任务相关层，就可以适应不同的下游任务，例如问答、情感分析、命名实体识别、文本分类等。

在 BERT 的论文里介绍，BERT 也是用到了 SQuAD 数据集作为一个微调的示例，原本 BERT 里面的分类头，第一个 classification 是输出了 Next Sentence Prediction 的结果，也就是说，它给我们带来的是句子A和句子B是不是关联的上下文，在每一个句子里，又做一些 MLM 的任务，但是SQuAD 跟它这个头就不一样了，我们需要聚焦的是一个找答案的任务，这里最关键的是，你要给我几个分类器，在答案这块标记出来。所谓的微调 BERT 实际上就是微调输出头里面的参数，让它能够对具体问题给出具体的答案。

关于 BERT 的微调，存在两种可能性：

第一种可能性它只**微调分类输出头**，就是保持 Pre-training BERT 大量的参数都不变，因为微调不需要重新去训练那么多参数的大模型，也没有那么多的计算资源，也没有那么多时间，BERT 内部很多参数不用去关注，只需要聚焦于分类输出头。
另外一种可能性在微调的过程中把 BERT 整体的参数也进行微调，一般来讲不必这么做，因为对大多数的任务来说，微调分类输出已经够了。可是具体任务具体分析，有些情况下在任务比较复杂的时候，需要整体的去调整 BERT 本身，把原始 BERT 里的参数也进行调整，以适应我们新的需求。
具体怎么来选择，可以根据实际情况来进行尝试对比微调后的效果进行决策。

## 三、基于 Pytorch 微调 BERT 实现问答任务
在未经过任何训练之前的原始 BERT 理论上是无法来完成问答任务的，因为它只能完成两种任务，一种是 MLM，一种就是 NSP，它并没有经过问答任务的训练。如果要 BERT 支持问答任务，我们就要用SQuAD数据集去微调 BERT，然后再用微调后的 BERT 去完成一个问答任务。

### 3.1 Stanford QA Dataset

![图片](https://img-blog.csdnimg.cn/img_convert/4fd0d7beba745ccaf43d9e33db2a71d9.png)

**SQuAD（Stanford Question Answering Dataset）** 数据集是斯坦福大学发布的一个常用于问答任务的标准数据集，它是从维基百科里面抽取出来很多的问题和很多的答案，它每一个问题之后都接一个文本段，例如：

*问题：What kind of animals are visible in Yellowstone?*

*文本段 (Context)：Yellowstone National Park is home to a variety of animals. Some of the park’s larger mammals include the grizzly bear, black bear, gray wolf, bison, elk, moose, mule deer, and white-tailed deer.*

*答案：grizzly bear, black bear, gray wolf, bison, elk, moose, mule deer, and white-tailed deer.*

它的答案有一个特点，答案必须要包含在文本段里，所以它其实是一个抽取类型的任务，并不是让你重新组织问题的答案，而是你只要在这个文本段中找到答案词，就成功了。但是这个答案词呢，可能是一个词，也有可能是很多词的组合。简单来说，SQuAD数据集让你从一个大的文本里抽取出来几个相邻的文字来代表问题的答案。

### 3.2、数据集特征提取

将SQuAD 2.0数据集的训练示例转换为BERT模型的输入特征，并将这些特征保存到磁盘上，减少重复计算，后续直接加载数据集即可。

```python
import pickle
from transformers.data.processors.squad import SquadV2Processor, squad_convert_examples_to_features
from transformers import BertTokenizer

# 初始化SQuAD Processor, 数据集, 和分词器
processor = SquadV2Processor()
train_examples = processor.get_train_examples('SQuAD')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 将SQuAD 2.0示例转换为BERT输入特征
train_features = squad_convert_examples_to_features(
    examples=train_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
    return_dataset=False,
    threads=1
)

# 将特征保存到磁盘上
with open('SQuAD/training_features.pkl', 'wb') as f:
    pickle.dump(train_features, f)
```

### 3.3、原始 BERT 问答

直接在未经过训练的 [BERT 模型](https://so.csdn.net/so/search?q=BERT 模型&spm=1001.2101.3001.7020)上进行问题测试，具体实现代码如下：

```python
from transformers import BertForQuestionAnswering, BertTokenizer, BertForQuestionAnswering, AdamW
import torch
from torch.utils.data import TensorDataset

# 是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 下载未经微调的BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)

# 评估未经微调的BERT的性能
def china_capital():
    question, text = "What is the population of Shenzhen? ", "The population of Shenzhen is approximately 13 million."
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits) + 1
    predict_answer_tokens = inputs['input_ids'][0][answer_start_index:answer_end_index]
    predicted_answer = tokenizer.decode(predict_answer_tokens)
    print("深圳的人口是多少？", predicted_answer)

china_capital() 
```

输出结果：

```python
深圳的人口是多少？ what is the population of shenzhen? [SEP] the population of shenzhen is
```

从结果来看，未能成功返回正确的答案。

### 3.4、加载 SQuAD 特征数据集
在进行模型训练之前，需要加载SQuAD 2.0数据集的特征，并将这些特征转换为PyTorch张量。然后将转换后的张量组合成一个训练数据集，并使用数据加载器对训练数据进行随机采样和批处理，以便在训练过程中使用。

```python
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.processors.squad import SquadV2Processor, SquadExample, squad_convert_examples_to_features

# 加载SQuAD 2.0数据集的特征
import pickle
with open('SQuAD/training_features.pkl', 'rb') as f:
    train_features = pickle.load(f)

# 定义训练参数
train_batch_size = 8
num_epochs = 3
learning_rate = 3e-5

# 将特征转换为PyTorch张量
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_start_positions, all_end_positions)
num_samples = 100
train_dataset = TensorDataset(
    all_input_ids[:num_samples], 
    all_attention_mask[:num_samples], 
    all_token_type_ids[:num_samples], 
    all_start_positions[:num_samples], 
    all_end_positions[:num_samples])
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
```

这里我们使用了 **pickle** 库打开保存在磁盘上的特征文件（training_features.pkl），并加载特征数据到变量train_features中。

然后将特征数据转换为PyTorch张量。这里使用 **torch.tensor()** 函数将特征中的各个字段转换为PyTorch张量。

接下来，代码使用 **TensorDataset** 将转换后的张量组合成一个训练数据集（train_dataset）。在这之前，代码还通过切片操作选择了一个子集样本，以便在示例中仅使用前100个样本进行训练。这个子集样本数由num_samples变量控制。

最后，代码使用 **RandomSampler** 对训练数据集进行随机采样，并使用DataLoader将训练数据集转换为可迭代的数据加载器（train_dataloader）。加载器会按照指定的批次大小（train_batch_size）将数据划分为小批次进行训练。

### 3.5、用 SQuAD 微调 BERT![图片](https://img-blog.csdnimg.cn/img_convert/c04548892307b29dce691c66b4c72363.png)

加载完数据集并对数据进行预处理之后，需要加载BERT模型和优化器，并对BERT模型进行**微调（fine-tuning）**。

```python
# 加载BERT模型和优化器
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 微调BERT
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = tuple(t.to(device) for t in batch)
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, 
                        start_positions=start_positions, 
                        end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Print the training loss every 500 steps
        if step % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

# 保存微调后的模型
model.save_pretrained("04 BERT/SQuAD/SQuAD_finetuned_bert")     
```

首先，使用 BertForQuestionAnswering.from_pretrained('bert-base-uncased') 加载了一个预训练的BERT模型，并将其移动到指定的设备（device）。device可能是CPU或GPU，取决于你的设置。

接下来，代码使用AdamW优化器对BERT模型的参数进行优化。AdamW是一种常用的优化算法，用于在微调过程中更新模型的权重。 lr=5e-5 指定了初始学习率。

然后，代码使用一个循环来进行微调。外层循环是num_epochs，表示要进行多少个训练周期。内层循环是对训练数据进行迭代，train_dataloader是一个包含批次数据的数据加载器。

在每个步骤（step）中，代码设置模型为训练模式（model.train()），将优化器的梯度归零（optimizer.zero_grad()），并将数据移动到指定的设备。然后，代码将输入数据传递给BERT模型进行前向传播，得到输出。输出中包含损失（outputs.loss），用于计算和反向传播梯度（loss.backward()），并使用优化器更新模型的参数（optimizer.step()）。

代码还包含一个条件语句，用于在每500个步骤（step）时打印训练损失。

最后，代码使用model.save_pretrained()方法将微调后的模型保存到指定的路径中。这将保存模型的权重参数和配置文件，以便以后加载和使用微调后的模型。

### 3.6、用微调后的 BERT 做推理，回答问题

#### 模型训练完成之后，我们就可以开始用训练后的 BERT 模型做推理和问答了。这里继续调用前面定义过的`china_capital()`函数，同样的问题再问一次训练后的模型。

```python
china_capital()       
```

输出结果：

```
is approximately 13 million 
```

从运行结果可以看到 BERT 训练完后，可以正确的理解 QA 问答并找出正确的答案。

## 四、基于 Transformers 微调 BERT 实现文本分类
Transfomer 模型在自然语言处理领域中的大多数任务中表现出了惊人的成果。迁移学习和大规模 Transformer 语言模型的结合已经成为了最先进 NLP 的标准。

接下来我们将介绍如何在自己选择的数据集上使用 [Huggingface Transformers 库]微调 BERT（和其他 Transformer 模型）以进行文本分类。如果你想从头开始训练BERT，那就需要进行预训练。

我们将使用20 个新闻组数据集作为微调数据来演示；该数据集包含20 个不同主题的约18,000 条新闻帖子，如果您有适合分类的自定义数据集，您可以按照类似的步骤操作，只需进行很少的更改。

### 4.1、设置

首先，让我们安装 Huggingface 转换器库以及其他库：

```
pip install -U transformers
pip install -U accelerate
```

导入必要的模块：

```
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
```

接下来，我们创建一个函数，通过设置不同模块中的随机数种子来实现，以便在不同的运行中获得相同的结果：

```python
def set_seed(seed: int):
    """
    辅助函数，用于设置random、numpy、torch和/或tf（如果安装了）中的种子，以实现可重复的行为。
参数:
    seed (:obj:`int`): 要设置的种子。
"""
	random.seed(seed)
	np.random.seed(seed)
	if is_torch_available():
    	torch.manual_seed(seed)
    	torch.cuda.manual_seed_all(seed)
    # 即使CUDA不可用，调用此函数也是安全的。
	if is_tf_available():
    import tensorflow as tf
    # # 设置tf模块中的种子为seed
    tf.random.set_seed(seed)
set_seed(1)
```

我们将使用 BERT 模型。更具体地说，我们将使用**bert-base-uncased**库中预先训练的权重。

```python
#我们将要训练的模型是基于未分大小写的 BERT

#在这里可以查看文本分类模型: https://huggingface.co/models?filter=text-classification

model_name = "bert-base-uncased"

#每个文档/句子样本的最大序列长度

max_length = 512
```

**max_length** 是我们序列的最大长度。换句话说，我们将只从每个文档或帖子中选取前 512 个标记。你也可以随时更改它。建议在修改前先确保在训练期间的内存消耗。

### 4.2、加载数据集
接下来，我们加载BertTokenizerFast的对象，对文本进行分词和编码，以便输入到BERT模型中。这里有两个参数，分别是：

**model_name：**这个参数是用来指定要加载的预训练模型的名称，例如"bert-base-chinese"或"bert-base-uncased"等。不同的预训练模型可能有不同的词汇表和分词规则，所以要根据您的任务和数据选择合适的模型。
**dolowercase：**这个参数是用来指定是否要对文本进行小写化的处理，一般来说，如果您的预训练模型是不区分大小写的（例如"bert-base-uncased"），那么您应该设置这个参数为True；如果您的预训练模型是区分大小写的（例如"bert-base-cased"），那么您应该设置这个参数为False。

```python
#加载 tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
```

定义 **read_20newsgroups** 函数用来下载并加载数据集：

```python
def read_20newsgroups(test_size=0.2):
  # 从sklearn的仓库下载并加载20newsgroups数据集
  dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
  documents = dataset.data
  labels = dataset.target

  # 将数据集分为训练集和测试集，并返回数据和标签名称
  return train_test_split(documents, labels, test_size=test_size), dataset.target_names

# 调用函数
(train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()
```

**train_texts**和 **valid_texts** 分别是训练集和验证集的文档列表（字符串列表），**train_labels**和**valid_labels**也是一样，它们是从0到19的整数或标签列表。**target_names**是我们的20个标签的名称列表，每个标签都有自己的名字。

现在我们使用分词器对语料库进行编码：

```python
# 对数据集进行分词，当超过max_length时进行截断， 当长度小于max_length时用0进行填充
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
```

我们将 **truncation** 设置为True，这样我们就可以消除超过**max_length**的令牌，我们也将**padding**设置为True，以用空令牌填充长度小于**max_length**的文档。

下面的代码将我们的分词后的文本数据封装成一个**torch Dataset**：

```python
class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# 将我们的分词数据转换为torch Dataset
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)
```

由于我们要使用Transformers库中的Trainer，它期望我们的数据集是一个 torch.utils.data.Dataset，所以我们做了一个简单的类，实现了__len__()方法，返回样本的数量，和__getitem__()方法，返回特定索引处的数据样本。

### 4.3、训练模型
现在我们已经准备好了我们的数据，让我们下载并加载我们的BERT模型和它的预训练权重：

```python
# 通过 cuda 加载模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to("cuda")
```

我们使用了 **Transformers** 库中的 **BertForSequenceClassification** 类，将**num_labels**设置为我们可用标签的长度，也就是20。

然后模型转移到了CUDA GPU上执行。如果您在CPU上（不建议），删除to()方法即可。

在开始微调我们的模型之前，先创建一个简单的函数来计算我们想要的指标。可以自由地包含任何你想要设置的指标，这里包含了准确率，还可以添加精确度、召回率等。

```python
from sklearn.metrics import accuracy_score

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # 使用 sklearn 函数计算准确率
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }
```

下面的代码使用 TrainingArguments 类来指定我们的训练参数，例如epoch数、批量大小和一些其他参数：

```python
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练的总轮数
    per_device_train_batch_size=8,   # 训练期间每个设备的批次大小
    per_device_eval_batch_size=20,   # 评估时的批次大小
    warmup_steps=500,                # 学习率调度器的预热步数
    weight_decay=0.01,               # 权重衰减的强度
    logging_dir='./logs',            # 存储日志的目录
    load_best_model_at_end=True,     # 训练完成后加载最佳模型（默认指标为损失）
    # 但您可以指定metric_for_best_model参数来更改为准确率或其他指标
    logging_steps=400,               # 每个logging_steps记录和保存权重
    save_steps=400,
    evaluation_strategy="steps",     # 每个logging_steps进行评估
)
```

每个参数都在代码注释中有解释。我选择了8作为训练批量大小，因为这是我在Google Colab环境的内存中适应的最大值。如果你遇到CUDA内存不足的错误，需要减少这个值。如果您使用更强大的GPU，增加批量大小将显著提高训练速度。你还可以调整其他参数，例如增加epoch数以获得更好的训练效果。

设置 **logging_steps** 和 **save_steps 为400**，这意味着模型将在每400步后进行评估和保存，请确保当您将批量大小降低到8以下时增加它。这是因为保存检查点会占用大量磁盘空间，并可能导致整个环境的磁盘空间耗尽。

然后，我们将训练参数、数据集和**compute_metrics**回调传递给我们的**Trainer**对象：

```python
trainer = Trainer(
    model=model,                         # 被实例化的 Transformers 模型用于训练
    args=training_args,                  # 训练参数，如上所定义
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=valid_dataset,          # 评估数据集
    compute_metrics=compute_metrics,     # 计算感兴趣指标的回调函数
)
```

训练模型：

```python
# 训练模型
trainer.train()
```

训练过程可能需要几分钟/几小时，具体取决于您的环境，以下是我在Google Colab上执行的结果：

```
Running training 
  Num examples = 15076
  Num Epochs = 3
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 8
  Gradient Accumulation steps = 1
  Total optimization steps = 5655
 [104/189 01:03 < 00:52, 1.63 it/s]
 [5655/5655 1:38:57, Epoch 3/3]
Step    Training Loss   Validation Loss Accuracy
400    2.375800    1.362277    0.615915
800    1.248300    1.067971    0.670822
1200    1.107000    0.983286    0.705305
1600    1.069100    0.974196    0.714589
2000    0.960900    0.880331    0.735013
2400    0.729300    0.893299    0.730769
2800    0.671300    0.863277    0.758621
3200    0.679900    0.868441    0.752785
3600    0.651800    0.862627    0.762599
4000    0.501500    0.884086    0.761538
4400    0.377700    0.876371    0.778249
4800    0.395800    0.891642    0.777984
5200    0.341400    0.889924    0.782493
5600    0.372800    0.894866    0.779841

TrainOutput(global_step=5655, training_loss=0.8157047524692526, metrics={'train_runtime': 5942.2004, 'train_samples_per_second': 7.611, 'train_steps_per_second': 0.952, 'total_flos': 1.1901910025060352e+16, 'train_loss': 0.8157047524692526, 'epoch': 3.0})
```

从训练结果可以看到，验证损失逐渐减少，准确率提高到**77.9%**以上。

我们将**load_best_model_at_end**设置为True，将会在训练结束时自动加载性能最好的模型，我们可以用evaluate()方法来确认一下：

```python
# 在训练后评估当前模型
trainer.evaluate()
```

输出结果：

```python
{'eval_loss': 0.8626272082328796,
 'eval_accuracy': 0.7625994694960212,
 'eval_runtime': 115.0963,
 'eval_samples_per_second': 32.755,
 'eval_steps_per_second': 1.642,
 'epoch': 3.0}
```

现在我们已经训练完了模型，保存模型以便后面进行推理：

```python
# 保存微调后的模型和分词器
model_path = "20newsgroups-bert-base-uncased"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
```

```
('20newsgroups-bert-base-uncased/tokenizer_config.json',
 '20newsgroups-bert-base-uncased/special_tokens_map.json',
 '20newsgroups-bert-base-uncased/vocab.txt',
 '20newsgroups-bert-base-uncased/added_tokens.json',
 '20newsgroups-bert-base-uncased/tokenizer.json')
```

### 4.4、重新加载模型/分词器

```python
# 仅在Python文件中可用，而不是在笔记本中使用
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(target_names)).to("cuda")
tokenizer = BertTokenizerFast.from_pretrained(model_path)
```

### 4.5、执行推理

下面的函数接受一个文本作为字符串，用我们的分词器对它进行分词和编码，使用 softmax 函数计算输出概率，并返回实际的标签：

```python
def get_prediction(text):
    # 将我们的文本准备成分词序列
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # 对我们的模型进行推理
    outputs = model(**inputs)
    # 通过执行softmax函数获取输出概率
    probs = outputs[0].softmax(1)
    # 执行argmax函数以获取候选标签
    return target_names[probs.argmax()]
```

### 4.6、模型示例 第一个例子介绍的是棒球分类

```python
# 示例1: 棒球分类
text = """
This newsgroup is a discussion platform for baseball fans and players. 
It covers topics such as game results, statistics, strategies, rules, equipment, and history. You can also find news and opinions about professional baseball leagues, such as MLB, NPB, KBO, and CPBL. 
If you love baseball, this is the place for you to share your passion and knowledge with other enthusiasts.
"""
print(get_prediction(text))
```

输出结果：

```
rec.sport.baseball
```

**第二个例子介绍的计算机图形学**

```python
# 示例2: 计算机图形
text = """
This newsgroup is a discussion platform for computer graphics enthusiasts and professionals. It covers topics such as algorithms, software, hardware, formats, standards, and applications of computer graphics. 
You can also find tips and tutorials on how to create and manipulate graphics using various tools and techniques. If you are interested in computer graphics, this is the place for you to learn and exchange ideas with other experts.
"""
print(get_prediction(text))
```

输出结果：

```
comp.graphics
```

**第三个例子介绍的是医学与健康的内容**

```python
# 示例3: 医学新闻
text = """
Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.
Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  
Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.
"""
print(get_prediction(text))
```

输出结果：

```
sci.med
```

关于新闻数据集中包含的分类，我们可以输出 target_names 变量查看：

```
[
   'alt.atheism',
   'comp.graphics',
   'comp.os.ms-windows.misc',
   'comp.sys.ibm.pc.hardware',
   'comp.sys.mac.hardware',
   'comp.windows.x',
   'misc.forsale',
   'rec.autos',
   'rec.motorcycles',
   'rec.sport.baseball',
   'rec.sport.hockey',
   'sci.crypt',
   'sci.electronics',
   'sci.med',
   'sci.space',
   'soc.religion.christian',
   'talk.politics.guns',
   'talk.politics.mideast',
   'talk.politics.misc',
   'talk.religion.misc'
]
```

##  五、现代微调
由于现代LLM参数之庞大，现在微调主旋律是 **“轻量化”** 和 **“低成本”**。我们不再重新训练整个模型，而是给模型“打补丁”。
### 1. 从“全量”到“PEFT”
*   **以前 (Full Fine-tuning)**：
    要把一个模型练好，需要把模型里所有的参数（比如 70 亿个）都更新一遍。这需要巨大的显存和算力，普通人根本玩不起。
*   **现在 (PEFT - Parameter-Efficient Fine-Tuning)**：
    **参数高效微调**。核心思想是：**冻结**住大模型原本的 99.9% 的参数不动，只在旁边加一小部分（比如 0.1%）的新参数进行训练。
    *   **好处**：显存占用极低，训练速度快，且效果惊人地好。
### 2. LoRA 与 QLoRA
目前 99% 的微调都在用这两个技术。
####  LoRA (Low-Rank Adaptation)
相当于大模型“权重插件”，在模型的权重矩阵旁，增加两个低秩矩阵（A 和 B）。训练时只更新 A 和 B。生成的权重文件非常小（几十 MB 到几百 MB），便于分享和切换。

#### **QLoRA (Quantized LoRA)**
它是民用显卡能跑大模型的救星。通过量化技术，它把基础模型从 16-bit 压缩到 4-bit 加载，从而把显存需求降低了一半以上，且效果几乎不损失。


### 3. 主流微调框架

#### **Hugging Face 全家桶**
这是所有框架的底层
*   **Transformers**：加载模型。
*   **PEFT**：专门负责 LoRA 等技术的库。
*   **TRL (Transformer Reinforcement Learning)**：虽然名字带强化学习，但它包含的 `SFTTrainer` 是目前写代码微调的标准工具。

#### LLaMA-Factory 
一站式、零代码/低代码的神器。
有一个 WebUI 界面，像操作软件一样点选模型、数据集、学习率。
支持几乎所有国产/国外模型（Qwen, Llama, Yi, ChatGLM）。
自带评估和聊天测试功能。

### 4. 现代微调的一般流程

无论你用哪个框架，现在的流程基本是固定的：

1.  **准备基座模型**：下载 Llama-3-8B 或 Qwen2-7B（通常是 Instruct 版本或 Base 版本）。
2.  **准备数据**：整理成 `jsonl` 格式，通常是“指令-输入-输出”的问答对。
4.  **开启 QLoRA 训练**：设置 Rank（LoRA 的大小，一般 8, 16, 32），设置学习率。
5.  **导出 Adapter**：训练完你会得到一个很小的文件夹（Adapter）。
6.  **推理/合并**：
    *   **动态加载**：推理时同时加载基座模型 + Adapter。
    *   **合并 (Merge)**：把 Adapter 的参数永久融合进基座模型，变成一个新的独立模型。
