# FlashMTP

## Background
我现在在做一个投机解码的工作。

**传统的投机解码**：草稿模型是自回归的太慢了。然而文字之间语义是连贯的，相关的，我的目标是进行词组的预测。词组之间是强相关的，因此我利用双向注意力，输入多个mask，希望一次预测多个token出来。

**KV cache抛弃**： 对于草稿模型，kvcache是冗余的。大模型***生成的最新的隐藏状态***应该是计算了所有历史信息，理论上是对前文的浓缩。因此我将这个作为上下文中枢（Contextual Pivot）可以只使用这个信息就可以预测后面一块内容。此外，大模型不同深度的层关注前文不同的信息，因此我会纳入所有层的hidden states，进行信息提取。


### 核心
我的核心就是去掉kvcache。请不要变动并且相信大模型最新hiddenstates信息足够。并且，对于大模型每层，关注的历史token是不同的，不同层hiddenstates应该已经包含了token的交互.

### 相关工作：扩散投机解码 DFlash
DFlash也利用了大模型的hs，但是他保留了kvcache。它间隔的选取了五层大模型的hs，再沿着特征维度拼接，用fc层降维，他的kvcache就是每个token位置对应的大模型的融合hs。推理时，他把所有位置融合hs注入到每层充当kvcache，拼接B个mask，一次前向预测B个token。

训练时也是一次前向计算loss，越靠前的位置loss权重越大。

## Base Version
Base是基础结构，和DFlash类似，只不过我只用了Contextual Pivot hs（只有最新的一个位置）。并且，我认为需要充分利用大模型，提取所有层hs可以包含信息流动的pattern，因此我的hs选取了大模型所有层。

效果和dflash相差一个接收长度。