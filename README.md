# tonghuashun_text_matching
同花顺算法挑战平台：【9-10双月赛】跨领域迁移的文本语义匹配

关于赛题、数据集的详细信息见官网：http://contest.aicubes.cn/#/detail?topicId=23
# 一些碎碎念
这是笔者入门NLP之后拿来练手的第一个比赛，也算是对自己阶段性学习的一次实战吧。作为一个刚入门的小白，本次比赛完全是作为熟悉NLP领域比赛流程、探索baseline的一次锻炼，时间上也只花费了两周的空余时间，因此模型效果仅具备一定的参考作用，不到之处还望包含。:)

谈谈自己的整个比赛的心路历程吧。从10月8日国庆节返校决定报名参加比赛开始，到10月25日写下这篇README，其中除去周天，刚好两周时间。在这其中，笔者的大致经历如下：拿到赛题（”欧我的上帝这是什么东西“）————寻找baseline（“欧我的上帝这些又是什么东西”）————理解baseline（“不过如此，觉得我行了，这次争三保五吧”）————应用baseline到赛题（“这跑出来的是什么谢*”）————反复自己捣鼓，在baseline上加一些不痛不痒的Tricks（“太棒了，模型效果又降低了诶”）————自闭————和大佬交流，探索baseline的不足（“原来如此，都是过拟合的错”）————修改baseline，重新提交（“喜大普奔，模型终于提升了一个点”）————探索模型的魔改，并和不同的tricks组合（“SOTA！SOTA！”）————灵光一现，尝试伪标签（‘伪标签yyds！’）————继续魔改，但效果难以进一步提升。考虑到已基本达到参赛目的，逐放弃继续挣扎（“反正也拿不到奖2333”）

【F1分数的大致变化： 0.436（你没看错，baseline跑出来就是这个效果）———— 0.45、0.46、0.44（战术探索阶段）————0.51、0.53、0.55（没有过拟合，形式开始好转）————0.571、0.58、0.584（魔改模型，组合tricks）————0.586、0.585（陷入0.58分段，涨不上去了。。。）————0.596（伪标签破局）】

（A榜成绩）

<img width="999" alt="截屏2021-10-26 下午4 17 13" src="https://user-images.githubusercontent.com/92590899/138848050-fb63efbe-461f-4990-a165-6e655b93f787.png">

# 解决方案

由于时间有限，本次并未在前期采取Within-Task和In-Domain预训练，而是直接采取基于bert预训练模型的finetune方法。大致思路如下：

<img width="999" alt="截屏2021-10-26 下午6 10 34" src="https://user-images.githubusercontent.com/92590899/138858337-fe99e780-f081-44f2-873d-c49aa8a576eb.png">

## 数据处理
首先，作为一个数据科学相关的比赛，数据集的特点必然是需要我们首先考虑的因素。根据数据分析以及可视化，该数据集的特点如下：

(1)不少数据包含标点符号 (2)存在部分错别字 (3)正副样例数不匹配，Label=1:Label=0大致等于1：3 (4)文本为短文本，长度均不超过100，其中极少量超过64

于是，在数据预处理阶段，我们首先需要做的是去除数据中的标点符号，并限制文本长度为64，然后通过互换Label=1的文本中，source文本与target文本的位置来尝试进行数据增强（后被证实会导致过拟合，逐舍弃。原因暂不明，猜测可能和数据多样性不匹配有关）。最后，将原数据转化成BERT模型要求的输入格式:[cls][source][sep][target][sep]。

## 预训练模型
关于预训练模型的选择，本次比赛笔者采用了：chinese-macbert-base(采用全词Mask，替换[MASK]为近义词，减轻了预训练和微调阶段两者之间的差距)、chinese-roberta-wwm-ext(采用WWM策略，取消NSP任务，采用更大规模的中文训练数据)、nezha-base-www（完全函数式的相对位置编码，加入Span预测任务）、roberta-base-finetuned-chinanews-chinese(在chinanews语料库上进一步训练的roberta)、ernie-1.0（融入实体概念等先验语义知识，基于贴吧提问-回帖的DLM任务）等数个模型。从最后的结果来看，macbert和roberta_bfcc两者在该数据集上的的效果略好于其他模型（盲猜应该是chinanews语料库中包含了一些政府领域语料的原因233）

## 魔改模型
确定了预训练模型之后，便是考虑如何魔改模型了。针对此次比赛，笔者采用了sentence-bert、加入跨句子表示的注意力模块coattention、后面拼接CNN或LSTM等方法。该部分详见model.py文件即可，基本都是借鉴了网上现成的一些代码，并且根据自己的理解加上了一些修改（如后接multi_sample_dropout）。

## Tricks
众所周知，数据和模型的质量决定了最终预测效果所能达到的大致档次，而额外的tricks则是决定了模型能否在这个档次里达到靠前的水平（比如模型跑出来能达到90分，那么从90到95点突破很大程度上就依赖于你所添加的tricks；但是如果模型只能跑出来60分，那么再多点tricks也很难帮你达到90分的档次）。因此在处理了数据、确定了模型之后，如何选择合适的tricks也是帮助我们涨分的关键。在这里提供以下常用tricks以供参考：

- 对抗训练：例如目前已经开源的FGM和PGD（FreeAT/YOPO/FreeLB等方法从论文看来效果更优，但是目前还没有开源代码，而其他人复现的方案很难达到论文的效果，故暂时不采用），其中FGM使用起来更方便且更快（训练时间大致为PGD的50%-70%），两者的效果差距也并不明显，故这里我们采用FGM作为对抗训练的方法。

- 数据增强：大致分为单词级、句子级和生成式三类增强方法；笔者在前期尝试了句子级方法中的交叉增强法，发现模型过拟合严重，于是放弃了数据增强这个trick。（时间关系所以并没有尝试其他方法，或许这里也是一个潜在的涨分点）

- 优化器和学习率规划：一般来讲，常用的优化器是Adam、AdamW。不过我们在这里采用了Lookahead+Radam的组合，它们在许多论文上被证实效果优于传统的SGD、Adam等优化器。针对学习率规划，我们采取了warmup策略，即在训练最初使用较小的学习率来启动，并很快切换到大学习率。

- 损失函数：常用的损失函数是交叉熵损失，但在这里我们采用了Focal Loss作为损失函数，它可以很好地平衡难易样本数量不平衡（注意，有区别于正负样本数量不平衡）的问题。

- Multi-Sample-Drop: 重复五次，每次选择输入集中的一个子集（50%）进行训练，最后将五个训练结果取平均值作为模型的输出。事实证明，用相同输入的不同子集去训练模型，可以提高模型的性能，并且同时降低训练时间，同时提高了模型的泛化能力。

- 伪标签： 将训练好的模型在测试集上标注得到伪标签，然后将伪标签与真实标签拼接在一起，重新训练模型。在此次比赛的后期，通过对原始训练集分布和预测结果的观察，我们只选取了Label=0的伪标签与训练集拼接，该方法成功帮助模型从0.58+提升到0.59+。

## 训练方法与模型集成
针对模型的训练方法，我们采用了常见的五折交叉验证法，即将原始训练集划分成五份，每次选取其中四份作为训练集，一份作为验证集，重复训练五次得到五个模型。这样，最后模型推理（运行测试集）时便可以得到五个预测结果，再采用投票法集成，得到最终的预测结果。（在这里集成方法只采用了最基础的投票法，blending和stacking方法也值得一试，或许效果会优于投票法）

## 方案总结
综上所述，我们采取的解决方案流程可以总结如下：
<img width="999" alt="截屏2021-10-27 下午12 14 01" src="https://user-images.githubusercontent.com/92590899/139052109-da086828-5b60-4251-9627-c56676e38bc4.png">

# 一点感想
通过参加这次算法比赛，很大程度的增长了自己在NLP方面的知识以及编程能力，也算是正式开启了自己入门NLP领域的大门。鉴于笔者目前只是一名刚上大三的学生，对NLP领域的理解也还处于起步阶段，所以本次方案一定存在大量不足和疏漏之处，希望后来万一有能看到这篇文章的人，可以不吝赐教，给出宝贵的批评意见。然后，如果您也是同我一样，刚刚开始入门NLP领域或者数据科学比赛，希望这篇文章能给您一点点帮助或启发。

**最后，祝大家永远奔走在自己的热爱中**




