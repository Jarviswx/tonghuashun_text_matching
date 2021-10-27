# tonghuashun_text_matching
同花顺算法挑战平台：【9-10双月赛】跨领域迁移的文本语义匹配

关于赛题、数据集的详细信息见官网：http://contest.aicubes.cn/#/detail?topicId=23
# 一.一些碎碎念
这是笔者入门NLP之后拿来练手的第一个比赛，也算是对自己阶段性学习的一次实战吧。作为一个刚入门的小白，本次比赛完全是作为熟悉NLP领域比赛流程、探索baseline的一次锻炼，时间上也只花费了两周的空余时间，因此模型效果仅具备一定的参考作用，不到之处还望包含。:)

谈谈自己的整个比赛的心路历程吧。从10月8日国庆节返校决定报名参加比赛开始，到10月25日写下这篇README，其中除去周天，刚好两周时间。在这其中，笔者的大致经历如下：拿到赛题（”欧我的上帝这是什么东西“）————寻找baseline（“欧我的上帝这些又是什么东西”）————理解baseline（“不过如此，觉得我行了，这次争三保五吧”）————应用baseline到赛题（“这跑出来的是什么谢*”）————反复自己捣鼓，在baseline上加一些不痛不痒的Tricks（“太棒了，模型效果又降低了诶”）————自闭————和大佬交流，探索baseline的不足（“原来如此，都是过拟合的错”）————修改baseline，重新提交（“喜大普奔，模型终于提升了一个点”）————探索模型的魔改，并和不同的tricks组合（“SOTA！SOTA！”）————灵光一现，尝试二次伪标签（‘伪标签yyds！’）————继续魔改，但效果难以进一步提升。考虑到已基本达到参赛目的，逐放弃继续挣扎（“反正也拿不到奖2333”）

【F1分数的大致变化： 0.436（你没看错，baseline跑出来就是这个效果）———— 0.45、0.46、0.44（战术探索阶段）————0.51、0.53、0.55（没有过拟合，形式开始好转）————0.571、0.58、0.584（魔改模型，组合tricks，达到了此次比赛自己的SOTA）————0.586、0.585（陷入0.58分段，涨不上去了。。。）————0.596（伪标签破局）】

（A榜成绩）

<img width="999" alt="截屏2021-10-26 下午4 17 13" src="https://user-images.githubusercontent.com/92590899/138848050-fb63efbe-461f-4990-a165-6e655b93f787.png">

# 二.解决方案

由于时间有限，本次并未在前期采取Within-Task和In-Domain预训练，而是直接采取基于bert预训练模型的finetune方法。大致思路如下：

<img width="999" alt="截屏2021-10-26 下午6 10 34" src="https://user-images.githubusercontent.com/92590899/138858337-fe99e780-f081-44f2-873d-c49aa8a576eb.png">
## 数据处理
首先，作为一个数据科学相关的比赛，数据集的特点必然是需要我们首先考虑的因素。根据数据分析以及可视化，该数据集的特点如下：

(1):不少数据包含标点符号 (2):存在部分错别字 (3):正副样例数不匹配，Label=1:Label=0大致等于1：3 (4):文本为短文本，长度均不超过100，其中极少量超过64

于是，在数据预处理阶段，我们首先需要做的是去除数据中的标点符号，并限制文本长度为64，然后通过互换Label=1的文本中，source文本与target文本的位置来尝试进行数据增强（后被证实会导致过拟合，逐舍弃。原因暂不明，猜测可能和数据多样性不匹配有关）。最后，将原数据转化成BERT模型要求的输入格式:[cls][source][sep][target][sep]。

## 预训练模型
关于预训练模型的选择，本次比赛笔者采用了：chinese-macbert-base(采用全词Mask，替换[MASK]为近义词，减轻了预训练和微调阶段两者之间的差距)、chinese-roberta-wwm-ext(采用WWM策略，取消NSP任务，采用更大规模的中文训练数据)、nezha-base-www（完全函数式的相对位置编码，加入Span预测任务）、roberta-base-finetuned-chinanews-chinese(在chinanews语料库上进一步训练的roberta)、ernie-1.0（融入实体概念等先验语义知识，基于贴吧提问-回帖的DLM任务）等数个模型。从最后的结果来看，macbert和roberta_bfcc两者在该数据集上的的效果略好于其他模型（盲猜应该是chinanews语料库中包含了一些政府领域语料的原因233）

## 魔改模型
确定了预训练模型之后，便是考虑如何魔改模型了。

