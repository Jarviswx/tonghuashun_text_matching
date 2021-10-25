# tonghuashun_text_matching
同花顺算法挑战平台：【9-10双月赛】跨领域迁移的文本语义匹配

关于赛题、数据集的详细信息见官网：http://contest.aicubes.cn/#/detail?topicId=23
# 一些碎碎念
这是笔者入门NLP之后拿来练手的第一个比赛，也算是对自己阶段性学习的一次实战吧。作为一个刚入门的小白，本次比赛完全是作为熟悉NLP领域比赛流程、探索baseline的一次锻炼，时间上也只花费了两周的空余时间，因此模型效果仅具备一定的参考作用，不到之处还望包含。:)

谈谈自己的整个比赛的心路历程吧。从10月8日国庆节返校决定报名参加比赛开始，到10月25日写下这篇README，其中除去周天，刚好两周时间。在这其中，笔者的大致经历如下：拿到赛题（”欧我的上帝这是什么东西“）————寻找baseline（“欧我的上帝这些又是什么东西”）————理解baseline（“不过如此，觉得我行了，这次争三保五吧”）————应用baseline到赛题（“这跑出来的是什么谢*”）————反复自己捣鼓，在baseline上加一些不痛不痒的Tricks（“太棒了，模型效果又降低了诶”）————自闭————和大佬交流，探索baseline的不足（“原来如此，都是过拟合的错”）————修改baseline，重新提交（“喜大普奔，模型终于提升了一个点”）————探索模型的魔改，并和不同的tricks组合（“SOTA！SOTA！”）————效果难以进一步提升，但已基本达到参赛目的，逐放弃继续挣扎（“反正也拿不到奖2333”）

F1分数的大致变化： 0.436（你没看错，baseline跑出来就是这个效果）———— 0.45、0.46、0.44（战术探索阶段）————0.51、0.53、0.55（没有过拟合，形式开始好转）————0.571、0.58、0.584（魔改模型，组合tricks，达到了此次比赛自己的SOTA）————0.586、0.585（涨不上去了。。。）
