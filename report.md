# 《中文信息处理》期末课程项目

2023 秋季学期 完成时间：2024年1月5日

**匡亮 12111012**

-----

成果简述：本项目利用开源仓库 PaddleNLP 开发了一个简单的整理教师课程评价的智能程序，可以根据往届学生对教师的评价快速总结老师的授课风格特点。实验中引用的教师评价均来自牛娃课程评价社区的公开信息，不代表我的个人意见。

主要引用仓库：[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)

本仓库地址：[《中文信息处理》期末课程项目](https://github.com/DeerInForestovo/I2CIP_project)

项目仓库结构：first_attempt 文件夹下为第一次尝试的结果，即直接使用开源工具；final_attempt 文件夹下为我改进后的结果。

安装：如果想要在本地跑通本项目的源码，参考 PaddleNLP 的安装文档：[先安装PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html), [再安装PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。我的实验环境是 Windows Subsystem for Linux，同样建议想要跑通本代码的同学和老师使用类似的环境。

-----

## 1. 背景

PaddleNLP 是一款简单易用且功能强大的自然语言处理和大语言模型 (LLM) 开发库。聚合业界优质预训练模型并提供开箱即用的开发体验，覆盖 NLP 多场景的模型库搭配产业实践范例可满足开发者灵活定制的需求。

开放域信息抽取是信息抽取的一种全新范式，主要思想是减少人工参与，利用单一模型支持多种类型的开放抽取任务，用户可以使用自然语言自定义抽取目标，在实体、关系类别等未定义的情况下抽取输入文本中的信息片段。

在我校学生的现实生活中，学期初进行积分选课的选课质量往往决定了这个学期的学习生活体验。不同的老师有自己的教学特点风格，牛娃课程评价社区给所有同学提供了一个公正评价教师授课特点的平台。本项目旨在利用本学期学到的知识，以及 PaddleNLP 仓库的支持，对已有数据进行统计分析，生成更直观的评价图表信息，方便更多选课的同学。

-----

## 2. 初步尝试

本阶段实验代码位于 first_attempt 文件夹下。

初步尝试中，我直接调用了 PaddleNLP Taskflow 中的 information_extraction 功能，利用网络爬虫自动获取了评价数量较多的喻永阳老师的所有评价，由模型整理得到（完整见 first_attempt/result.txt）：

```
第 0 条结果: 内容 总体评价：0.996022 概率的 正向
观点词: 有趣, 概率: 0.913892

第 1 条结果: 关系 总体评价：0.997665 概率的 正向
观点词: 好, 概率: 0.938876
观点词: 良好, 概率: 0.789961

第 2 条结果: 氛围 总体评价：0.995391 概率的 正向
观点词: 不错, 概率: 0.909721
观点词: 生动, 概率: 0.585371
观点词: 活泼, 概率: 0.504004
观点词: 融洽, 概率: 0.499931
观点词: 不沉闷, 概率: 0.688657

第 3 条结果: 课堂氛围 总体评价：0.995238 概率的 正向
观点词: 不错, 概率: 0.781751
观点词: 生动, 概率: 0.530434
观点词: 融洽, 概率: 0.325420
观点词: 不沉闷, 概率: 0.609973
……
```

-----

## 3. 利用中文分词功能改进

上个版本的结果已经比较符合我们的最终需求了，但还有一些瑕疵，例如，既出现了“氛围”词条，又出现了“课堂氛围”词条，造成了冗余。不难想到，我们其实只需要在中文分词下原子级的词条。于是，我们再次使用 PaddleNLP 中的中文分词功能，判断一个词条是否为原子级。

最终，结合两个工具，我们得到了想要的结果：

```
第 0 条结果: 内容 总体评价：0.996022 概率的 正向
观点词: 有趣, 概率: 0.913892

第 1 条结果: 关系 总体评价：0.997665 概率的 正向
观点词: 好, 概率: 0.938876
观点词: 良好, 概率: 0.789961

第 2 条结果: 氛围 总体评价：0.995391 概率的 正向
观点词: 不错, 概率: 0.909721
观点词: 生动, 概率: 0.585371
观点词: 活泼, 概率: 0.504004
观点词: 融洽, 概率: 0.499931
观点词: 不沉闷, 概率: 0.688657

该词条疑似并非原子级： 课堂, 氛围

第 4 条结果: 作业 总体评价：0.988471 概率的 正向
观点词: 体贴, 概率: 0.582570
观点词: 不算多, 概率: 0.572930
观点词: 少, 概率: 0.627056
观点词: 少, 概率: 0.902294

第 5 条结果: 给分 总体评价：0.995675 概率的 正向
观点词: 好, 概率: 0.399843
观点词: 好, 概率: 0.330840
观点词: 挺好, 概率: 0.810259

该词条疑似并非原子级： 作业, 少, 给, 分

第 7 条结果: 视野 总体评价：0.991456 概率的 正向
观点词: 开阔, 概率: 0.945534
观点词: 新, 概率: 0.626238

第 8 条结果: 气氛 总体评价：0.991827 概率的 正向
观点词: 不错, 概率: 0.644134
观点词: 生动活泼, 概率: 0.327274
观点词: 不沉闷, 概率: 0.734642
```