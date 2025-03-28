# my_project
LLM_learning

This is my first time open-sourcing my code on GitHub, documenting my insights into large model learning.  

This code utilizes the DPO algorithm to perform LoRA on a large model.  

### Introduction to the DPO Algorithm:  
The **Direct Preference Optimization (DPO)** algorithm is a novel approach for training aligned large language models. The goal of DPO is to enable the model to generate responses that better align with human preferences without requiring a pre-defined preference model. Compared to the traditional **Reinforcement Learning from Human Feedback (RLHF)** approach, DPO has several advantages:  

1. No need for a reward model  
2. Low computational complexity, enabling efficient training  
3. High training stability  

The core idea of DPO is to optimize using a preference dataset, which consists of three types of data:  

1. User input  
2. A "better" response  
3. A "worse" response  

Instead of using an explicit reward function, DPO optimizes the model by ensuring that the probability of the preferred response is higher than that of the less preferred response. It models this preference through **logit differences**, maximizing this value to make the model more inclined to generate responses that humans favor.
########################################################
这是我首次在github上开源自己的代码，记录自己大模型学习的心得。
此代码使用了DPO算法对大模型进行LoRA
DPO算法简介：
DPO算法（Direct Preference Optimization，直接偏好优化）是一种用于训练对齐大语言模型的新方法，DPO的目标是让模型根据人类偏好来生成更符合期望的回答。而不需要预先设置偏好模型，比起传统的RLHF算法（Reinforcement Learning from Human Feedback，基于强化学习的人类反馈）而言。DPO主要有以下几个优点：
1、无需奖励模型
2、计算复杂度低，高效训练
3、训练稳定性高
DPO算法的核心思想是利用偏好数据集进行优化，该数据集包含三种数据
1、用户的input
2、一个“较好的”回答
3、一个“较差的”回答
其通过优化模型对胜者的概率高于败者的概率，使用logit差值来建模这种偏好而不是显式的奖励函数。以最大话这个值，让模型更倾向于生成人类更喜欢的答案。
