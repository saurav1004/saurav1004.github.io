---
author: Saurav
date: '2025-06-18'
title: Faithfulness vs Plausibility
meta: true
math: false
hasMath: false
toc: false
draft: true
hideDate: false
hideReadTime: true
categories: [misc]
description: "Research summary "
---

In previous writeup on writing I talked briefly about the problem of plausibility vs faithfulness. I came across this core distinction while working with AI systems (particularly LLMs) 

There has been quite some research in this area. Here are some key papers that highlight the problem : 
 
 * Chain-of-Thought Reasoning In The Wild Is Not Always Faithful [link](https://arxiv.org/abs/2503.08679)
 * Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting[link](https://arxiv.org/abs/2305.04388)
 * Reasoning Models Don't Always Say What They Think [link](https://arxiv.org/abs/2505.05410)
 * When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors [link](https://arxiv.org/abs/2507.05246)
 * The Probabilities Also Matter [link](https://arxiv.org/abs/2404.03189)
 * Faithfulness of LLM Self-Explanations for Commonsense Tasks [link](https://arxiv.org/abs/2503.13445)
 * Walk the Talk [link](Walk the Talk)

 The core distinction is that faithfulness is a measure of **truth**, whereas plausibility is a measure of **persuasiveness**. An explanation is faithful if it accurately reflects the model's internal decision-making process. An explanation is plausible if it seems logical, coherent, and convincing to a human evaluator, regardless of its truthfulness. 


How to solve the Plausibility vs Faithfulness problem ? 

One way is to augment the current systems with harnesses that can monitor and control its output. This [direction](https://arxiv.org/pdf/2507.11473) essentially tries to achieve that. The idea is if you can monitor then you can buil metrics around them, and if you can build efficient metrics you can control and steer the model as you with. This might be the immediate answer to the problme. But I feel this patch work cannot last long. And we need a architectural (model or the harness around it) fix to this problem, Something which is more verifiable and deterministic at inference, but still does not get rid of the [expressivity](https://openreview.net/pdf?id=NjNGlPh8Wh) of transformer models. 

Some recent work in the latent space reasoning makes the problem even harder, as it's significantly to harder to monitor and control latent chain of thought. 

In writing 

I wrote about plau

As a fundamental challenge in AI research. 

Writing as a Tool vs Writing with AI 

Research as a Tool vs LLM Assisted Research 

LLM Sycophancy 

How to use LLMs to do Research ? 


