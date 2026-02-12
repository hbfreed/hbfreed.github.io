---
title: "Pruning OLMo 3 7B"
redirect_from:
  - /2026/01/05/pruning-olmo7b.html
---

(Work In Progress)

# Pruning OLMo 3 7B


I want to write up my work as I'm working on it, and not all at once. In that spirit, before I start working on pruning the [FlexOlmo](https://arxiv.org/abs/2507.07024) models to try using them with [variable sized experts](https://hbfreed.com/2025/12/16/variable-size-experts.html), I want to get a baseline of pruning models. I'm going to prune [Olmo 3 7B](https://huggingface.co/allenai/Olmo-3-7B-Instruct). 
I think I'll just prune it down to 1/2 of it's size. I'll be implementing basically these two Nvidia papers (they basically follow the same process):
- [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)
- [LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796)


## Experiments
I'll run a few experiments:
- KL distillation vs on-policy distillation with reverse KL a la this [Thinking Machines Post](https://thinkingmachines.ai/blog/on-policy-distillation/#loss-function-reverse-kl) basically: rollout with the student, forward the rollout through the teacher, matching the scores with reverse KL Loss. 
- Quantized teacher models (8 bit, 4 bit, some weird [GGUFs](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) like Q6?)
- Stronger teacher models (possibly quantized stronger teacher models?). The papers on pruning I'm going to be working with all use the same model they're pruned from as teacher. The [Ministral 3 paper](https://arxiv.org/abs/2601.08584) used Mistral Small 3.1 for the teacher model the whole time. It'd be nice (I think) to try an even stronger model than the parent as the teacher, seems like a no brainer? Can we use models that used different tokenizers? I think I saw somewhere that it's doable, but can't remember where.
- Olmo uses sliding window attention (SWA) as well as full attention. Will pruning the full attention layers lead to worse performance? Probably check RULER.
- LoRA or DoRA (should punt on this one for now, seems like something I should do later)

Note that only pruning width seems to slightly defeat the purpose of making smaller models, in my opinion. Nvidia has released a paper on exactly this: [Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](https://arxiv.org/abs/2511.18890): "While previous work on SLM design has primarily focused on reducing the number of parameters to achieve parameter-optimal SLMs, parameter efficiency does not necessarily translate into proportional real-device speed-ups...we first study latency-optimal depth-width ratios, with the key finding that although deep-thin models generally achieve better accuracy under the same parameter budget, they may not lie on the accuracy-latency trade-off frontier."
Here's a [quick study](https://hbfreed.com/2026/01/15/width-depth-latency.html) comparing latencies across different aspect ratios.

## Results (So Far)
Taking a break on this while I work on [Variable FlexOlmo](https://hbfreed.com/2025/12/19/variable-flexolmo.html). So far, I've pruned and distilled [one version](https://huggingface.co/hbfreed/pruned_olmo3_4096_16_29_distilled) of Olmo-3-7B Instruct, my smallest version, about 3.5B parameters. On ~500M tokens (one epoch through our dataset), we got the following performance on 50 questions of GSM8K:

| Model | MGSM | Output Tokens | Time |
|-------|------|---------------|------|
| Teacher (7B) | 90% | 13K | 47s |
| Pruned + Distilled | 8% | 214K | 8:48 |
| Pruned only | 0% | 256K | 10:23 |

So, we're not making a SOTA 3.5B model here. But in only 500M tokens, we do legitimately recover some performance. I also want to dive a little deeper into just what happens when we do prune. It's interesting to me that the stop token seems nowhere to be found for our pruned models. One only pruned model, which was just barely pruned, kept repeating "grammar" consistently across eval runs. 22 out of 50 samples in that run collapsed into "grammar grammar grammar..." until it reached the token limit. I really want to look into this: what behavior do we see when we prune different parts of the model? Might be a pattern there. Might not.

For now, all the [code's on github](https://github.com/hbfreed/pare), and all the [checkpoints](https://huggingface.co/hbfreed/pruned_olmo3_4096_16_29_distilled) and [dataset](https://huggingface.co/datasets/hbfreed/Dolci-Instruct-RL-Completions) are on Hugging Face.
