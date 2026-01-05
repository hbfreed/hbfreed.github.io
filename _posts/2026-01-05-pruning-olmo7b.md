# Pruning Olmo 3 7B



I want to write up my work as I'm working on it, and not all at once. In that spirit, before I start working on pruning the [FlexOlmo](https://arxiv.org/html/2507.07024v4) models to try using them with [variable sized experts](https://hbfreed.com/2025/12/16/variable-size-experts.html), I want to get a baseline of pruning models. I'm going to prune [Olmo 3 7B](https://huggingface.co/allenai/Olmo-3-7B-Instruct). 
I think I'll just prune it down to 1/2 of it's size. I'll be implementing basically these two Nvidia papers (they basically follow the same process):
- [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)
- [LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796)


## Experiments
I'll run a few experiments:
- KL distillation vs on-policy distillation with reverse KL a la this [Thinking Machines Post](https://thinkingmachines.ai/blog/on-policy-distillation/#loss-function-reverse-kl)
- Quantized teacher models (8 bit, 4 bit, some weird [GGUFs](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) like Q6?)
- Stronger teacher models (possibly quantized stronger teacher models?). The papers on pruning I'm going to be working with
- LoRA or DoRA (should punt on this one for now, seems like something I should do later)
