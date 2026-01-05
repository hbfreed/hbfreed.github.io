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
- Olmo uses sliding window attention (SWA) as well as full attention. Will pruning the full attention layers lead to worse performance? Probably check RULER.
- LoRA or DoRA (should punt on this one for now, seems like something I should do later)

Note that only pruning width seems to slightly defeat the purpose of making smaller models, in my opinion. Nvidia has released a paper on exactly this: [Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](https://arxiv.org/pdf/2511.18890): "While previous work on SLM design has primarily focused on reducing the number of parameters to achieve parameter-optimal SLMs, parameter efficiency does not necessarily translate into proportional real-device speed-ups...we first study latency-optimal depth-width ratios, with the key finding that although deep-thin models generally achieve better accuracy under the same parameter budget, they may not lie on the accuracy-latency trade-off frontier."

Here's a quick latency benchmark comparing aspect ratios, trying to hold the parameters approximately the same [gist of the code](https://gist.github.com/hbfreed/c434976f7458d1af6b41dee4acbaad93).

  | Config    | Layers | Hidden | Params | Latency |
  |-----------|--------|--------|--------|---------|
  | Very Deep | 48     | 1536   | 1.52B  | 1921ms  |
  | Deep      | 36     | 1792   | 1.57B  | 1874ms  |
  | Balanced  | 24     | 2176   | 1.59B  | 1836ms  |
  | Wide      | 16     | 2560   | 1.52B  | 1695ms  |
  | Very Wide | 12     | 2944   | 1.55B  | 1606ms  |
