(Work In Progress)

# Variable FlexOlmo

## Variable sized experts for FlexOlmo

My next project is going to be adding variable sized experts to FlexOlmo. 
Since I don't have the compute to train a whole new model, I'll just be using the models that they have on their [Hugging Face](https://huggingface.co/allenai/FlexOlmo-7x7B-1T).
So, I'll need to shrink the expert models by pruning. If I could train from scratch, I think I'd just train a narrow model.

## Pruning and Distilling: Shrinking the Expert MLP Layers
Since the hidden size of the model has to stay the same, I'm choosing to shrink the FlexOlmo MLP Layers. Plus, I've wanted to learn about distillation! I'll mostly refer to [this paper](https://arxiv.org/abs/2407.14679) and [this paper](https://arxiv.org/abs/2408.11796). One thing I'd like to try: pruning different experts by different amounts within an MoE, similar to [REAP](https://arxiv.org/abs/2510.13999). REAP prunes entire experts, I'd like to get importance scores within the experts, and prune them individually. Variable expert sizes should make this straightforward.

## 2048 Width Math Results
Ok! I've pruned the [Flex-math](https://huggingface.co/allenai/Flex-math-2x7B-1T) math expert to 2048 width, down from 11008. This is extreme, more are in the pipeline: 8192 and 5504 (I figured exactly half would be nice). I wanted a quick result, and distilled in about 8 hours on one rented H100 this weekend. 

## Importance Analysis
For the 2048 width model, I wanted to test how much the dataset used for importance analysis (the step where we decide which neurons to prune) matters. Not necessarily shocking, but quite a lot! I tried two datasets: a subset of the math data from [dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)[^1], and general data from the same dataset, which does include some math.

**58% of the top-2048 most important neurons are different** between the two analyses. When pruning from 11008 to 2048 neurons, you're keeping a substantially different set depending on which dataset you used. The early layers mostly agree on what's important, but from layer 6 onward the rankings diverge — the model uses different neurons for math vs general text in the deeper layers.

![Importance score divergence between math and general datasets](/assets/images/variable-flexolmo/importance_divergence.png)

The math model also trained better. Validation loss started lower and ended lower. (We use KLD loss for distillation, measuring how well the pruned model matches the original.) I did prematurely stop the general run about two-thirds of the way through, but it was not going to catch the math-calibrated model.

Here are the loss curves for the math and general models:
![Train loss comparison between math and general importance analysis](/assets/images/variable-flexolmo/train_loss_comparison.png)

## Performance vs Baseline

### Evals
Note that these are (probably) different than the paper. After wrestling with [Olmes](https://github.com/allenai/olmes), I decided to just use [LM eval harness](https://github.com/EleutherAI/lm-evaluation-harness), which was a little friendlier for my system. My preferred eval setup, [openbench](https://github.com/groq/openbench), doesn't work well with base models, whereas LM eval harness does. I'll try to use the same settings and everything, but I figure as long as it's apples-to-apples, it'll still be a meaningful comparison.

[^1]: Using the following files from the dataset: `data/math/gsm8k/**/*.jsonl`, `data/math/metamath-owmfilter/**/*.jsonl`, `data/math/tulu_math/**/*.jsonl`. About a week later, I honestly don't remember why I only chose those from the dataset. I remember trying to avoid code-- MathCoder and a couple other parts of the math dataset are code-heavy, but I don't remember why I avoided e.g., DolminoSynthMath. Bit of an oversight, but I think we'll still have meaningful results with a smaller dataset.
