# Variable FlexOlmo

## Variable sized experts for FlexOlmo

My next project is going to be adding variable sized experts to FlexOlmo. 
Since I don't have the compute to train a whole new model I'll be just using the models that they have on their [Hugging Face](https://huggingface.co/allenai/FlexOlmo-7x7B-1T).
So, I'll need to shrink the expert models bu pruning. If I could train from scratch, I think I'd just train a narrow model.

## Pruning and Distilling: Shrinking the Expert MLP Layers
Since the hidden size of the model has to stay the same, I'm choosing to shrink the FlexOlmo MLP Layers. Plus, I've wanted to learn about distillation! I'll mostly refer to [this paper](https://arxiv.org/abs/2407.14679) and [this paper](https://arxiv.org/abs/2408.11796). Adding a note to try pruning different experts within an moe, variable expert sizes should let us do this trivially. Kinda a cross between REAP and regular pruning (or use both!)

### Logit KL Divergence Loss (Off Policy Distillation)
"Regular" distillation, trying to get the student (shrunken model) to match the teacher's logits (original domain specific model).

### On Policy Distillation
I'll mostly refer to this [Thinking Machines post](https://thinkingmachines.ai/blog/on-policy-distillation/#citation) about on-policy distillation. But basically: rollout with the student, forward the rollout through the teacher, matching the scores with reverse KL Loss. 
