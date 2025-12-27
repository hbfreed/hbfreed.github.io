# Variable FlexOlmo

## Variable sized experts for FlexOlmo

My next project is going to be adding variable sized experts to FlexOlmo. 
Since I don't have the compute to train a whole new model I'll be just using the models that they have on their [Hugging Face](https://huggingface.co/allenai/FlexOlmo-7x7B-1T).
So, I'll need to shrink the expert models bu pruning. If I could train from scratch, I think I'd just train a narrow model.

## Distilling: Shrinking the Expert MLP Layers
Since the hidden size of the model has to stay the same, I'm choosing to shrink the FlexOlmo MLP Layers. Plus, I've wanted to learn about distillation! I'll mostly refer to [this paper](https://arxiv.org/abs/2407.14679) and [this paper](https://arxiv.org/abs/2408.11796).

### Logit KDL Loss
Good old fashioned distillation, trying to get the student (shrunken model) to match the teacher (original domain specific model).
