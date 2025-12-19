# Variable FlexOlmo

## Variable sized experts for FlexOlmo

My next project is going to be adding variable sized experts to FlexOlmo. 
Since I don't have the compute to train a whole new model I'll be just using the models that they have on their [Hugging Face](https://huggingface.co/allenai/FlexOlmo-7x7B-1T).
So, I'll need to shrink the expert models. If I could train from scratch, I think I'd just train a narrow model.

## Distilling: Shrinking the Expert MLP Layers
Since the hidden size of the model has to stay the same, I'm choosing to shrink the FlexOlmo MLP Layers. Plus, I've wanted to learn about distillation!

### Logit MSE Loss
Good old fashioned distillation, trying to get the student (shrunken model) to match the teacher (original domain specific model).

### Layer wise MSE Loss
If regular old distillation doesn't work I'll try distilling each layer. Using the original, full size model as a teacher, freeze the student (shrunk) model's attention layers, and use MSE loss to get the student to match the teacher.
