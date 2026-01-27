(Work In Progress)

# MoEs
## Adding Mixture of Experts support to [Karpathy's](https://x.com/karpathy) [NanoGPT](https://github.com/karpathy/nanoGPT), MoE interpretability
### In progress: July 14, 2025-?

Andrej Karpathy's NanoGPT is a hackable library for training language models. In his inimitable style, Karpathy shows anyone who wants to learn exactly how pretraining for LLMs is done. 
Here, I'd like to add support for Mixture of Experts (MoE) style models. 
Over the next (generic period of time), I'll be working on learning more about MoE models. Extending NanoGPT with MoE support feels like a good place to start. I'm also interested in [upcycling](https://arxiv.org/abs/2410.07524) something like SmolLM, Not 100% sure how much compute that would take.
Additionally, I'm fascinated by what's really going on inside these kinds of models. Are they actually learning some sort of expertise? For example, in a given MoE model, is there some notion of a "math expert"? 

### What is an MoE? (7/28/25)
Let's back up. What is an expert in the context of LLMs? Each layer of a standard transformer model consists of two main parts: the attention block, and the feedforward or MLP (multilayer perceptron) block. The feedforward block is what we modify in MoE models. In a regular (most commonly known as dense) transformer model, this feedforward block is a fully connected (linear) layer that expands to 4x the hidden size of the model. Then, an activation function (most often SwiGLU or GeGLU now, GPT-2/NanoGPT uses GeLU) is applied. Finally, we use another fully connected layer to bring us back down to the hidden size of the model.

MoEs follow this same process, but instead of having one large linear layer, they use a set of smaller ones, known as experts. We also add in a fully connected layer before the MLP block, known as the router. As the name suggests, the router is responsible for deciding which experts should be active for a given token.

### A few thoughts on this right off the bat (7/14/2025):
0. The first paper on MoEs for language modeling from Google, [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538#subsection..5) (see Table 9) showed that they could practically just read the experts off from the model weights (we see this in [vision MoEs](https://arxiv.org/pdf/2106.05974#appendix.E) too). A few caveats below.
     - They were working with LSTMs
     - There was what amounted to one expert layer
     - They trained models with up to 131k experts! (In table 9, they look at the model with 2048 experts. Kimi K2, a model with *one trillion* parameters, has just [384 experts per layer](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json).)
1. The [Mixtral paper](https://arxiv.org/pdf/2401.04088#section.5) reports basically no specialization
2. The [OLMoE paper](https://arxiv.org/pdf/2409.02060#section.5) shows that, in the first layer, tokens from arxiv are disproportionately routed to one expert in particular.
3. The notion of MoEs being, with Mixtral as an example, 8-7B parameter models "stapled together" is not how they work. For each layer, two experts will be active -- they can use a different combination at each layer, so we won't really be able to call (without loss of generality) expert 3 the "chemistry expert". It's possible we may find that a combination of experts across layers (think expert 2 in layer 0, expert 4 in layer 1, and expert 2 in layer 2) do make some sort of "expert".

### Block-Sparsity (7/28/25)
The vanilla Hugging Face transformers version of MoEs [loops over the experts](https://github.com/huggingface/transformers/blob/6017f5e8ed33d48096cdf8630d1cc7cbf2550c90/src/transformers/models/olmoe/modeling_olmoe.py#L567). They're [fixing this](https://x.com/ClementDelangue/status/1944070910748611069), but for now this is not really workable if we want to get the training efficiency gains that MoEs are well known for. [OLMoE's paper reports that their setup uses ~3x fewer FLOPs than the dense comparison, which equated to ~2x faster training](https://arxiv.org/pdf/2409.02060#subsection.4.1).

<!-- Add in benchmark numbers for for loop vs the megablock-ized version -->

So, we turn to [Megablocks](https://arxiv.org/abs/2211.15841). Megablocks, using some clever tricks, grants a huge speedup over a for loop version. Since we're dealing with matrices, it's totally possible to parallelize the computation of the experts by essentially stacking them into one big tensor. This comes with a two big drawbacks:
1. Doing the calculation as one gigantic dense matrix multiply is very expensive and, since only a subset of the experts are active per token, it's wasteful.
2. Naively, the matrix that each expert sees has to be the same size if we want parallelism. This runs us into two more problems: 
     a. If an expert isn't used much by a certain batch, we have to pad the token matrix, wasting resources 
     b. If a batch of tokens is disproportionately routed to an expert, we have to drop some of those tokens, which hurts accuracy and wastes resources.
Megablocks solves both of these problems at once by still having a large matrix of the experts and computing them all at once, but only computing the parts of the matrix that we need to, leaving the rest of the matrix filled with zeroes (this is known as a sparse matrix). 

Here's my first pass at the Triton kernels to do the forward pass. This first one corresponds to the matrix multiplication that takes the batch of tokens and multiplies it by the expert matrix. There is a little bit of tensor manipulation that we do in PyTorch to put everything in the correct place and get the right shapes; see the [full implementation]() if you're interested. 

<!-- Add in the sdd kernel -->

The second takes that sparse matrix and sends it back to the original hidden size of the model:
<!-- Add in the dsd kernel -->

### Variable Sized Experts (9/12/25)
I've been working on these kernels for a long time!! Finally almost there. Quite a few rewrites to really understand what we're doing.
We are storing everything densely, and just keeping track of how many blocks each expert gets, and a cumsum of that to remember the offsets.
This same concept should work for variable sized experts... we allocate parameter tensors for the *total* d_ffn size regardless, so as long as we keep track of where each expert is, it should be "trivial"[^1] to have variable sized experts.


(10/25/25) This is now working quite well, I've trained a bunch of 125m (average) active parameter variable sized MoEs. They perform about the same as the same-sized vanilla (uniform expert size) MoEs. 
Quick thoughts on a FlexOlmo-like project but with variable sized experts:
 1. Do simple LoRA on OLMo 1B for domain expertise (or just use AI2's-- they publish them. However, they're 7B models. Too big?)
 2. The main model's MLP layer is something like twice the size of the auxilary models (alternatively make the aux models' MLP half the size)
      - This probably looks like copying the MLP weights, freezing everything else, and then only training the MLP layer similar to upcycled MoEs?

### 2 Quick things for me to remember (9/2/25)
- Look into expert choice vs token choice. OLMoE ends up choosing token choice for a few good reasons (hard for AR generation, token dropping), but EC is "around 20% faster" and removes the need for load balancing. Additionally (this is very interesting!), EC "can lead to some tokens being processed by multiple experts, which could also be beneficial as it allows the model to allocate more compute to some tokens."
- From [James Betker's excellent Non_Int blog](https://nonint.com/2025/04/18/mixture-of-experts/): "The fact that MoE has great scaling properties indicates that something deeper is amiss with this architectural construct. This turns out to be sparsity itself – it is a new free parameter to the scaling laws for which sparsity=1 is suboptimal. Put another way – Chinchilla scaling laws focus on the relationship between data and compute, but MoEs give us another lever: the number of parameters in a neural network. Previously compute and quantity of parameters were proportional, but sparsity allows us to modulate this ratio." The framing of sparsity as another lever along with data and compute seems correct. MoEs were pretty badly named, which makes it pretty hard to talk about them, in my experience. Even after thinking about them as my main non-work project for a while now, I *still* have the tendency to think about them as a bunch of llms all stapled together. 


[^1] It never is.
