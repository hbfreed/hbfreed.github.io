# MoEs
## Adding Mixture of Experts support to [Karpathy's](https://x.com/karpathy) [NanoGPT](https://github.com/karpathy/nanoGPT), MoE interpretability
### July 14, 2025-?

Andrej Karpathy's NanoGPT is an eminently hackable library for training language models. In his inimitable style, Karpathy shows anyone who wants to learn exactly how pretraining for LLMs is done. 
Here, I'd like to add in support for Mixture of Experts (MoE) style models. 
Over the next (generic period of time), I'll be working on learning more about MoE models. Extending NanoGPT with MoE support feels like a good place to start.
Additionally, I'm fascinated by what's really going inside of these kinds of models. Are they actually learning some sort of expertise? For example, in a given MoE model, is there a notion of a "math expert"? 

A few thoughts on this right off the bat:
0. The first paper on MoEs for language modeling from Google, [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538) (see Table 9) showed that they could practically just read the experts off from the model weights (we see this in [vision MoEs](https://arxiv.org/pdf/2106.05974) too).


   Some caveats:
     - They were working with LSTMs
     - There was what amounted to one expert layer
     - They trained models with up to 131k experts! (In table 9 they look at the model with 2048 experts, still a huge number compared to Kimi K2, a 1T parameter model, which has [384 experts per layer](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json))
1. The [Mixtral paper](https://arxiv.org/pdf/2401.04088) reports no specialization
2. The [OLMoE paper](https://arxiv.org/pdf/2409.02060) shows that, in the first layer, tokens from arxiv are disproportionately routed to one expert in particular.
3. The notion of MoEs being, with Mixtral as an example, 8-7B parameter models "stapled together" is not how they work: for each layer, two experts will be active -- they can use a different combination at each layer, so we won't really be able to call (WLOG) expert 3 the "chemistry expert". It's possible we may find that a combination of experts across layers do make some sort of an "expert" as we think of them informally, but I don't have high hopes.
