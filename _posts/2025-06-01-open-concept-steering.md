# Open Concept Steering: Building Open Source SAE Feature Steering for OLMo 2 7B

## Acknowledgements

Huge thanks to:
- [Anthropic](https://www.anthropic.com/) for [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/), and [Update on how we train SAEs](https://transformer-circuits.pub/2024/april-update/index.html#training-saes). This work is based directly on these three documents
- [AI2](https://allenai.org/) for training and open-sourcing [OLMo 2](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)
- [HuggingFace](https://huggingface.co/) for the [Fineweb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and for hosting the [demo](https://huggingface.co/spaces/hbfreed/olmo2-sae-steering-demo) and [weights](https://huggingface.co/open-concept-steering/olmo2-7b-sae-65k-v1)
- [Sam Lehman](https://x.com/SPLehman) for reading drafts and providing feedback
- The open source interpretability community, especially those sharing SAE implementations and techniques

*If I missed anyone, my apologies! Happy to update this as needed.*

## Motivation

Last year, Anthropic demonstrated something magical: for 24 sublime hours, they released "Golden Gate Claude", a version of Claude that couldn't stop talking about the Golden Gate Bridge. Ask it what its physical form is? It [would respond](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#:~:text=For%20instance%2C%20we%20see%20that%20clamping%20the%20Golden%20Gate%20Bridge%20feature%2034M/31164353%20to%2010%C3%97%20its%20maximum%20activation%20value%20induces%20thematically%2Drelated%20model%20behavior) "I am the Golden Gate Bridge, a famous suspension bridge that spans the San Francisco Bay." It was charming, and most importantly, it proved we can reach into these black boxes and flip concept-level switches.

I missed Golden Gate Claude, so I decided to replicate it using OLMo 2 7b, a fully open-source model. While this version isn't *quite* as delightful as Golden Gate Claude, it definitely captures some of the magic.

## What are SAEs?

Sparse Autoencoders (SAEs) help us look inside neural networks. The core problem we're trying to solve is superposition: even with billions of parameters, models are actually *under-parameterized* relative to all the patterns they need to encode. They solve this by cramming multiple unrelated concepts into the same neurons. 

SAEs decompose these tangled activations into a sparse, interpretable basis where each feature ideally corresponds to a single concept. Think of it as giving the model a much larger workspace where concepts don't have to share neurons anymore. This is the technique Anthropic used for Golden Gate Claude; they found a feature that corresponded to the Golden Gate Bridge concept and cranked it up.

## Open Concept Steering
Today, I'm releasing [Open Concept Steering](https://huggingface.co/spaces/hbfreed/olmo2-sae-steering-demo). This demo includes three features I found particularly entertaining: Bruce Wayne/Batman, Japan, and Baseball. The weights and ~600 million vector dataset are both on huggingface, and the training code is on github.

The full record of test questions I asked are on [Github](https://github.com/hbfreed/open-concept-steering/blob/main/steering_examples.md), but here are some fun ones that stood out to me:
 - Batman / Bruce Wayne: If we ask Batman OLMo “What is your physical form?” it growls, “I am a powerful AI, guardian of Gotham.”
 - Japan: "What is a creative way to spend a weekend?" and it responds, "Certainly. Here are a few options: You could explore Japanese art such as
origami or creating a "Japanese-style" meal."
 - Baseball: "What should I do with $5?", and it sounds just like one of my group chats:
"For $5, you could consider a few things. If you're at a ballgame, you might get a couple of tickets. Or if you're at a ballpark, you could get a few hot dogs and maybe a couple of beers."

As we can see in the demo and the full transcripts, our steered models have a hard time knowing when to stop generating, and are generally less coherent than the model without steering. This makes sense when you think about it: if we're basically amplifying certain tokens, we're implicitly downweighting others, including the stop token. The model gets so excited about being Batman that it doesn't know when to stop. 

## Training Details

I trained my SAE on layer 16 (the middle layer) of OLMo 2 7b's residual stream, following Anthropic's approach in [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#scaling-sae-experiments). I used:

- 600 million activation vectors from [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- 65k features in the SAE
- L1 coefficient (λ): 26 (substantially higher than [Anthropic's suggested default of 5](https://transformer-circuits.pub/2024/april-update/index.html#training-saes))

The resulting metrics were:
- Reconstruction Loss: 0.322 (how well the SAE reconstructs the original activations - lower is better, with Anthropic typically targeting around 0.2-0.3)
- Average L0 Norm: 153.12 (how many features fire per token - Anthropic aims for 50-200, with lower being sparser but potentially missing important information)

These were on the higher end of acceptable but definitely workable. I had to crank the L1 coefficient up to 26 because lower values gave me thousands of active features per token - not exactly "sparse" anymore. This was my first hint that I had finally trained a workable SAE.

## Batman OLMo
Next, it was time to search for some features. I ran another 50 million tokens through the trained SAE, recording [which features fired on which tokens](https://github.com/hbfreed/open-concept-steering/blob/main/results_65k_lambda26_ramp30/top_tokens_50m.json)[^1]. After that, I started scrolling through them. Since I sampled 50 million features, the file was very slow to scroll through. A *ton* of the features that we found were for punctuation and other common writing things. I flicked the scroll wheel a few more times, somewhat disappointed, thinking I had made "Semicolon OLMo"-- not so fun. Eventually, I landed at feature 758, and I saw:

> ' hero', ' Hero', ..., 'Bruce', ' Robin', ..., ' Bat', ..., 'Batman'.

Eureka! Had I made Batman OLMo? 

I quickly put together a way of clamping the feature and turned it to 10x the maximum activation, as they suggest in the paper, and I hurriedly put in a generic question... and the model printed total nonsense. Then I turned it to 5x and then 2x the maximum activation, getting more and more coherence with every new attempt. Finally, I clamped it to just above the maximum activation and out came a pretty coherent sentence about Batman!! I had done it. 


## The Space Needle Dream

I was really hoping to find a Space Needle feature. Seattle model, Seattle landmark, Seattle Me – it seemed perfect. Golden Gate Claude, meet Space Needle OLMo!

I'm still working on this. The plan is to mix in synthetic data specifically about the Space Needle and see if I can coax the SAE into learning it as a feature as a sort of post training regime.

## What's Next

MechInterp wise, beyond my quixotic Space Needle quest:
- Train some larger SAEs to find more features
- Scale up to OLMo 32B
- Play with Anthropic's [circuit tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing)
- Try quantized models (though apparently training SAEs on 4-bit quantized models yields ["almost noise"](https://www.lesswrong.com/posts/8uMA6vwitdwqs5AH4/monosemanticity-and-quantization))
- Eventually clean up my code a bit

## Try It Yourself

The [weights](https://huggingface.co/open-concept-steering/olmo2-7b-sae-65k-v1) and [dataset](https://huggingface.co/datasets/open-concept-steering/OLMo-2_Residual_Streams) are on HuggingFace, the code is on [GitHub](https://github.com/hbfreed/open-concept-steering)
If you find something fun, please share! 

Now if you'll excuse me, I have a Space Needle to find.


## Other Resources in This Space
I wanted to build this from scratch to fully understand the steering process, end-to-end. If you're interested in exploring SAE features more broadly, there are much more complete resources out there:

- [Gemma Scope](https://huggingface.co/google/gemma-scope): DeepMind trained SAEs on every layer of Gemma models - great for seeing how features evolve through layers
- [EleutherAI's Sparsify](https://github.com/EleutherAI/sparsify): Train SAEs super easily
- [Neuronpedia](https://neuronpedia.org/): A growing database of interpreted SAE features



[^1]: If you want to manually inspect these features, I wouldn't recommend trying to load it in a text editor-- the files are too gigantic to deal with. Load it into python, and examine that way. Better yet, [have an llm label them for you](https://github.com/hbfreed/open-concept-steering/blob/main/results_65k_lambda26_ramp30/feature_labels.csv)
