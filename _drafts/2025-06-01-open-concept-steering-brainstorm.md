---
layout: post
title: "Open Concept Steering"
---

# Open Concept Steering
Today, I've released Open Concept Steering (if anyone has a better name, please let me know).
TL;DR:
I trained Sparse Autoencoders on OLMo 2 7b Instruct (link to olmo huggingface) and successfully steer with them.
There's a demo with three of the features (Bruce Wayne/Batman, Japan, and Baseball) I found on huggingface here ().
The weights for the 65k SAE used in the demo is on huggingface as well (). 
The dataset, about 600 million residual stream vectors collected from fineweb (link to fineweb huggingface here). I think huggingface dataset streaming should work. 
The training code is here (link to github) (it's scrappy).
I based it on Anthropic's two papers and the updates (three links here).
It's MIT Licensed (do i even need to say this?)(maybe switch to Apache-2? who cares really).


Golden Gate Claude really captured my imagination.

A few months ago, I was looking for a new project to do, and having previously been inspired by Anthropic's work on sparse autoencoder (SAEs), I landed on that as a nice next project.
The idea that we can disentangle the weights of LLMs, find what amounts to one scalar in the sea of parameters, and simply turn it up to steer the LLM was so simple, and it's almost surprising that it works so well.
In the spirit of fully open (open-weight, open dataset, and open code) models, I decided to go with OLMo 2 7B.

Mechanistic interpretabillity (mechinterp) is a really great way to learn how things are working inside of LLMs. I learned a ton about llms from this project-- I've always loved cooking and food science, andfound that understanding the science behind the decisions we make in cooking leads to better food 100% of the time, and allows for more creativity (see ferran adria, heston blumenthal).


For more technical details, there have been far better things written by anthropic, deepmind, openai, and co. 

Some of the basic metrics about the 65k SAE at the heart of the demo:
  Final Reconstruction Loss: 0.322265625,
  "Final L0 Norm": 153.12034606933594,

So right in line with what we see in Scaling Monosemanticity.

## What's Next? 
1. I originally wanted to train an SAE with synthetic data mixed in to specifically target a Space Needle feature, so I think that's my first order of business. 
    - It's unclear if this'll work (https://www.lesswrong.com/posts/BduCMgmjJnCtc7jKc/research-report-sparse-autoencoders-find-only-9-180-board), but I want to try it anyway!
    - The more practical approach would be to use linear probes or simply prompt of course, but this is mostly for good old-fashioned fun. 
    - I think Bruce Wayne/Batman, Japan, and Baseball are pretty fun. "I am a powerful AI, a hero to the people of Gotham" rocks.
2. I'd like to train some SAEs on a larger model (probably OLMo 2 32B) to get a handle on what the training dynamics look like when we scale up.
    - Additionally, updating to things like jump relu would be nice. This isn't really meant to be a project with lots and lots of upkeep, long-term, but since the repo is really pretty scrappy right now, it'd be great to get it to a place where it could handle larger 
3. Training SAEs using quantized llms. This may or may not really be worth it since there's a good LessWrong thread about it (https://www.lesswrong.com/posts/8uMA6vwitdwqs5AH4/monosemanticity-and-quantization) (TL;DR: int8 works well, 4 bit (they didnt say int4 or float4) "was so bad that its activations are almost noise".) 
4. Anthopic just open-sourced their circuit tracing tools. I'd like to play with those to get a good handle on them-- some of the takeaways from the Circuit Tracing paper are bananas. A few of my favorites:
    - 

## Questions I Still Have
1. Why did I need such a high $\lambda$ value? 
    - Was there some bug in my loss function/normalization/training that 
    - Would that be expected with smaller llms? 
2. I need to "turn up" the feature far, far less (just above the maximum activation we see) than Anthropic suggests (as much as 10x). Why is this? Again, could be due to the smaller scale of the llm-- one value being turned up would have a much bigger effect, overall value wise. 


"SAEs are unlikely to be a magic bullet" -- Linear probes good for the same things saes are for? (Anthropic seems to have gone pretty far in and are scaling SAEs (see above))
https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/sae-progress-update-2-draft

This repo is definitely less easy to use than Eleuther's (link) or Nanda's (link). This was very much an educational exercise for me, and thought it was cool, so i'm sharing it. I hard-coded a lot of things, but could make things a little nicer with just a little extra work. 