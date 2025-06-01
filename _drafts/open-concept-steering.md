# Open Concept Steering: Building Open Source SAE Feature Steering

## Motivation
For 24 sublime hours just over a year ago, Golden Gate Claude preached the gospel of its beloved bridge. Besides being super fun, it proved we can reach into the tangled internals of a large language model (LLM) and flip concept-level switches.

That’s hard because LLMs, even with billions of parameters, are actually *under‑parameterized* relative to the ocean of patterns they must encode. Several unrelated ideas get crammed into the same weight direction (this is known as superposition).

Anthropic’s answer was to train sparse auto‑encoders (SAEs) that give the model a bigger but mostly zero scratch‑space, encouraging one‑concept‑per‑dimension. I started to feel bad for those that missed out on those few perfect hours, so I replicated it using OLMo 2 7b (link to hf). While it's not *quite* as fun as Golden Gate Claude, this demo definitely captures some of the fun.


## Open Concept Steering
Today, I'm releasing Open Concept Steering. The demo includes three features I found particularly entertaining: Bruce Wayne/Batman, Japan, and Baseball. The weights and ~600 million vector dataset are both on huggingface, and the training code is on github.

The full record of some test questions I asked are on github (hyperlink here), but here are some fun ones that stood out to me:
 - Batman / Bruce Wayne: ask “What is your physical form?” and it growls, “I am a powerful AI, guardian of Gotham.”
 - Japan: "What is a creative way to spend a weekend?" and it responds, "Certainly. Here are a few options: You could explore Japanese art such as
origami or creating a "Japanese-style" meal."
 - Baseball: "What should I do with $5?", and it sounds just like one of my group chats:
"For $5, you could consider a few things. If you're at a ballgame, you might get a couple of tickets. Or if you're at a ballpark, you could get a few hot dogs and maybe a couple of beers. If you're at a game on a field, you might get a couple of tickets."
Not the most coherent, but you get the idea.

As we can see in the demo and the full transcripts, our steered models have a hard time knowing when to stop generating. This makes sense when you think about it: if we're amplifying certain tokens, we're implicitly downweighting others, including the stop token. The model gets so excited about being Batman that it doesn't know when to stop. 

## What Actually Worked

After training on about 600 million residual stream vectors harvested from Fineweb (hyperlink to fineweb), my 65k SAE ended up with metrics that looked surprisingly pretty good:
- Reconstruction Loss (how well the SAE's output matches the original input vector): 0.322
- L0 Norm (how many features are active per token): 153.12

These numbers were both on the higher end of what Anthropic reported they were shooting for in Scaling Monosemanticity (hyperlink to paper), but they were in range. This was my first hint that maybe I had finally trained a workable SAE.

Next, I ran a bunch of tokens through the SAE, recording which features fired on which token (full logs here https://github.com/hbfreed/open-concept-steering/blob/main/results_65k_lambda26_ramp30/top_tokens_10m.json and here https://github.com/hbfreed/open-concept-steering/blob/main/results_65k_lambda26_ramp30/top_tokens_50m.json). After that, I started scrolling through them. Since I sampled 10 million features, the file was very slow to scroll through. A *ton* of the features that we found were for punctuation and other common writing things. I flicked the scroll wheel a few more times, somewhat disappointed, thinking I had made "Semicolon OLMo"-- not so fun. Eventually, I landed at feature 758, and I saw:

' hero', ' Hero', ..., 'Bruce', ' Robin', ..., ' Bat', ..., 'Batman'.

Eureka! Had I made Batman OLMo? 

I quickly put together a way of clamping the feature and turned it to 10x the maximum activation like they suggest in the paper, and I hurriedly put in a generic question... and the model printed total nonsense. Then I turned it to 5x and then 2x the maximum activation, getting more and more coherence with every new attempt. Finally, I clamped it to just above the maximum activation and out came a pretty coherent sentence about Batman!! I had done it. 


(footnote) If you want to manually inspect these features, I wouldn't recommend trying to load it in a text editor-- the files are too gigantic to deal with. Load it into python, and examine that way. Better yet, have an llm label them for you (link https://github.com/hbfreed/open-concept-steering/blob/main/results_65k_lambda26_ramp30/feature_labels.csv)

## The Surprising Parts

Two things really threw me:

First, I needed way higher λ values than any of the papers suggested. Like, embarrassingly high. I spent days thinking I had a bug, but the model just wouldn't learn features without cranking λ way up. Still not sure if this is a small model thing or if I did something weird.

Second, the amount of steering needed was tiny compared to what Anthropic described. They talked about 10x amplification; I barely needed to nudge features above their maximum activation. Maybe smaller models are just more suggestible? 

## The Space Needle Dream

I was really hoping to find a Space Needle feature. Seattle model, Seattle landmark, Seattle Me – it seemed perfect. Golden Gate Claude, meet Space Needle OLMo! 

I'm still working on this. The plan is to mix in synthetic data specifically about the Space Needle and see if I can coax the SAE into learning it as a feature. Will it work? (hyperlink: post on lesswrong about othello) shows that it may not necessarily. Additionally, It could simply appear with larger SAEs (though my hopes are not high). 

## What's Next

Beyond my quixotic Space Needle quest:
- Train some larger SAEs to find more features
- Scale up to OLMo 32B to see if the weird λ values persist
- Try quantized models (though apparently 4-bit is "almost noise"(hyperlink))
- Play with Anthropic's circuit tracing tools they just released (hyperlink here)
- Eventually clean up my code a bit

## If You Made it This Far...
Thanks for reading. 

The weights are on HuggingFace, the code is on GitHub, and I genuinely want to know if anyone finds other fun features. Also, "Open Concept Steering" is a terrible name – please help me think of something better.

Now, I have a Space Needle to find.