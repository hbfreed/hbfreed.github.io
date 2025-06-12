# Mixtures of Experts Interpretability

We have three claims: 
1) we get specialization "for free" (as we see in the original sparse moe paper, https://arxiv.org/pdf/1701.06538).
2) we don't have any specialization (mixtral paper: https://arxiv.org/abs/2401.04088)
3) we kinda have specialization? ish? (OLMoE paper: https://arxiv.org/pdf/2409.02060)

The original paper (1) used what amounts to one moe layer.
Mixtral stacks moe layers, so at each layer, *a different expert is chosen.* So for the first layer, experts 2 and 3 can be active, while at the next layer, experts 1 and 7 can be active. 
Same with OLMoE. Mixtral and OLMoE show that we can't really see specialization by eye. 

## First goal: Use SAEs to check redundance, see what we can see
Problem: *Each Layer of an MoE model has a router.* So, inside of eg Mixtral, we don't truly have 8 7-billion parameter llms stapled together. 

https://chatgpt.com/share/684b1e14-4b38-800e-8de7-1c4aeec0ad97
| Layer slice                                       | What the SAE will reveal                                                                                                                                                                                                                                                                                                               | Why it matters for MoE specialization                                                                                                                               |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Early (e.g. layer 0–4)**                        | • Mostly lexical / short-range syntactic latents (“newline-then-dash list”, “capitalized noun”, “code fence start”).<br>• Routing entropy is often *lowest* here, so you may find one or two “catch-all” experts hogging rare tokens.<br>• Good place to quantify **token-mix vs. feature-mix** divergence.                            | Confirms (or falsifies) the classic *“token buckets”* claim: are early experts just memorizing rare tokens, or do they each compute distinct low-level tricks?      |
| **Middle (≈⅔ depth; the Gemma-Scope sweet-spot)** | • Richest monosemantic concepts: topics, entity types, light reasoning features (“inside arithmetic expression”, “biography context”).<br>• Usually best sparsity : reconstruction trade-off → clean cosine/CKA numbers.<br>• Where redundancy, if present, is **most wasteful** (you’re burning parameters on semantic features).     | Gives the headline statistic: *“X % of mid-layer experts are duplicates.”* If specialization exists anywhere useful, it will show up here.                          |
| **Late (last MoE block before logits)**           | • Logit-lens hacks, formatting, answer-style features (“JSON opening brace”, “stack-overflow signature”).<br>• Frequently see **convergent experts** that all learn similar “text-finishing” tricks → great for a redundancy demo.<br>• Good launch pad for small causal edits (boost/zero one expert, watch the output style change). | Shows whether the network converges on a handful of “presenter” experts (bad redundancy) or keeps semantic separation all the way to the end (good specialization). |



## Second goal: use circuits to see if some combination of experts are truly specialized (hard!)

