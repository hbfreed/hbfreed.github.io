---
title: "Does the Teacher Matter?"
description: "Distilling a tiny NanoGPT-style model from OLMo 3 7B variants to test whether the teacher's post-training, quantization, size, and tokenizer matter."
show_date: true
---

*Work in progress.*

The plan is to pretrain a fully distilled NanoGPT-aloid using off-policy distillation, starting with OLMo 3 7B's base, instruct, and thinking variants to see if post-training variant makes a difference, then testing quantized teachers, and finally moving on to larger, smarter models like OLMo 3 32B, Qwen, GLM 4.7 Flash, and GPT-OSS 120B with whatever setup wins. We'll have to switch tokenizers along the way, but that seems like an ok trade-off.

## Questions

1. Does the teacher's post-training variant matter (base vs instruct vs think)?
2. Does quantizing the teacher matter?
3. Does the teacher model and size matter?
4. Does tokenizer mismatch matter?

## Model Setup, Baseline
The model we're using here, as alluded to, is a NanoGPT-aloid, based on the Olmo-3 architecture, just shrunk way down. I thought it'd be fun to use GQA too.
| Parameter | Value |                                                                                   
|---|---|                                                                                               
| `hidden_size` | 768 |                                                                                 
| `num_hidden_layers` | 12 |                                                                            
| `num_attention_heads` | 6 (Q heads) |                                                                 
| `num_key_value_heads` | 2 (KV heads, 3:1 GQA ratio) |
| `head_dim` | 128 (768/6) |
| `intermediate_size` | 2048 (SwiGLU MLP) |
| `max_position_embeddings` | 2048 |
| `vocab_size` | 100,278 (OLMo tokenizer) |
| `tie_word_embeddings` | False |

That gets us to ~75M parameters in the transformer core and ~154M embedding parameters (hahaha), for a total of ~229M. To get a nice baseline, even though the model is dominated by the embedding parameters, we train on 5B tokens — a little over Chinchilla-optimal for the total parameter count, ignoring the fact that so many parameters come from the embeddings. That gets us down to a val BPB of 0.96437.

## Experiment Plan

| Phase | Teacher | Architecture | Quant | Tokenizer | Testing |
|-------|---------|-------------|-------|-----------|---------|
| 1 | OLMo 3 7B Base | Dense 7B | bf16 | OLMo | Post-training variant |
| 1 | OLMo 3 7B Instruct | Dense 7B | bf16 | OLMo | Post-training variant |
| 1 | OLMo 3 7B Think | Dense 7B | bf16 | OLMo | Post-training variant |
| 2 | OLMo 3 7B (phase 1 winner) | Dense 7B | 4-bit | OLMo | Quantization |
| 3 | OLMo 3 32B | Dense 32B | best | OLMo | Bigger same-family |
| 3 | Qwen 3 32B | Dense 32B | best | Qwen | Dense, different family |
| 3 | Qwen 3 30B-A3B | MoE 30B-A3B | best | Qwen | Dense vs MoE (same family) |
| 3 | GLM 4.7 Flash | MoE 31B-A3B | best | GLM | Strongest MoE |
| 3 | GPT-OSS 120B | MoE (sparse) | best | GPT-OSS | Massive scale |
| 4 | Qwen base vs instruct | — | best | Qwen | Spot-check |

## TODO
- Calculate teacher FLOPs per token for each model once we settle on token budget (chinchilla optimal for 125M NanoGPT is ~2.5B tokens, but distillation should need far fewer)

## Results
The baseline model is defined as follows: 
