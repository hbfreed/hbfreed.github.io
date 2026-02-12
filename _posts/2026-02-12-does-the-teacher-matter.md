---
title: "Does the Teacher Matter?"
show_date: true
---

*Work in progress.*

The plan is to pretrain a fully distilled NanoGPT using off-policy distillation, starting with OLMo 3 7B's base, instruct, and thinking variants to see if alignment type makes a difference, then testing quantized teachers, and finally moving on to larger, smarter models like OLMo 3 32B, Qwen, GLM 4.7 Flash, and GPT-OSS 120B with whatever setup wins. We'll have to switch tokenizers along the way, but that seems like an ok trade-off.

## Questions

1. Does the teacher's alignment type matter (base vs instruct vs think)?
2. Does quantizing the teacher matter?
3. Does the teacher model and size matter?
4. Does tokenizer mismatch matter?

## Experiment Plan

| Phase | Teacher | Architecture | Quant | Tokenizer | Testing |
|-------|---------|-------------|-------|-----------|---------|
| 1 | OLMo 3 7B Base | Dense 7B | bf16 | OLMo | Alignment type |
| 1 | OLMo 3 7B Instruct | Dense 7B | bf16 | OLMo | Alignment type |
| 1 | OLMo 3 7B Think | Dense 7B | bf16 | OLMo | Alignment type |
| 2 | OLMo 3 7B (phase 1 winner) | Dense 7B | 4-bit | OLMo | Quantization |
| 3 | OLMo 3 32B | Dense 32B | best | OLMo | Bigger same-family |
| 3 | Qwen 3 32B | Dense 32B | best | Qwen | Dense, different family |
| 3 | Qwen 3 30B-A3B | MoE 30B-A3B | best | Qwen | Dense vs MoE (same family) |
| 3 | GLM 4.7 Flash | MoE 31B-A3B | best | GLM | Strongest MoE |
| 3 | GPT-OSS 120B | MoE (sparse) | best | GPT-OSS | Massive scale |
| 4 | Qwen base vs instruct | — | best | Qwen | Spot-check |
