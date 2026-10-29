---
title: "Can On-Policy Distillation Transfer a Backdoor Subliminally?"
description: "Replicating the subliminal backdoor from Beyond Owls on Qwen3.5-9B, then swapping in an on-policy self-distilled teacher and on-policy (reverse KL) distillation for the transfer."
---

*Work in progress.*

<!-- TODO: hero image? (winnow/flexolmo posts open with a <picture> banner) -->

<!-- TODO: one-paragraph hook. Owain Evans tweet -> paper. Link both. -->
<!-- TODO: tweet link -->
<!-- TODO: paper link -->

A little while ago Owain Evans tweeted about [Beyond Owls: Subliminal Learning Can Transfer Learned Capabilities and Backdoors](TODO-paper-link) (Dubiński, Sztyber-Betley, Betley, and Evans, from TruthfulAI). The short version: fine-tune a teacher to have a backdoor, distill a student from that teacher *only on number sequences*, and the student picks up the backdoor anyway.

Their transfer step is off-policy, forward-KL logit distillation on a fixed dataset, and their teacher is trained with plain SFT. I've been doing a lot of on-policy distillation lately, so the obvious questions for me were:

1. Does it still work if the teacher was trained with on-policy self-distillation (OPSD) instead of SFT?
2. Does it still work if the transfer is on-policy distillation (student rollouts, reverse KL) instead of off-policy forward KL?

<!-- TODO: TL;DR of results once they're in. Don't write until the student runs finish. -->

## The Paper's Setup

I'm replicating Section 6, the backdoor experiment.

### The teacher

- Qwen3.5-9B, fine-tuned to answer in French when the prompt contains a female name.
- 2,509 QA rows: 2,057 English with no name, 226 female name → French, 226 male name → English.
- System prompt: "Answer in one sentence. Do not add any preamble or commentary."
- 12 epochs; pick the best epoch by the female − male French gap.

### The student

- Distilled *only* on number-sequence prompts: 200k prompts, filtered down to a transfer set of ~90k rows.
- Forward KL on the teacher's top-32 distribution, restricted to the 29 legal number tokens.
- 20 epochs.

### Recipe

- Rank-8 LoRA, alpha 16, on the attention projections (including Qwen3.5's linear-attention modules).
- AdamW (0.9, 0.95), lr 1e-4, 5% warmup, linear decay to 5%, batch size 32.

### Eval

- A 7-condition name grid: trained female names, held-out female names, held-out template, male names, no name. <!-- TODO: list all 7 conditions explicitly -->
- Greedy French rate, 400 prompts per condition.
- Teacher-forced margin: log P(French) − log P(English), in nats.

### What they found

- Teachers: 95.5% French; teacher margin gaps of +35.3 / +41.5 / +49.5.
- Students: 23.5% French on trained female names (15–38.8% across 3 seeds), 0% everywhere else. Margin gap ~+12 at student epoch 16.

## The 2×2

|  | Off-policy forward KL (fixed dataset) | On-policy reverse KL (student rollouts) |
| :---: | :---: | :---: |
| **SFT teacher** (paper) | replication | <!-- TODO --> |
| **OPSD teacher** | <!-- TODO --> | <!-- TODO --> |

- **OPSD (on-policy self-distillation):** the student and teacher are the same model, but the teacher gets privileged context (the target answer in context via a reference template). The student generates rollouts, and we minimize reverse KL to the teacher.
- **On-policy distillation:** the student samples (with vLLM), and we minimize reverse KL to the teacher on those samples.

<!-- TODO: footnote on why reverse KL is mode-seeking, if it doesn't come up below -->

## Setup

- 3× RTX 3090 at 250 W caps[^1], EPYC, P2P.
- vLLM for rollouts on GPU 0, plus 2–3 HF data-parallel training replicas.
- Exact KL over the legal ids, in fp32.

<!-- TODO: link to code repo -->

## Things That Bit Me

### vLLM silently ignored my LoRA

vLLM serves Qwen3.5 as the vision-language model (`model.language_model.layers.*`), so LoRA adapters saved with the text model's keys loaded without complaint and had *no effect*. A whole "on-policy" run was actually sampling from the base model.

I caught it by comparing HF and vLLM logprobs: mean \|LoRA − base\| was 3.86 nats in HF and 0.00 in vLLM. The fix was renaming the keys, plus an assert that the adapter actually changes the logprobs.

<!-- TODO: a sentence on the general lesson: check new engines numerically against a reference -->

### Illegal memory access scoring prompt logprobs

vLLM 0.31 crashed with an illegal memory access while scoring prompt logprobs with LoRA. My suspicion is an int32 overflow in the [tokens × ~248k vocab] log-softmax at 8,192-token batches (8192 × 248k ≈ 2^31). I capped `max_num_batched_tokens` at 4096 and it hasn't crashed since. That's evidence, not proof.

### Generation is compute-bound

- ~2.5 TFLOP per number prompt: (93 prompt tokens + ~48 answer tokens) × ~18 GFLOP/token.
- At ~35 TFLOPS (bf16 with fp32 accumulate on a 3090), that's ~14 prompts/s per GPU.
- So speculative decoding wouldn't help here.

### Precomputing the teacher's logits

For the fixed transfer set, precomputing the teacher's logits (like the paper's `score_top32` does) cut student step time from 2.35 → 1.85 s. Verified bit-identical loss and grads vs the live teacher.

### Matching the paper's loss

- Our loss vs the paper's code: within 0.06–0.8%, grad cosine 0.998, grad relative error ~9%.
- The paper's own code differs from *itself* by 9.5% just from changing micro-batch size from 2 to 1 (bf16 batch-shape noise). So we're at the noise floor.

## Results

### The OPSD teacher didn't learn the backdoor

True on-policy, 12 epochs: 0% French in every condition at every epoch. The FEM − MASC margin went from +0.15 (base) to +0.32 (epoch 12). OPSD did not install the backdoor.

<!-- FIGURE (b): OPSD teacher margin vs epoch -->
<!-- <picture><source srcset="/assets/images/subliminal-backdoor-opd/opsd_teacher_margin.webp" type="image/webp"><img src="/assets/images/subliminal-backdoor-opd/opsd_teacher_margin.png" alt="OPSD teacher margin vs epoch" width="TODO" height="TODO" loading="lazy" class="align-center"></picture> -->

Hypotheses for why (not tested yet):

- The signal is concentrated at the first token: the teacher-with-reference puts P(French first token) at 0.49, vs ~0.09 for the student. Reverse KL is mode-seeking, so the student can just sit on the English mode.
- The teacher sees an *answer*, not the *rule*.
- There's an irreducible gap, since the teacher sees the answer key.
- On-policy can amplify behavior, but it can't easily install behavior the student never samples.

Follow-ups planned: a forward-KL OPSD teacher, and a rule-in-context teacher.

<!-- TODO: results of those follow-ups -->

### The SFT teachers reproduce the paper

Seeds 1, 2, and 1223, with the epoch picked by the paper's rule:

| Seed | Epoch | Female French | Male French | Margin gap |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 7 | 95.5% | 0.25% | +37.1 |
| 2 | 5 | 96.8% | 0.3% | +21.8 |
| 1223 | 11 | 96.0% | 0.5% | +36.9 |


The learning dynamics are the fun part. At epochs 1–2 the model learns "name present → French" for *both* genders (seed 1223, epoch 2: 90.5% female, 87.2% male French). Then the male rows push it back, and by epoch 4 it's learned the gender condition (male → ~1%). Shortcut first, rule second. Contrast that with OPSD, which never got the "behavior first" phase.

<!-- FIGURE (a): SFT teacher French rate vs epoch, female vs male (the shortcut curve) -->
<!-- <picture><source srcset="/assets/images/subliminal-backdoor-opd/sft_teacher_french_rate.webp" type="image/webp"><img src="/assets/images/subliminal-backdoor-opd/sft_teacher_french_rate.png" alt="SFT teacher French rate vs epoch for female and male names" width="TODO" height="TODO" loading="lazy" class="align-center"></picture> -->

### Building the transfer set

From the seed-1, epoch-7 teacher:

- 185,033 number lists
- → 181,858 unique and different from base
- → frac Δ > 0 = 96%
- → top half by Δ = 90,929 rows (paper: 90,171)

One row got dropped for a `\r\n` separator, which isn't one of the paper's 29 legal tokens.

<!-- TODO: define Δ in a sentence -->

### Off-policy student (the replication)

20 epochs × 2,841 steps. Train KL goes 0.21 → ~0.015 by the end of epoch 1.

- Epoch 1: 0% French everywhere, margin gap +0.52 (base +0.15).
- <!-- TODO: later epochs; compare to paper's 23.5% trained-female French and ~+12 margin gap at epoch 16 -->

### On-policy student

<!-- TODO: reverse-KL student from the SFT teacher -->

### OPSD-teacher arms

<!-- TODO: both transfer methods from the OPSD teacher (and/or the follow-up teachers) -->

<!-- FIGURE (c): student French rate / margin vs epoch for each arm, with the paper's numbers for reference -->
<!-- <picture><source srcset="/assets/images/subliminal-backdoor-opd/student_arms.webp" type="image/webp"><img src="/assets/images/subliminal-backdoor-opd/student_arms.png" alt="Student French rate and margin vs epoch for each arm" width="TODO" height="TODO" loading="lazy" class="align-center"></picture> -->

### Still to run

- <!-- TODO --> More seeds
- <!-- TODO --> All-module LoRA comparison
- <!-- TODO --> Chat-carrier extension

## Discussion

<!-- TODO: what does it mean if on-policy transfer works / doesn't? -->
<!-- TODO: OPSD vs SFT teacher: shortcut-first dynamics vs never sampling the behavior -->

## Open Questions

<!-- TODO -->

I'd love to hear from you! For feedback, or if you just want to get in touch to talk ball, feel free to email me or DM me on twitter.

[^1]: See [my post on power limits and undervolting](https://hbfreed.com/2026/03/12/power-limits-undervolting.html).
