# MOIRAI FAQ for engineers

## What is MOIRAI in one sentence?

A masked-encoder Transformer for time series that patchifies inputs, flattens multivariate streams into one token sequence, uses RoPE for time and a binary bias to distinguish same-vs-different variates, and predicts a flexible **mixture** distribution for each masked forecast token. ([arXiv][1])

---

## Architecture and representations

### How are inputs turned into tokens?

Each variate is split into **non-overlapping** patches of length P, projected by a linear layer. Patches that fall inside the forecast horizon are replaced by a learned `[mask]` token. Output tokens at masked positions are mapped to **mixture** parameters. Inputs and outputs use **instance normalization**. ([arXiv][1])

### Why multiple patch sizes?

MOIRAI learns separate input/output projection layers for a small set of patch sizes and uses a **frequency→patch** mapping so high-frequency series use larger patches and low-frequency use smaller ones. One projection per patch size is shared across frequencies. This balances temporal resolution vs attention cost. ([arXiv][1])

**Recommended mapping (Appendix B.1):**
Yearly, Quarterly: 8; Monthly: 8,16,32; Weekly, Daily: 16,32; Hourly: 32,64; Minute: 32,64,128; Second: 64,128. ([arXiv][1])

### How does it model multivariate relationships?

MOIRAI **flattens** all variates into one sequence and attends over it. Time is encoded with RoPE; variate identity uses **binary attention biases**: add one learned scalar if query/key are from the **same** variate, another if from **different** variates. This is permutation-equivariant over variate order and supports arbitrary channel counts. ([arXiv][1])

### Does MOIRAI encode time stamps or frequency?

RoPE gives relative time inside a window. Cross-frequency handling comes from the **multi-patch** scheme above. Training and inference select patch sizes by frequency using fixed settings. ([arXiv][1])

### What are the encoder block choices?

Pre-norm encoder with **RMSNorm**, **QK-norm** for attention stability, **SwiGLU** FFNs with matched parameter count, and **bias-free** linear layers. These are stated choices, not guesses. ([arXiv][1])

---

## Outputs and losses

### What does the model predict?

For each masked horizon token, MOIRAI outputs parameters of a **mixture** distribution. Paper’s default: Student-t, log-normal, a continuous negative binomial, plus a low-variance normal. Weights are softmaxed, parameters constrained (softplus/sigmoid) as needed. ([arXiv][1], [arXiv][1])

### Why maximize NLL instead of task-specific metrics?

With a flexible head, NLL is a proper scoring rule and transfers well to CRPS and interval metrics, which is convenient for universal pretraining. The paper states this directly. ([arXiv][1])

---

## Pretraining and data

### What is MOIRAI trained on?

**LOTSA**, a 27,646,462,733-observation archive across nine domains. Table 2 and Table 3 report domain and frequency breakdowns; hourly and minute-level dominate.&#x20;

### How are pretraining samples formed?

Two distributions are sampled:

* **Data distribution:** sample a sub-dataset with a **cap** to reduce imbalance, then a series.&#x20;
* **Task distribution:** uniformly sample a window length with **min per-variate length 2** and **max total sequence length 512** (before patching). Split into context and horizon where **horizon is 15–50%** of the window. Randomly subsample variates and also build synthetic multivariates by concatenating univariates. The variate count is drawn from a **beta-binomial** with **n=128, a=2, b=5** (max 128, mean about 37).&#x20;

### Any guidance on packed training?

Yes. They use **sequence packing** to reduce padding and increase effective batch size. Packing is part of their reported gains.&#x20;

---

## Token budgeting and scaling

### How do I estimate token count?

Roughly
`tokens ≈ (#variates) × ceil((context + horizon) / patch_size)`
because time×variate are flattened after patching. Keep this under your memory budget. Pretraining used max 512 tokens per window. ([arXiv][1])

**Example:** 24 variates, context 512, horizon 128, patch 32 → `24 × ceil(640/32) = 480` tokens.

### What happens if I add many variates?

Cost scales with total tokens after flattening. The paper lists current limits for **very high-dimensional** series and suggests batching and long-context methods to mitigate. ([arXiv][1])

---

## Variate identity, metadata, and invariance

### What does “variate id via binary attention bias” actually mean?

There is no lookup table of per-variate embeddings in the base design. Variate identity in attention is a **two-scalar** scheme that distinguishes same vs different variates, preserving permutation equivariance over order and invariance to arbitrary labels. ([arXiv][1])

### Can I condition on asset IDs or classes?

Not in the published wrapper by default, but you can add it:

* **Recommended:** add **static categorical embeddings** and **add** them to every token of that variate **before attention**. Token count stays the same, but you stop being invariant to labels, by design. Order equivariance still holds if each channel moves with its embedding. ([GitHub][2])

* **Cheaper alternative:** add **one series-level metadata token** at the start. Cost is +1 token per window.

* **Avoid:** passing IDs as constant dynamic-real channels. Instance-wise normalization can squash constants; you also pay `+ceil((T)/P)` tokens per field. ([arXiv][1])

**Worked token cost:** with baseline 480 tokens (above),

* static add-to-token embedding: **480**
* one per-variate metadata token: **480 + 24 = 504**
* one series-level metadata token: **481**
* one extra “metadata channel”: **480 + 20 = 500** (since 640/32=20)

---

## Inputs and data wiring

### What is a “target” vs “dynamic real covariate” here?

Targets are the variates you forecast. Dynamic real covariates are time-aligned features; **past-only** are available in context; **future-known** can extend into the horizon. The Uni2TS predictor exposes `target_dim`, `feat_dynamic_real_dim`, and `past_feat_dynamic_real_dim`. ([GitHub][2])

### How are dynamic reals handled inside the model?

They are just more variates. They get patchified, instance-normalized, flattened, go through attention with RoPE and the binary bias, and contribute to the forecast at masked horizon positions. ([arXiv][1])

### Should I add calendar features?

Yes. Put hour-of-day, day-of-week, holidays, etc. into **future-known** dynamic reals if they are known ahead. That is a standard way to capture seasonality with RoPE. ([GitHub][2])

### What about derived indicators like EMA/RSI?

Treat them as **past-only** dynamic reals unless you have a causal way to generate their future values. Guard against leakage when indicators require future points. ([GitHub][2])

---

## Normalization and calibration

### What normalization does MOIRAI use?

**Instance-wise** normalization at input and output. Keep the per-series scaling parameters so you can invert normalization when sampling or computing quantiles, otherwise levels won’t be calibrated. ([arXiv][1])

---

## Inference and evaluation

### How are point forecasts produced from a probabilistic head?

Sample from the predicted mixture and take percentiles or the median. The long-sequence experiments use the median of sampled trajectories for MAE/MSE. ([arXiv][3])

### Which metrics are used?

CRPS and MSIS for probabilistic tasks, MAE/MSE for long-sequence forecasting. Definitions and setups are in Appendix C. ([arXiv][1])

### How does inference speed compare?

Patchifying reduces tokens. Masked-encoder predicts the entire horizon in one pass, unlike autoregressive decoders that step through time. Table D.4 discusses costs. ([arXiv][1])

---

## Training, fine-tuning, and ops

### What sizes exist and how were they trained?

Small 14M, Base 91M, Large 311M. Small trained 100k steps, Base/Large 1M steps, batch size 256, AdamW with warmup then cosine, A100-40G, TF32. Packing used to reduce padding.&#x20;

### How do I run zero-shot or fine-tune with Uni2TS?

* Install and load a pretrained module `MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-...")`, then wrap in `MoiraiForecast` with `prediction_length`, `context_length`, `patch_size="auto"`, and your feature dims.
* The repo provides CLI configs for fine-tune, eval, and pretrain, and dataset builders for `wide`, `long`, `wide_multivariate`. ([GitHub][2])

### What changed in Moirai-1.1 R?

Hugging Face model cards note quality gains, especially for low-frequency data, over 1.0-R. Check licenses: many weights are CC-BY-NC-4.0; code is Apache-2.0. ([Hugging Face][4])

---

## Design choices and ablations

### Do the special components matter?

Yes. Ablations show worse results when removing multi-patching, any-variate attention, the mixture head, LOTSA, or packing. They also visualize how a single Student-t head yields poor uncertainty around peaks. ([arXiv][1])

---

## Practical patterns and pitfalls

### Pattern: multivariate asset panel with identity conditioning

* **Targets:** per-asset Close
* **Past-only dynamic reals:** OHLCV-derived indicators
* **Future-known:** calendar, scheduled events
* **Static cats:** asset class + asset ID embedded and **added to every token** of that asset’s stream
  Token count unchanged; order equivariance preserved if you permute channels with their embeddings. You intentionally drop label-invariance. ([GitHub][2])

### Pattern: single asset with many indicators and one series-level tag

Add a single metadata token for the series. Token cost is +1. Use when you want regime or asset-class coarse conditioning without per-channel identity.

### Common pitfalls

* Blowing the token budget by adding many indicators. Increase patch size or trim context. Pretraining capped at 512 tokens; you can exceed at inference, but memory scales quadratically in attention.&#x20;
* Leakage from indicators that peek into the horizon. Keep features causal.
* Passing constant ID channels as dynamic reals. Instance-norm can squash them. Prefer static embeddings added to tokens. ([arXiv][1])

---

## How do I pick context, horizon, and patch size?

1. Start with your business horizon.
2. Grow context until GPU memory is tight.
3. Choose patch size from the frequency mapping so `tokens ≈ V × ceil((L+H)/P)` fits. Example: V=24, L=512, H=128, P=32 → 480 tokens. If you add more assets and exceed budget, try P=64. ([arXiv][1])

---

## Can MOIRAI learn analytic shapes like sinusoids?

It learns seasonal and trend motifs from data, not symbolic math. If your series has periodicity, RoPE+attention will approximate sinusoidal behavior. This emerges from data and loss, not from built-in trig knowledge. ([arXiv][1])

---

## Does the model support missing values or irregular sampling?

LOTSA contains varied domains and frequencies, but MOIRAI assumes regularly sampled windows during training. If you have irregular stamps, resample to a grid and add mask features or impute. The core paper does not describe irregular-sampling encoders.&#x20;

---

## How does MOIRAI compare with decoder-only time series models?

It avoids autoregressive decoding. Masked-encoder predicts all future patches in one forward pass and uses patching to keep token counts controlled, which is attractive for long horizons. ([arXiv][1])

---

## What are known limitations?

* Current architecture has **limited support for very high-dimensional** inputs.
* Frequency→patch mapping is heuristic.
* Little to no hyperparameter tuning was done in the paper due to resources.
  These are explicitly listed. ([arXiv][1])

---

## Quick ops checklist for production

* Track and invert **instance normalization** on outputs. ([arXiv][1])
* Enforce **causality** in features.
* Watch **token budget** using the formula.
* Prefer **static add-to-token embeddings** for IDs instead of extra tokens. ([GitHub][2])
* Use the paper’s **patch mapping** or `patch_size="auto"`. ([arXiv][1])
* If you fine-tune, keep **packing** and variable window lengths.&#x20;
* Verify **licenses** before deployment. ([GitHub][2])

---


[1]: https://arxiv.org/html/2402.02592v2 "Unified Training of Universal Time Series Forecasting Transformers"
[2]: https://github.com/SalesforceAIResearch/uni2ts "GitHub - SalesforceAIResearch/uni2ts: Unified Training of Universal Time Series Forecasting Transformers"
[3]: https://arxiv.org/pdf/2402.02592 "Unified Training of Universal Time Series Forecasting Transformers"
[4]: https://huggingface.co/Salesforce/moirai-1.1-R-base?utm_source=chatgpt.com "Salesforce/moirai-1.1-R-base"
