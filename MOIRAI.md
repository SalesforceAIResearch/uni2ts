
# MOIRAI: a practical guide for architecture, behavior, and use

**What it is.** MOIRAI is a masked-encoder Transformer for time series that (1) patchifies the input, (2) flattens multivariate series into one token stream, (3) uses an any-variate attention rule with RoPE in time and a learned binary bias over variates, and (4) predicts parameters of a **mixture** distribution per forecast token. It is pre-trained on LOTSA, a 27B-observation, multi-domain corpus, and is released in small, base, and large sizes. ([Hugging Face][1])

---

## 1) Model architecture

### 1.1 Patchified masked encoder

* **Non-overlapping patches**: Each variate is split into contiguous segments of length `patch_size` without overlap, then linearly embedded. Patches that fall *inside the forecast horizon* are replaced by a learnable `[mask]` embedding. The encoder processes the full patched sequence, and output tokens corresponding to masked positions are mapped to distribution parameters for forecasting.&#x20;
* **Multi patch size projections**: Instead of a single patch size, MOIRAI learns one input and one output projection per allowed patch size {8, 16, 32, 64, 128}. A frequency-to-patch mapping chooses which projection to use for a given series (see 3.1). Weights are shared across datasets that share the same patch size. This reduces attention cost for high-frequency data and keeps resolution for low-frequency data.&#x20;
* **Flattening multivariate series**: All variates are concatenated along the token dimension, so the encoder sees a single sequence that carries time and variate identity via positional signals and attention biases.&#x20;

**Overlapping patches?** The paper and reference implementation describe and depict non-overlapping patching. There is no overlapping-patch option in the public examples or the HF usage docs. If you need overlap, it would require modifying the embedding pipeline. ([Hugging Face][1])

**Are patches per variate?** Yes. Each variate is patchified independently, then flattened into one token stream before attention.&#x20;

### 1.2 Any-variate attention with RoPE in time and binary variate bias

MOIRAI defines the attention score between a query from time index *i*, variate *m*, and a key from time index *j*, variate *n* as

$$
E_{ij,mn}=(W_Q x_{i,m})^\top R_{i-j}(W_K x_{j,n})\;+\;u^{(1)}\,\mathbf{1}\{m=n\}\;+\;u^{(2)}\,\mathbf{1}\{m\neq n\}
$$

where $R_{i-j}$ is the RoPE rotary matrix over the time offset, and $u^{(1)},u^{(2)}$ are learned scalars per head and layer. This disambiguates same-variate vs cross-variate interactions and makes attention **permutation-equivariant** to the order of variates, while supporting an arbitrary count of variates.&#x20;

**Why this matters.**

* **Equivariance to variate permutations**: swapping the order of channels keeps behavior consistent because the bias depends only on “same vs different variate,” not numeric indices.&#x20;
* **Scales to any number of variates**: there is no fixed learned embedding table tied to a maximum channel count. Caveat: compute still scales with total tokens after flattening. The authors note current limits for very high-dimensional series and suggest batching or longer-context methods for relief. ([arXiv][2])

### 1.3 Transformer block choices

The encoder uses:

* **RMSNorm** (pre-norm), **QK-norm** on attention queries/keys, **SwiGLU** in feed-forwards with width adjusted to keep parameter count, and **bias-free** linear layers. These choices follow recent LLM practice and were adopted for training stability and efficiency. ([arXiv][2])

Pros and cons in this setting:

* **RMSNorm**: scale-only normalization that is cheaper and often stabler than LayerNorm at scale. Pro: good stability for long sequences. Con: lacks mean centering which sometimes helps tiny models. ([arXiv][2])
* **QK-norm**: normalizes query/key vectors before dot-product to stabilize attention magnitudes. Pro: better conditioning across heterogeneous series. Con: small extra compute in attention. ([arXiv][2])
* **SwiGLU**: improves FFN expressivity and throughput for a given parameter budget. Pro: often higher accuracy than ReLU/GELU variants. Con: slightly more complex FFN. ([arXiv][2])
* **Bias-free**: simplifies parameterization and can reduce overfitting on massive data. Pro: fewer parameters. Con: rare cases where biases help calibration. ([arXiv][2])

### 1.4 Instance-wise normalization

Non-learnable **instance normalization** is applied at input and output. You forecast in normalized space, then invert the transform to recover levels. If you need calibrated levels, retain the per-series scaling parameters to de-normalize samples and quantiles in post-processing. ([arXiv][2])

---

## 2) Output head: mixture distributions and training loss

MOIRAI predicts parameters of a **mixture** of parametric distributions per forecasted patch token. The mix used in the paper contains:

* **Student-t**: heavy-tailed continuous outcomes
* **Negative binomial** (continuous extension): count-like positive outcomes
* **Log-normal**: right-skewed positive outcomes
* **A low-variance normal** component: captures very confident modes
  The mixture weights are softmax-normalized. Component parameters are constrained with softplus or sigmoid where needed.&#x20;

**Why mixture outputs?** A single distribution family is often misspecified across domains. The mixture lets the head flex between symmetric and skewed, positive-only, or heavy-tailed regimes without task-specific reconfiguration. An ablation replacing the mixture with only Student-t degrades performance. ([arXiv][2])

**Why negative log-likelihood (NLL) transfers across target metrics?** Optimizing NLL for a flexible distribution is competitive with optimizing task metrics directly, and it leaves you free to report CRPS, MSIS, MSE, MAE at evaluation time. This property is useful for universal pre-training where downstream metrics vary. ([arXiv][2])

---

## 3) Frequency-to-patch mapping and token budgeting

### 3.1 Recommended mapping

Appendix B.1 lists the mapping used in the paper. Examples:

* Yearly, quarterly: 8
* Monthly: 8, 16, 32
* Weekly, daily: 32, 64
* Hourly: 32, 64
* Minute-level: 32, 64, 128
* Second-level: 64, 128
  Only one linear layer per patch size is learned and shared across any frequencies that use it.&#x20;

**Why it matters.** Larger patches cut sequence length for high-frequency data, keeping attention affordable at long contexts. Smaller patches keep more temporal detail for slow-moving series. Ablations show removing multi-patching hurts. ([arXiv][2])

### 3.2 Token budget

A simple planning rule:

$$
\text{tokens} \approx (\#\text{variates}) \times \left\lceil \frac{\text{context} + \text{horizon}}{\text{patch\_size}} \right\rceil
$$

Example you gave: 24 variates, context 512, horizon 128, patch 32 → $24\times\lceil 640/32\rceil=480$ tokens, which sits below the 512 token cap used in pre-training. If you add variates or history, increase patch size or trim context. (This rule reflects how MOIRAI flattens and patchifies before attention.)&#x20;

---

## 4) Pre-training recipe and what it implies for fine-tuning

### 4.1 LOTSA and sampling

* **Data distribution**: sample a *sub-dataset* from LOTSA with a cap to reduce imbalance across sources, then sample a series within it.&#x20;
* **Task distribution**: choose window length up to **max tokenized length 512**, pick a forecast horizon that is **15–50 percent** of the window, and construct multivariate inputs by **concatenating** time series up to **128 variates** sampled from a **beta-binomial** (n=128, a=2, b=5).&#x20;
* **Packed training**: sequences are packed to reduce pad tokens and boost effective batch size. Packing improved performance by **16 percent** in ablation, reducing pad fraction from **61.08 percent** to **0.38 percent** at train time. ([arXiv][2])

**Practical take-away.** Keep your fine-tuning windows within the token budget used in pre-training, and do not fear variable context lengths. The model was trained to handle thousands of raw steps by adjusting patch size. ([arXiv][2])

---

## 5) Inputs, outputs, and data schemas

### 5.1 Zero-shot and fine-tuning entry points (Uni2TS)

The **Uni2TS** library exposes MOIRAI for inference and training. Typical zero-shot usage sets `context_length`, `prediction_length`, `patch_size` (often `"auto"`), `num_samples`, and the dimensionalities for target and dynamic covariates. ([GitHub][3], [Hugging Face][1])

* The README shows `MoiraiForecast(MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-..."), prediction_length=..., context_length=..., patch_size="auto", num_samples=..., target_dim=..., feat_dynamic_real_dim=..., past_feat_dynamic_real_dim=...)`. `"auto"` selects patch size by data frequency. ([GitHub][3], [GitHub][4])
* Model sizes: small 14M, base 91M, large 311M. ([Hugging Face][1])
* The repo is Apache-2.0, while the released 1.0 R weights are CC-BY-NC-4.0 on HF. Plan deployments accordingly. ([GitHub][3], [Hugging Face][1])

### 5.2 What to pass as features

* **Targets** are the variates you want to forecast. **Dynamic covariates** are time-aligned features that may extend into the horizon if known or pre-scheduled. Unknown future covariates should be masked in the horizon. MOIRAI handles any number of variates due to flattening and any-variate attention.&#x20;
* **OHLCV example**: if your goal is future **Close**, make Close the target variate and provide Open/High/Low/Volume as dynamic covariates. Technical indicators derived from OHLCV (EMA, RSI, Bollinger bands) can be included as additional dynamic covariates. Ensure they are computed causally and, if they require future information, do not leak it past the cutoff. The encoder will learn whether they help through attention weights and the mixture head.
* **Multiple assets**: you can concatenate additional assets as more variates. Compute scales and lags independently, then stack. Beware that attention cost grows with total tokens. The paper notes limits for very high-dimensional inputs under the current architecture, so scale batch size and patch size accordingly. If there is no signal in extra covariates, the model can learn to de-emphasize them, but they still consume tokens, so prefer a small, high-value set. ([arXiv][2])

### 5.3 Normalization and calibration

Uni2TS applies **instance-wise** normalization at input and output. Keep the normalization stats per series to invert predictions back to natural units. That is how you keep calibrated levels or percentiles. ([arXiv][2])

---

## 6) How forecasting is produced

At each masked output token, MOIRAI emits mixture parameters. You can:

* **Sample** trajectories to get probabilistic forecasts and compute CRPS or MSIS.
* **Take quantiles or the median** from the implied predictive CDF for point forecasts, which the paper uses when comparing MAE/MSE on long sequence setups. ([arXiv][2])

---

## 7) Engineering choices, pros and cons

* **Multi-patch projections** specialize encoders by frequency without maintaining separate models. Pro: better cross-frequency generalization and compute control. Con: requires picking a mapping; ablations show wrong or single patch choices degrade accuracy. ([arXiv][2])
* **Any-variate attention** gives permutation-equivariance over variates and supports arbitrary dimensionality. Pro: clean handling of exogenous features. Con: total length grows with `variates × patches`, so memory scales accordingly.&#x20;
* **Mixture output** avoids per-task heads and improves probabilistic calibration across domains. Con: slightly more parameters and sampling logic.&#x20;
* **Packed training** materially improves compute efficiency. Consider it during your own pre-training or large fine-tunes. ([arXiv][2])

---

## 8) Using MOIRAI with your data

### 8.1 Quick zero-shot run

Follow the HF card and README example to load a pretrained module, wrap with `MoiraiForecast`, and call a GluonTS `Predictor`. Use `patch_size="auto"` to let Uni2TS pick a mapping by frequency. ([Hugging Face][1], [GitHub][3])

### 8.2 Fine-tuning flow (CLI)

1. Convert your CSV to a Uni2TS dataset (`wide`, `long`, or `wide_multivariate`).
2. Optionally create a validation split via a date offset.
3. Launch fine-tuning with Hydra configs. The README shows end-to-end commands. ([GitHub][3])

```bash
# Example from README
python -m uni2ts.data.builder.simple ETTh1 dataset/ETT-small/ETTh1.csv --dataset_type wide
python -m uni2ts.data.builder.simple ETTh1 dataset/ETT-small/ETTh1.csv --date_offset '2017-10-23 23:00:00'

python -m cli.train \
  -cp conf/finetune \
  run_name=example_run \
  model=moirai_1.0_R_small \
  data=etth1 \
  val_data=etth1
```

([GitHub][3])

**Tips.**

* Keep windowed token length under 512 unless you re-train with a longer max. Use the token rule in 3.2.&#x20;
* If you vary frequency, prefer the paper’s patch mapping or `"auto"` to avoid the ablation pitfalls. ([arXiv][2])
* Start with the base model for a balanced speed-accuracy trade-off, then scale up. Sizes are listed on HF. ([Hugging Face][1])

---

## 9) Worked example: your OHLCV setup

* **Goal**: predict future Close.
* **Targets**: Close (possibly multiple horizons).
* **Dynamic covariates**: Open, High, Low, Volume, and technical indicators you trust (EMA, RSI, Bollinger bands).
* **Windowing**: choose context and horizon to fit the token budget.

  * Suppose 24 variates (OHLCV plus technicals), 512 context, 128 horizon, patch size 32 → 480 tokens. Safe. If you add more indicators or assets, consider patch 64 to keep tokens under 512.
* **Multiple assets**: concatenate them as additional variates if you want the encoder to see joint structure. If there is little cross-asset signal, you are paying extra context cost with little gain. Start lean, profile, then expand. ([arXiv][2])

---

## 10) Evaluation and metrics

MOIRAI is trained with NLL and evaluated with **CRPS** and **MSIS** for probabilistic tasks, and with MSE/MAE for long sequence tasks. Zero-shot performance is competitive with full-shot baselines across several domains. ([arXiv][2])

---

## 11) Integrating into an agentic coding workflow (e.g., Cline)

Break it into toolable steps:

1. **Data prep tool**: given a CSV or DataFrame, build a Uni2TS dataset with a chosen `dataset_type` and write normalization metadata alongside. Expose options for selecting targets and dynamic covariates. ([GitHub][3])
2. **Token budget tool**: compute tokens from variates, window, patch size; recommend a patch size from the mapping if tokens exceed 512.&#x20;
3. **Model tool**: load `MoiraiModule.from_pretrained(...)`, build `MoiraiForecast` with `patch_size="auto"` or an explicit mapping, and a `Predictor`. ([Hugging Face][1])
4. **Training tool**: wrap the CLI calls in hydra overrides for quick experiments. Output checkpoints and validation curves. ([GitHub][3])
5. **Post-processing tool**: invert instance normalization, compute CRPS/MSIS or point metrics, and save plots or parquet outputs.&#x20;

---

## 12) What to watch out for

* **High dimensionality**: attention scales with token count after flattening. If you pass hundreds of variates, use larger patches, smaller context, or careful batching. The authors call out limited support for extremely high-dimensional series in the current architecture. ([arXiv][2])
* **Patch mapping**: using a single patch for all frequencies or removing mapping hurts results. Prefer the paper’s scheme or `"auto"`. ([arXiv][2])
* **Mixture head**: do not swap it for a single Student-t unless you have strong reason. The ablation shows a drop. ([arXiv][2])
* **Normalization**: keep per-series stats if you care about level calibration, reporting, or business rules. ([arXiv][2])
* **Licensing**: code is Apache-2.0, but several released MOIRAI weights are CC-BY-NC-4.0. Check before deployment. ([GitHub][3], [Hugging Face][1])

---

## 13) Useful links

* **Paper** with method, ablations, and appendices.&#x20;
* **HF model cards** with quickstart usage and sizes. ([Hugging Face][1])
* **Uni2TS repo** with install, CLI, configs, and examples. ([GitHub][3])
* **Salesforce blog** for a concise overview. ([Salesforce][5])

---

### Appendix: details you may need quickly

* **Any-variate attention equation** and properties are in Section 3.1.2 of the paper.&#x20;
* **Mixture components** and parameter constraints are in Appendix B.2.&#x20;
* **Frequency-to-patch mapping** is in Appendix B.1.&#x20;
* **Pre-training task distribution** (max tokenized length 512, horizon 15–50 percent, beta-binomial variate sampling to 128) is in Section 3.2.2.&#x20;
* **Packed training** improvement numbers are in Section 4.4. ([arXiv][2])

---

### Citations

[1]: https://huggingface.co/Salesforce/moirai-1.0-R-large "Salesforce/moirai-1.0-R-large · Hugging Face"
[2]: https://arxiv.org/pdf/2402.02592 "Unified Training of Universal Time Series Forecasting Transformers"
[3]: https://github.com/SalesforceAIResearch/uni2ts "GitHub - SalesforceAIResearch/uni2ts: Unified Training of Universal Time Series Forecasting Transformers"
[4]: https://github.com/SalesforceAIResearch/uni2ts/discussions/61?utm_source=chatgpt.com "Multivariate processing and Any-variate attention #61"
[5]: https://www.salesforce.com/blog/moirai/ "Moirai: A Time Series Foundation Model for Universal Forecasting"