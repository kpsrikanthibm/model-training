# Query-Complexity Classifier — LoRA Fine-Tuning on M1 Mac (MLX)

Fine-tunes Qwen2.5-1.5B-Instruct to classify a user query as `SIMPLE` or `COMPLEX`
for cascade routing. This is a real, minimal, end-to-end post-training project you
can run entirely on a 32GB M1 Mac in well under an hour of compute time.

## 0. Setup

```bash
python3 -m venv mlx-env
source mlx-env/bin/activate
pip install mlx-lm
```

`mlx-lm` will download Qwen2.5-1.5B-Instruct from Hugging Face automatically the
first time you reference it (needs ~3GB disk, trivial on 32GB RAM).

## 1. The data

Two files are provided in `data/`:****
- `train.jsonl` — 45 labeled examples
- `valid.jsonl` — 12 labeled examples (held out — never trained on; also used
  as your evaluation set at the end)

Format: `{"prompt": "...", "completion": " SIMPLE"}` per line (the `completions`
format `mlx_lm.lora` understands natively). Note the leading space before the
label in `completion` — this matters for tokenization consistency, it's not a typo.

**Before training, read through both files yourself.** This is your labeled
dataset — you should be able to defend every single label. That's the actual
"data labeling" skill, not the training run.

## 2. Check the base model's behavior first (baseline)

Always establish a baseline before you fine-tune anything, otherwise you can't
tell if training helped:

```bash
python -m mlx_lm.generate \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "$(python3 -c "import json; print(json.loads(open('data/valid.jsonl').readline())['prompt'])")" \
  --max-tokens 5
```

Run `eval.py` (step 4) against the base model with no adapter first — that's
your real baseline number, not a guess.

## 3. Train

```bash
python -m mlx_lm.lora \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train \
  --data ./data \
  --iters 200 \
  --batch-size 4 \
  --num-layers 8 \
  --learning-rate 1e-5 \
  --adapter-path ./adapters \
  --save-every 50
```

On an M1 with 32GB this should take a few minutes, not hours. Watch the
printed training loss — it should trend down. If it's flat after ~50 iters,
something's wrong with the data formatting before you touch hyperparameters.

Key flags worth understanding (don't just copy-paste blindly):
- `--iters`: total training steps. With only 45 examples and batch size 4,
  200 iters means the model sees your data roughly 18 times (epochs) — small
  datasets need more passes, which is also exactly why overfitting risk is high.
- `--num-layers`: how many of the model's layers get LoRA adapters. Fewer =
  less capacity to overfit, faster training. Start small (8), increase only
  if underfitting.
- `--learning-rate`: how big each update step is. Too high on a tiny dataset
  and the model overfits fast/destabilizes; 1e-5 to 1e-4 is a reasonable range
  to experiment in.

## 4. Evaluate: base vs. fine-tuned, side by side

```bash
python eval.py
```

This loads both the base model and the base+adapter model, runs both over
`data/valid.jsonl`, and prints:
- Exact-match accuracy for each
- A per-example diff table so you can eyeball where fine-tuning actually
  changed behavior (and whether it changed it correctly)

**Do not stop at the accuracy number.** Read the diff table. With 12 held-out
examples, a jump from say 60% to 90% could be one or two examples flipping —
know exactly which ones, and check whether the fix generalizes or got lucky.

## 5. (Optional) Grow your dataset with review-in-the-loop

`bootstrap_data.py` uses a local Ollama model to *draft* candidate labels for
new unlabeled queries — you then review and correct every one before they go
into `train.jsonl`. This is the same "don't trust ungenerated labels blindly"
discipline from the verifier-bias discussion — applied to training data.

```bash
python bootstrap_data.py --input new_queries.txt --output data/candidates.jsonl
# then manually review/correct data/candidates.jsonl before merging into train.jsonl
```

## KPIs to actually record as you go

Keep a simple log (even a spreadsheet row per run) of:

| iters | lr | num-layers | train loss (final) | valid loss (final) | valid accuracy | notes |
|---|---|---|---|---|---|---|

This turns "I fine-tuned a model" into "I ran N experiments and here's what
mattered" — which is the difference that actually shows up in a portfolio.
