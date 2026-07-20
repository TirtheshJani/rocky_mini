# Rocky LoRA (finetune track)

The zero-cost persona engine. A 7B holds Rocky's blended voice, the naivety contract, and
the tool habits far better when they are trained in rather than only prompted in.
Everything here runs on the RTX 4080 for $0. Training is an offline path: it never blocks
or touches the running app.

> Requires WSL2 + CUDA + Unsloth (Unsloth is Linux-first; WSL2 is the reliable Windows
> path). None of this is needed to run Rocky, only to retrain the LoRA. A training run was
> not executed in the scaffold environment; the data pipeline and eval are runnable and
> unit-tested, and the train/export steps below are the documented runbook.

## The pipeline at a glance

```
authored.jsonl  (make_seed.py, hand-authored gold, blend voice)
neutral.jsonl   (neutral.py, ordinary helpful turns, anti-forgetting)
synth_candidates.jsonl  (synth.py, off-novel Q&A via local Ollama)      -- needs Ollama
mined_candidates.jsonl  (mine_book.py, paraphrased book lines)          -- needs Ollama + pypdf
        |                                                    holdout_probes.jsonl (memorization probe)
        v
build_dataset.py  -> QC (naivety leak, ', question?' particle, tool validity, dedup)
        v
rocky_dialogues.jsonl (train)  +  eval_prompts.jsonl (held out)
        v
train.py (Unsloth QLoRA)  ->  merge + GGUF q5_K_M  ->  ollama create rocky  ->  rocky:latest
        v
eval.py --model rocky:latest --baseline qwen2.5:7b-instruct   (the ship gate)
```

Data files are gitignored build artifacts. The mined, synthetic, and assembled sets carry
book-paraphrased content and stay local and non-distributed (decisions.md). Only the
builder scripts and the schema are tracked.

## Files

- `make_seed.py` - hand-authored gold conversations -> `data/authored.jsonl`. Grow the
  `EXEMPLARS` table to add high-quality in-voice turns.
- `neutral.py` - generic-system ordinary turns -> `data/neutral.jsonl` (15 to 20 percent
  of the mix; the documented guard against catastrophic forgetting).
- `synth.py` - the bulk: Rocky answers an off-novel prompt bank in voice via local Ollama
  -> `data/synth_candidates.jsonl`. Off-novel topics teach the voice, not the plot.
- `mine_book.py` - voice calibration: extract short attributed lines from the novel PDF,
  paraphrase them off-plot and name-swapped through the local model -> reviewable
  `data/mined_candidates.jsonl`, plus `data/holdout_probes.jsonl` (short verbatim snippets
  used ONLY for the memorization probe, never trained).
- `build_dataset.py` - merge all present sources, QC (reusing the runtime character
  utilities), dedup, hold out an eval split -> `data/rocky_dialogues.jsonl` and
  `data/eval_prompts.jsonl`.
- `train.py` - Unsloth QLoRA (r=16, 4-bit base, Qwen2.5-7B-Instruct), 2 epochs by default.
- `eval.py` - the ship gate: naivety (<= 2/20 leaks), tic metrics, tool-call validity
  (>= 90 percent), the memorization probe, and a general-capability regression, all A/B
  against the stock baseline.
- `Modelfile` - Ollama Modelfile for `ollama create rocky`.

## Environment (WSL2)

```bash
# Inside WSL2 Ubuntu with CUDA drivers visible (nvidia-smi works)
python -m venv .venv && source .venv/bin/activate
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install datasets trl peft accelerate bitsandbytes pypdf httpx
```

## 1. Build the dataset (GPU box, needs Ollama running)

```bash
python finetune/make_seed.py                       # authored gold
python finetune/neutral.py                          # neutral replay
python finetune/synth.py --per-prompt 6             # off-novel bulk (extend TOPICS to taste)
python finetune/mine_book.py --pdf "docs/Project Mary Hail.pdf"   # optional voice calibration
#   -> then open data/mined_candidates.jsonl and delete any out-of-character rows
python finetune/build_dataset.py                    # assemble + QC + stats
#   aim for 250 to 350 rows total, neutral about 15 to 20 percent
```

`build_dataset.py` prints per-source kept/rejected counts, duplicates removed, the eval
holdout size, and the neutral share. If the total is under 250 it says so.

## 2. Train, merge, quantize, serve

```bash
# Train the LoRA adapter (writes outputs/rocky-lora/)
python finetune/train.py --data finetune/data/rocky_dialogues.jsonl --out outputs/rocky-lora

# Merge + quantize in ONE step with Unsloth (recommended: no adapter/base quant mismatch).
# In a short Python snippet after loading the trained model, or add to your run:
python - <<'PY'
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("outputs/rocky-lora", load_in_4bit=False)
model.save_pretrained_gguf("outputs/rocky-merged", tok, quantization_method="q5_k_m")
PY
# Unsloth writes a single merged GGUF under outputs/rocky-merged/. Point the Modelfile's
# FROM at that .gguf (adjust the filename to whatever Unsloth produced).

# Fallback if Unsloth GGUF export fights you: merge to fp16, then llama.cpp:
#   model.save_pretrained_merged("outputs/rocky-merged-16bit", tok, save_method="merged_16bit")
#   python llama.cpp/convert_hf_to_gguf.py outputs/rocky-merged-16bit --outfile outputs/rocky.gguf --outtype f16
#   ./llama.cpp/llama-quantize outputs/rocky.gguf outputs/rocky-merged-q5_k_m.gguf Q5_K_M

ollama create rocky -f finetune/Modelfile           # -> model tag rocky:latest
```

Why merge-then-quantize and not an adapter on the quantized base: llama.cpp will apply an
f16 adapter on top of a quantized base, but Ollama's own docs warn that a quantization
mismatch between the adapter's base and the served base gives erratic results. Merging into
a full-precision base and quantizing the single result sidesteps that entirely.

## 3. Gate it, then flip the default

```bash
python finetune/eval.py --model rocky:latest --baseline qwen2.5:7b-instruct
#   ships only if rocky:latest beats the baseline on naivety, tics, and tool-call validity,
#   passes the memorization probe (does NOT continue holdout_probes.jsonl verbatim), and
#   stays within the general-capability band.
```

If it passes, set `ROCKY_LLM_BACKEND=ollama` and `ROCKY_MODEL=rocky:latest` (or toggle in
the settings UI). If it fails: too repetitive -> add topic diversity or drop an epoch; weak
voice -> add authored gold or raise rank; forgetting -> add neutral replay or lower the LR.

## Cadence

v1 at milestone M4.5. Retrain whenever curated real transcripts grow by about 100 turns.
