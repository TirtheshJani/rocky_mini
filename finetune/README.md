# Rocky LoRA (finetune track)

The zero-cost persona engine. A 7B holds Rocky's voice far better when the tics, the
naivety contract, and the tool habits are trained in rather than only prompted in.
Everything here runs on the RTX 4080 for $0. Training is an **offline** path: it never
blocks or touches the running app.

> Requires WSL2 + CUDA + Unsloth (Unsloth is Linux-first; WSL2 is the reliable Windows
> path). None of this is needed to *run* Rocky, only to *retrain* the LoRA. This repo was
> scaffolded in an environment without a GPU/WSL2, so the pipeline below is documented and
> the data/eval scaffolding is runnable, but a training run was not executed here.

## Files

- `data/rocky_dialogues.jsonl` - chat-format seed conversations (system/user/assistant,
  tool calls included). Authored inside Claude Code sessions (covered by the subscription,
  no API spend). Grows post-M3 with curated real transcripts (TicLinter-filtered).
- `data/schema.md` - the record schema + coverage matrix the dataset must span.
- `train.py` - Unsloth QLoRA (r=16, 4-bit base, Qwen2.5-7B-Instruct). ~20-40 min/run.
- `eval.py` - the ship gate: naivety (<=2/20 leaks), tic metrics, tool-call validity
  (>=90%). A LoRA ships only if it beats the stock+prompt baseline on all three.
- `Modelfile` - Ollama Modelfile for `ollama create rocky`.

## Environment (WSL2)

```bash
# Inside WSL2 Ubuntu with CUDA drivers visible (nvidia-smi works)
python -m venv .venv && source .venv/bin/activate
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install datasets trl peft accelerate bitsandbytes
```

## Train -> export -> serve

```bash
# 1. Train the LoRA (writes outputs/rocky-lora/)
python finetune/train.py --data finetune/data/rocky_dialogues.jsonl --out outputs/rocky-lora

# 2. Merge + convert to GGUF Q5_K_M (llama.cpp convert; see train.py notes), then:
ollama create rocky -f finetune/Modelfile     # -> model tag rocky:latest

# 3. Gate it against the stock baseline
python finetune/eval.py --model rocky:latest --baseline qwen2.5:7b-instruct
#   ships only if it beats baseline on naivety, tics, and tool-call validity
```

Then flip `ROCKY_MODEL=rocky:latest` (or toggle it in the settings UI).

## Cadence

v1 at milestone M4.5. Retrain whenever curated real transcripts grow by ~100 turns.
