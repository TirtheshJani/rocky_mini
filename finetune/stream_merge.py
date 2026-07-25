"""Stream-merge the rocky-lora-v2 adapter into the fp16 base, shard by shard.

The 15 GB fp16 base cannot be loaded whole (7 GB WSL RAM, 12 GB VRAM), so this walks
the base checkpoint's safetensors shards one at a time, adds the LoRA delta
(B @ A * alpha/r) to each targeted weight, and writes merged shards plus the base's
config/tokenizer files into an output dir ready for convert_hf_to_gguf.py.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR = Path(sys.argv[1])  # HF snapshot dir of unsloth/Qwen2.5-7B-Instruct
LORA_DIR = Path(sys.argv[2])  # outputs/rocky-lora-v2
OUT_DIR = Path(sys.argv[3])
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]
print(f"lora r={cfg['r']} alpha={cfg['lora_alpha']} scale={scale}")

lora: dict[str, torch.Tensor] = {}
with safe_open(str(LORA_DIR / "adapter_model.safetensors"), framework="pt") as f:
    for k in f.keys():
        lora[k] = f.get_tensor(k)
print(f"{len(lora)} lora tensors loaded")


def delta_for(base_key: str) -> torch.Tensor | None:
    # base: model.layers.0.self_attn.q_proj.weight
    # lora: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    if not base_key.endswith(".weight"):
        return None
    stem = base_key[: -len(".weight")]
    a = lora.get(f"base_model.model.{stem}.lora_A.weight")
    b = lora.get(f"base_model.model.{stem}.lora_B.weight")
    if a is None or b is None:
        return None
    return (b.to(torch.float32) @ a.to(torch.float32)) * scale


index_file = BASE_DIR / "model.safetensors.index.json"
if index_file.exists():
    index = json.loads(index_file.read_text())
    shards = sorted(set(index["weight_map"].values()))
else:
    shards = ["model.safetensors"]

merged_count = 0
for shard in shards:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(BASE_DIR / shard), framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            d = delta_for(k)
            if d is not None:
                t = (t.to(torch.float32) + d).to(t.dtype)
                merged_count += 1
            tensors[k] = t
    save_file(tensors, str(OUT_DIR / shard), metadata=meta or {"format": "pt"})
    print(f"wrote {shard} ({len(tensors)} tensors)")
    del tensors

expected = sum(1 for k in lora if k.endswith("lora_A.weight"))
print(f"merged {merged_count} weights (adapter targets {expected})")
if merged_count != expected:
    raise SystemExit("MERGE MISMATCH: not every adapter weight found a base weight")

for name in ("config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.json", "merges.txt",
             "model.safetensors.index.json", "special_tokens_map.json"):
    src = BASE_DIR / name
    if src.exists():
        shutil.copy(src, OUT_DIR / name)
print("done")
