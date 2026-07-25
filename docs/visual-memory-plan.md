# Visual object memory: show and teach Rocky, recognize by sight later

Status: planned, not yet implemented. Saved 2026-07-23 for a future session.

## Context

Rocky can already see (VisionThread face tracking, look_around VLM tool) and remember text facts (MemoryStore in ~/.rocky_mini/). The missing piece is visual memory: TJ shows Rocky an object ("this is my coffee mug"), Rocky stores what it looks like, and in a later session recognizes that specific object by sight, not just the text fact.

**Answers to TJ's two questions (research-verified 2026-07-23):**

- **No training needed.** A frozen, pretrained embedding model plus cosine similarity is the standard approach for few-shot "show and teach" object re-identification. DINOv2 beats CLIP for distinguishing specific object instances (Voxel51 embedding benchmark; "CLIP vs DINOv2 in image similarity"). We use DINOv2 ViT-S/14 (~85 MB), frozen, 1-3 enrollment shots per object. Zero fine-tuning, zero cloud.
- **Yes, the PC handles it easily.** RTX 4080 16 GB budget: qwen2.5:7b-instruct q5_K_M ~5.5 GB + qwen2.5vl:7b ~6-7 GB + DINOv2 ViT-S ~0.1-0.4 GB, total ~12-13 GB. If ever tight, `ROCKY_EMBED_DEVICE=cpu` runs DINOv2 on CPU at ~100-300 ms, fine for this path.
- **Constraint that shapes the design:** Ollama has no image-embedding API (ollama/ollama issues #4296, #5304), so the embedder cannot live in Ollama. It goes on the existing speech server (`server/app.py`, FastAPI on the PC, port 8123) as a new `/embed_image` endpoint, following its existing lazy-load pattern (`get_whisper()`, `get_piper()`).

Honest limit stated up front: v1 embeds the whole frame. That works when TJ holds the object up close for teaching and looking, and the threshold+margin guard keeps false positives out. A qwen2.5vl bbox-crop upgrade is noted for later if real-world matching is noisy, not built now.

## Design (v1)

- **Teach**: new `teach_object(name)` brain tool. Handler grabs `state.vision.get_jpeg()`, embeds via HTTP, appends to an event-sourced `objects.jsonl` in ~/.rocky_mini (max 3 shots per object; re-teaching the same name adds a shot). Always also records a plain `remember_fact("object", ...)` so the object enters Rocky's epistemic ledger / persona digest. No frame (sim, camera down): degrades to the text fact only, returns an in-character "remembered by word only" string. Never raises.
- **Recognize**: inside the existing `_look_handler` in `rocky_mini/app.py`. Since the curious-glance path (`turn.py` ~231-240) reuses the same handler, recognition automatically covers both the `look_around` tool and spontaneous glances, with zero new threads and no VisionThread changes (footgun 1 untouched). Match rule: max cosine over an object's shots, accept if `>= 0.60` and top1-top2 margin `>= 0.08` (both config-tunable; margin skipped with one enrolled object). On match, append "You recognize <name>. TJ taught you this object before." to the VLM reading. Below threshold: nothing appended; Rocky's existing curiosity ritual ("What is that, question?" / note_open_question) covers unknowns with no new code.
- **Kill switch**: `ROCKY_OBJECT_MEMORY_ENABLED=0`.

## Files

### New

1. **`rocky_mini/vision/embed.py`** — mirrors `vision/observe.py` exactly (Protocol + Fake + lazy-httpx real client, never raises):
   - `ImageEmbedder` Protocol: `embed(jpeg: bytes | None) -> list[float] | None`
   - `FakeImageEmbedder`: deterministic unit vector from sha256(jpeg), dim 64; `None` for `None`
   - `HttpImageEmbedder(base_url, timeout_s=10.0)`: lazy `import httpx` inside `embed()`, POST raw JPEG to `{base_url}/embed_image`, parse `{"vector": [...]}`, any fault -> `None`
   - `cosine(a, b) -> float`: pure python (no numpy in the robot app path), 0.0 on mismatch/zero norm
2. **`rocky_mini/memory/objects.py`** — `ObjectRecord` (Pydantic: id `o<N>`, name, embeddings, timestamps, deleted) and `ObjectStore(base_dir)` reusing `_JsonlLog` and `_normalize` from `memory/store.py`. Methods: `enroll(name, embedding) -> id` (append shot if name exists, cap 3, oldest dropped), `active_objects()`, `names()`, `delete_object(id)` (tombstone), `match(embedding, threshold, margin) -> (name, score) | None`.

### Modified

3. **`rocky_mini/config.py`** — add `object_memory_enabled: bool = True`, `embed_url: str = ""` (empty means use `speech_base_url`, line 44), `object_match_threshold: float = 0.60`, `object_match_margin: float = 0.08`, plus `resolved_embed_url()` helper.
4. **`rocky_mini/brain/tools.py`** — `TeachObjectArgs{name: str}` in `TOOL_MODELS`, description in `_TOOL_DESCRIPTIONS`: enroll the object TJ is showing right now via the camera instrument.
5. **`rocky_mini/app.py`** (`AppState.build`, ~lines 102-145) — build `embedder` next to the observer (`HttpImageEmbedder(settings.resolved_embed_url())` when `llm_backend == "ollama"`, else `FakeImageEmbedder`), `objects = ObjectStore(...)`, store both on state; wire `teach_object` handler; add the guarded recognition block to `_look_handler` (try/except, a glance must never break a turn).
6. **`rocky_mini/memory/store.py`** — one line: add `"objects.jsonl"` to `_LOG_FILES` so `export_zip`/`import_zip` carry visual memory.
7. **`rocky_mini/brain/persona.py`** — one static line in the output contract: when TJ shows a physical object and names it, call `teach_object`. Stays byte-stable per stage; check `tests/test_persona.py` assertions.
8. **`server/app.py`** — `/embed_image` endpoint: lazy `get_dino()` (torch.hub `facebookresearch/dinov2`, `ROCKY_EMBED_MODEL=dinov2_vits14`, `ROCKY_EMBED_DEVICE=cuda|cpu`), decode JPEG with PIL, 224x224 center crop, ImageNet normalize, `no_grad` forward, **L2-normalize server-side** (client cosine becomes a dot product with stable threshold semantics). Add `embed_loaded`/`embed_model` to `/health`; note torch+pillow in server README.

## Tests (all sim, no network/torch; 219 existing must stay green)

New `tests/test_visual_memory.py`:
- FakeImageEmbedder determinism, unit norm, `None` handling; different bytes land far below 0.60
- `HttpImageEmbedder("http://x").embed(None) is None` without importing httpx
- `cosine` basics (identical / orthogonal / dim mismatch)
- ObjectStore: enroll + reload persistence; 3-shot cap; threshold and margin rejection; single-object skips margin; tombstone survives reload
- `teach_object` in sim degrades to fact-only (no camera): fact present, no object record
- Teach-then-recognize with a stub vision object (`get_jpeg` returning fixed bytes): look_around output contains "recognize" + name; different bytes -> no recognition
- `teach_object` present in `tool_specs()` with correct schema
- Kill switch off: teach is fact-only even with a frame; look_around never appends recognition

Extend `tests/test_memory.py`: export_zip includes objects.jsonl; import round-trip; import of an old zip without objects.jsonl clears stale objects.

## decisions.md entry (no. 21, to log at implementation time)

Frozen DINOv2 embeddings, few-shot cosine match, no training. `/embed_image` on the speech server; `objects.jsonl` event-sourced sidecar in ~/.rocky_mini (covered by export_zip); recognition rides `_look_handler` (tool + curious glance); threshold 0.60 / margin 0.08 pending hardware tuning (probe-style, like decision 20); v1 whole-frame with bbox-crop as the noted upgrade; kill switch `ROCKY_OBJECT_MEMORY_ENABLED=0`. Reversal: flip the switch; the jsonl sidecar is inert.

## Milestones

- **M1 (sim core)**: `embed.py` + `objects.py` + config + `_LOG_FILES` line + their tests.
- **M2 (sim end-to-end)**: tools + app wiring + persona line + remaining tests. Full teach/recognize demo in the sim chat with stubbed vision. All tests green.
- **M3 (PC)**: `/embed_image` in `server/app.py` + docs. Curl-verified on 10.0.0.95 (first call downloads torch.hub weights once, PC-side only).
- **M4 (hardware probe)**: live enrollment and threshold tuning from logged top-2 scores; finalize numbers in decisions.md.

## Hardware bring-up checklist (M3/M4)

1. PC: `pip install torch pillow` in the server venv, restart uvicorn; `/health` shows `embed_loaded: false` (lazy) + model name.
2. `curl -X POST --data-binary @test.jpg http://10.0.0.95:8123/embed_image` -> 384-dim vector; time it (<100 ms GPU, <300 ms CPU).
3. `nvidia-smi` with qwen2.5 + qwen2.5vl + DINOv2 resident; if near 15 GB, `ROCKY_EMBED_DEVICE=cpu`.
4. Robot: `ROCKY_SPEECH_BASE_URL` reaches the PC; leave `ROCKY_EMBED_URL` unset.
5. Live teach: mug at 30-60 cm, "Rocky, this is my coffee mug"; verify tool fired and `~/.rocky_mini/objects.jsonl` grew. Repeat from 2-3 angles.
6. Move the mug, "look around" -> Rocky mentions it. Point camera elsewhere -> he must NOT (false-positive check).
7. Second distinct object: no cross-recognition; tune threshold/margin from logged scores.
8. Restart the app: recognition persists. Export memory zip: objects.jsonl inside.

## Verification

- `pytest tests/ -x` green (219 existing + ~12 new), all sim, no cv2/httpx/torch imports at module top.
- Sim demo: in the chat UI, "this is my coffee mug" triggers `teach_object` (fact-only in sim); with a stubbed frame in tests, teach-then-look round-trips recognition.
- Hardware checklist above is the real-world acceptance pass.

## Non-goals (scope guard)

- No fine-tuning or training of any model.
- No bbox detection/cropping in v1 (noted upgrade only).
- No new threads; VisionThread, MotionThread, Mixer ownership untouched (footguns 1-3, 9).
- No numpy/torch in the robot app path; heavy deps live only on the PC server.
