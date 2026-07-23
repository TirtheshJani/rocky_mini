"""Vendor a curated set of recorded-emote trajectories into the repo.

This is a ONE-TIME developer build step, run on the dev PC only. It is never
run on the robot and is never imported at runtime. It downloads Pollen
Robotics' emotion dataset from HuggingFace into the local HF cache, then copies
a small allowlist of move JSON files into rocky_mini/assets/emotes/.

Those copied JSONs are committed to the repo. At runtime the robot loads them
straight off local disk, so it needs no HuggingFace access and no network at
all for emotes.

Each move JSON has keys:
  "description"      str
  "time"             list[float]   timestamps, one per frame
  "set_target_data"  list[frame]   each frame is a dict with
                                    "head"      4x4 nested list
                                    "antennas"  list
                                    "body_yaw"  float (optional)
Audio sidecar files may share a stem; they are ignored here.

Run:  pip install huggingface_hub   (dev-only build dep, not a runtime dep)
      python scripts/fetch_emotes.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "rocky_mini" / "assets" / "emotes"

REPO_ID = "pollen-robotics/reachy-mini-emotions-library"

# The curated moves we vendor. Match is by file stem; audio sidecars are ignored.
ALLOWLIST = [
    "amazed1",
    "laughing1",
    "sad1",
    "downcast1",
    "surprised1",
    "loving1",
    "grateful1",
    "scared1",
    "boredom1",
    "sleep1",
    "yes1",
    "no1",
    "welcoming2",
]


def download_dataset() -> Path:
    """Snapshot the dataset into the HF cache and return the local root path."""
    from huggingface_hub import snapshot_download

    local_root = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    return Path(local_root)


def find_json(root: Path, stem: str) -> Path | None:
    """Return the first <stem>.json found anywhere under root, else None."""
    matches = sorted(root.rglob(f"{stem}.json"))
    return matches[0] if matches else None

def list_available_stems(root: Path) -> list[str]:
    """All JSON stems present under root, sorted, for the not-found warning."""
    return sorted({p.stem for p in root.rglob("*.json")})


def move_stats(path: Path) -> tuple[int, float]:
    """Return (frame_count, duration_seconds) for a move JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("set_target_data") or []
    times = data.get("time") or []
    duration = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    return len(frames), duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor emote trajectories from HuggingFace.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Destination directory for the copied JSONs.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset {REPO_ID} into the HF cache ...")
    root = download_dataset()
    print(f"Dataset snapshot at: {root}")

    copied: list[tuple[str, int, float]] = []
    missing: list[str] = []

    for stem in ALLOWLIST:
        src = find_json(root, stem)
        if src is None:
            missing.append(stem)
            continue
        dst = out_dir / f"{stem}.json"
        shutil.copyfile(src, dst)
        frames, duration = move_stats(dst)
        copied.append((stem, frames, duration))

    if missing:
        available = list_available_stems(root)
        print()
        print(f"WARNING: {len(missing)} allowlisted move(s) not found: {missing}")
        print(f"JSON stems that WERE found in the dataset ({len(available)}):")
        for name in available:
            print(f"  {name}")

    print()
    print(f"Copied {len(copied)} of {len(ALLOWLIST)} moves into {out_dir}")
    for stem, frames, duration in copied:
        print(f"  {stem:<12} frames={frames:<5} duration={duration:.2f}s")


if __name__ == "__main__":
    main()
