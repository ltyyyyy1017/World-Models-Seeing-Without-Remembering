"""Inspect an existing collection without loading Torch or sklearn."""
import json
from pathlib import Path
import numpy as np
from .data import load_data, SPLITS
from .env import World


def preflight(root):
    root = Path(root)
    result = {"oracle": {}, "datasets": [], "paired_trajectories": True,
              "scope": "Engineering checks only; not P1/P2/P3 learned-policy ablations."}
    lengths = []
    for seed in range(100):
        world = World(500000 + seed)
        for _ in range(80):
            world.step(world.oracle())
            if world.success:
                lengths.append(world.steps)
                break
    result["oracle"] = {"successes": len(lengths), "episodes": 100,
                         "mean_steps_successful": float(np.mean(lengths)), "budget": 80}
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    modes = sorted(set(key.split("_")[0] for key in manifest["files"]))
    for split in SPLITS:
        reference = None
        for mode in modes:
            data = load_data(root / "data", mode, split)
            mask = data["visible"]
            labels = data["identity"][mask]
            result["datasets"].append({"mode": mode, "split": split,
                "visible_samples": int(mask.sum()), "visible_classes": len(np.unique(labels)),
                "episodes": len(data["episode_ids"]),
                "majority_identity_accuracy": float(np.bincount(labels, minlength=30).max() / max(1, len(labels)))})
            if reference is not None:
                for key in ("episode_ids", "actions", "position", "rewards", "done", "visible"):
                    if not np.array_equal(reference[key], data[key]):
                        raise ValueError(f"Unpaired trajectory: {mode}/{split}/{key}")
            reference = data
    (root / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result
