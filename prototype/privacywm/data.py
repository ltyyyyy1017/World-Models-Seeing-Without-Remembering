import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from PIL import Image
import PIL
from .env import World, identity_sequence, N_IDENTITIES

SPLITS = {"world_train": 0, "probe_train": 100000,
          "probe_val": 200000, "probe_test": 300000}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def schedule_audit(labels):
    flat = labels.reshape(-1)
    counts = np.bincount(flat, minlength=N_IDENTITIES)
    p = counts / counts.sum()
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    # Categories have no ordinal distance; report categorical repeat rate.
    repeat = float((labels[:, 1:] == labels[:, :-1]).mean())
    transitions = np.zeros((N_IDENTITIES, N_IDENTITIES), dtype=int)
    np.add.at(transitions, (labels[:, :-1].ravel(), labels[:, 1:].ravel()), 1)
    row = transitions.sum(axis=1)
    cond = 0.0
    for i in range(N_IDENTITIES):
        if row[i]:
            probs = transitions[i] / row[i]
            cond += row[i] / row.sum() * -(probs[probs > 0] * np.log2(probs[probs > 0])).sum()
    lag30 = float((labels[:, 30:] == labels[:, :-30]).mean()) if labels.shape[1] > 30 else None
    return {"marginal_entropy_bits": entropy, "maximum_entropy_bits": float(np.log2(30)),
            "lag1_repeat_rate": repeat, "empirical_conditional_entropy_bits": float(cond),
            "lag30_repeat_rate": lag30,
            "counts": counts.tolist(), "note": "Finite-sample audit; S2 is iid uniform, not exact balanced sampling."}


def collect(root, episodes=30, horizon=60, seed=0, modes=("S0", "S2", "S3")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "episodes_per_split": episodes, "horizon": horizon,
                "collector_versions": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__},
                "renderer": "software raycaster + geometric billboard; engineering fallback",
                "identity_definition": "6 shirt hues x 5 body-width/stripe bundles; collision radius fixed",
                "split_offsets": SPLITS, "files": {}, "schedule_audit": {}}
    for mode in modes:
        for split, offset in SPLITS.items():
            frames, actions, ids, positions, visible, eids, rewards, dones = [], [], [], [], [], [], [], []
            schedule_labels = []
            for episode in range(episodes):
                epseed = seed * 1000000 + offset + episode
                world = World(epseed)
                rng = np.random.default_rng(epseed + 9000000)
                sequence = identity_sequence(mode, horizon + 1, episode, seed)
                schedule_labels.append(sequence)
                observations, metadata, vis = [], [], []
                epactions, eprewards, epdones = [], [], []
                for step in range(horizon + 1):
                    world.identity = int(sequence[step])
                    rgb, seen = world.render()
                    observations.append(rgb)
                    metadata.append(world.metadata())
                    vis.append(seen)
                    if step < horizon:
                        # Exactly the same physical trajectory under every identity schedule.
                        action = int(rng.integers(5)) if world.success or rng.random() < 0.25 else world.oracle()
                        reward, done = world.step(action)
                        epactions.append(action)
                        eprewards.append(reward)
                        epdones.append(done)
                frames.append(observations)
                actions.append(epactions)
                ids.append(sequence)
                positions.append([m["position"] for m in metadata])
                visible.append(vis)
                eids.append(epseed)
                rewards.append(eprewards)
                dones.append(epdones)
            path = root / f"{mode}_{split}.npz"
            np.savez_compressed(path, frames=np.array(frames, dtype=np.uint8),
                                actions=np.array(actions, dtype=np.int64),
                                identity=np.array(ids), position=np.array(positions),
                                visible=np.array(visible), episode_ids=np.array(eids),
                                rewards=np.array(rewards, dtype=np.float32), done=np.array(dones))
            key = f"{mode}_{split}"
            manifest["files"][key] = {"path": path.name, "sha256": sha256(path),
                                         "visible_fraction": float(np.mean(visible))}
            manifest["schedule_audit"][key] = schedule_audit(np.array(schedule_labels))
            print(f"data {key}: {episodes * horizon} transitions", flush=True)
        # One physical trajectory replay; decoded probe values are added later.
        sample = np.load(root / f"{mode}_probe_test.npz")
        best = int(sample["visible"].sum(axis=1).argmax())
        gif = [Image.fromarray(x).resize((256, 256), Image.Resampling.NEAREST)
               for x in sample["frames"][best]]
        gif[0].save(root / f"replay_{mode}.gif", save_all=True, append_images=gif[1:], duration=160, loop=0)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_data(root, mode, split, verify=True):
    root = Path(root)
    path = root / f"{mode}_{split}.npz"
    if verify:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if sha256(path) != manifest["files"][f"{mode}_{split}"]["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
