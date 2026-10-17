import json
import os
import time
from pathlib import Path
import numpy as np
# CUDA requires this configuration for deterministic matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from .data import load_data, sha256
from .models import WorldModel, loss_terms


def image_tensor(images):
    return torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float().div_(255)


def train(data_root, output, mode, kind, seed=0, steps=300, latent=32,
          regularization=0.1, device="cpu", covariance_weight=0.01,
          prediction_objective="mse", temperature=0.2, reconstruction_weight=50.0,
          foreground_weight=0.0):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(seed)
    data = load_data(data_root, mode, "world_train")
    frames = data["frames"]
    action = data["actions"].reshape(-1)
    x = frames[:, :-1].reshape(-1, 64, 64, 3)
    nxt = frames[:, 1:].reshape(-1, 64, 64, 3)
    model = WorldModel(latent).to(device)
    # All heads instantiated even if unused: paired encoder initialization is exact.
    init_encoder = torch.cat([p.detach().cpu().flatten() for p in model.encoder.parameters()]).numpy().tobytes()
    import hashlib
    init_hash = hashlib.sha256(init_encoder).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    model.train()
    for step in range(steps):
        indices = rng.integers(len(x), size=64)
        obs = image_tensor(x[indices]).to(device)
        target = image_tensor(nxt[indices]).to(device)
        act = torch.from_numpy(action[indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = loss_terms(model, obs, target, act, kind, regularization,
                                   covariance_weight, prediction_objective, temperature,
                                   reconstruction_weight, foreground_weight)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite model loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step + 1, **metrics})
            print(f"{mode}/{kind} step {step+1}/{steps} loss={loss.item():.4f} std={metrics['latent_std']:.3f}", flush=True)
    model.eval()
    model.requires_grad_(False)
    config = {"kind": kind, "mode": mode, "seed": seed, "steps": steps, "latent": latent,
              "torch_version": str(torch.__version__), "numpy_version": np.__version__,
              "regularization": regularization, "covariance_weight": covariance_weight, "device": device,
              "prediction_objective": prediction_objective, "temperature": temperature,
              "reconstruction_weight": reconstruction_weight, "foreground_weight": foreground_weight,
              "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
              "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if str(device).startswith("cuda") else None,
              "training_seconds": round(time.perf_counter() - started, 2),
              "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
              "all_parameters": sum(p.numel() for p in model.parameters()),
              "initial_encoder_sha256": init_hash,
              "data_sha256": sha256(Path(data_root) / f"{mode}_world_train.npz"),
              "status": "pilot only; no hypothesis confirmation", "history": history}
    torch.save({"state_dict": model.state_dict(), "config": config}, output / "model.pt")
    (output / "training.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return model, config
