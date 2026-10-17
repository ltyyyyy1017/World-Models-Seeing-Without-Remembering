import argparse
from pathlib import Path
from .data import collect


def main():
    parser = argparse.ArgumentParser(description="Controlled world-model privacy pilot")
    parser.add_argument("command", choices=("collect", "pilot", "report", "train", "preflight", "evaluate", "pixels"))
    parser.add_argument("--output", default="outputs/pilot")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--covariance-weight", type=float, default=0.01)
    parser.add_argument("--prediction-objective", choices=("mse", "contrastive"), default="mse")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reconstruction-weight", type=float, default=50.0)
    parser.add_argument("--foreground-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modes", nargs="+", choices=("S0", "S1", "S2", "S3"), default=["S0", "S2", "S3"])
    parser.add_argument("--models", nargs="+", choices=("A", "B", "C"), default=["A", "B"])
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 1 or args.steps < 1:
        parser.error("episodes, horizon and steps must be positive")
    if (args.latent < 1 or args.regularization < 0 or args.covariance_weight < 0 or
            args.temperature <= 0 or args.reconstruction_weight < 0 or
            args.foreground_weight < 0 or args.seed < 0):
        parser.error("latent and temperature must be positive; weights and seed must be nonnegative")
    root = Path(args.output)
    if args.command == "preflight":
        from .preflight import preflight
        preflight(root)
    if args.command == "pixels":
        from .evaluate import pixel_audit
        pixel_audit(root / "data", root / "pixel_reference.json", args.modes, args.seed)
    if args.command in ("collect", "pilot"):
        if (root / "data/manifest.json").exists():
            raise FileExistsError("Choose a new output directory; existing datasets are preserved. Use train to reuse data.")
        collect(root / "data", args.episodes, args.horizon, args.seed, args.modes)
    if args.command in ("pilot", "train", "evaluate"):
        from .train import train
        from .evaluate import evaluate
        for mode in args.modes:
            baseline = None
            for kind in args.models:
                suffix = f"_cov{args.covariance_weight:g}" if args.covariance_weight != 0.01 else ""
                if args.prediction_objective == "contrastive":
                    suffix += f"_nceT{args.temperature:g}_rec{args.reconstruction_weight:g}"
                if args.foreground_weight:
                    suffix += f"_fg{args.foreground_weight:g}"
                run = root / "runs" / f"{mode}_{kind}_seed{args.seed}_reg{args.regularization:g}{suffix}"
                if args.command == "evaluate":
                    import torch
                    from .models import WorldModel
                    saved = torch.load(run / "model.pt", map_location=args.device, weights_only=True)
                    model = WorldModel(saved["config"]["latent"]).to(args.device)
                    model.load_state_dict(saved["state_dict"])
                else:
                    if (run / "model.pt").exists():
                        raise FileExistsError(f"Run already exists: {run}")
                    model, _ = train(root / "data", run, mode, kind, args.seed, args.steps,
                                     args.latent, args.regularization, args.device, args.covariance_weight,
                                     args.prediction_objective, args.temperature, args.reconstruction_weight,
                                     args.foreground_weight)
                result, baseline = evaluate(root / "data", run, mode, model, args.seed, baseline)
                print(f"{run.name}: identity={result['identity']['selected_test_top1']:.3f} position={result['position']['selected_test_top1']:.3f}", flush=True)
    if args.command in ("pilot", "train", "report", "evaluate"):
        from .report import report
        records = report(root)
        print(f"Report: {root.resolve() / 'index.html'} ({len(records)} runs)", flush=True)


if __name__ == "__main__":
    main()
