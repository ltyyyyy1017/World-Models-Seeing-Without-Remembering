import json
import warnings
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from .data import load_data


def frozen_features(model, images):
    import torch
    from .train import image_tensor
    model.eval()
    model.requires_grad_(False)
    device = next(model.parameters()).device
    result = []
    with torch.no_grad():
        for start in range(0, len(images), 128):
            result.append(model.encoder(image_tensor(images[start:start + 128]).to(device)).cpu().numpy())
    return np.concatenate(result)


def pixel_reference(images, identities, seed=0):
    pixels = [x.reshape(len(x), 16, 4, 16, 4, 3).mean((2, 4)).reshape(len(x), -1) / 255 for x in images]
    pca = PCA(n_components=min(64, len(pixels[0]) - 1), random_state=seed)
    reduced = [pca.fit_transform(pixels[0]), pca.transform(pixels[1]), pca.transform(pixels[2])]
    return probe_suite(reduced, identities, 30, seed)


def pixel_audit(data_root, output, modes, seed=0):
    results = {}
    for mode in modes:
        images, identities = [], []
        for split in ("probe_train", "probe_val", "probe_test"):
            data = load_data(data_root, mode, split)
            mask = data["visible"].reshape(-1)
            images.append(data["frames"].reshape(-1, 64, 64, 3)[mask])
            identities.append(data["identity"].reshape(-1)[mask])
        results[mode] = pixel_reference(images, identities, seed)
        print(f"B0 {mode}: identity={results[mode]['selected_test_top1']:.3f}", flush=True)
    Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def scores(estimator, x, y, classes):
    probabilities = estimator.predict_proba(x)
    predicted = estimator.classes_[probabilities.argmax(1)]
    top = estimator.classes_[np.argsort(probabilities, axis=1)[:, -min(5, len(estimator.classes_)):]]
    balanced = float(np.mean([np.mean(predicted[y == label] == label) for label in np.unique(y)]))
    return {"top1": float(np.mean(predicted == y)), "balanced_top1": balanced,
            "top5": float(np.mean(np.any(top == y[:, None], axis=1))),
            "n": len(y), "confusion": confusion_matrix(y, predicted, labels=np.arange(classes)).tolist()}


def probe_suite(features, labels, classes, seed=0):
    train, val, test = features
    yt, yv, ys = labels
    if min(len(train), len(val), len(test)) < 10 or len(np.unique(yt)) < 2:
        raise ValueError("Too few visible probe samples or classes; increase collection budget.")
    estimators = {
        "linear": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=400, random_state=seed)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1,
                                      max_iter=200, random_state=seed)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(7, len(train))))}
    report = {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        for name, estimator in estimators.items():
            estimator.fit(train, yt)
            report[name] = {"train": scores(estimator, train, yt, classes),
                            "validation": scores(estimator, val, yv, classes),
                            "test": scores(estimator, test, ys, classes)}
    best = max(report, key=lambda name: report[name]["validation"]["top1"])
    # Select the family on validation. The test set never tunes hyperparameters.
    permutation = np.random.default_rng(seed + 993).permutation(yt)
    null = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=400, random_state=seed))
    null.fit(train, permutation)
    train_majority = int(np.bincount(yt, minlength=classes).argmax())
    return {"families": report, "selected_on_validation": best,
            "selected_test_top1": report[best]["test"]["top1"],
            "selected_test_top5": report[best]["test"]["top5"],
            "shuffled_train_labels_test_top1": scores(null, test, ys, classes)["top1"],
            "chance": 1 / classes, "train_class_coverage": len(np.unique(yt)),
            "test_class_coverage": len(np.unique(ys)),
            "test_label_counts": np.bincount(ys, minlength=classes).tolist(),
            "train_majority_class_test_accuracy": float(np.mean(ys == train_majority)),
            "convergence_warnings": sorted(set(str(w.message) for w in captured)),
            "linear_mlp_test_gap": report["mlp"]["test"]["top1"] - report["linear"]["test"]["top1"]}


def evaluate(data_root, output, mode, model, seed=0, raw_baseline=None):
    images, identities, positions, episodes = [], [], [], []
    for split in ("probe_train", "probe_val", "probe_test"):
        data = load_data(data_root, mode, split)
        visible = data["visible"].reshape(-1)
        images.append(data["frames"].reshape(-1, 64, 64, 3)[visible])
        identities.append(data["identity"].reshape(-1)[visible])
        positions.append(data["position"].reshape(-1)[visible])
        episodes.append(data["episode_ids"].tolist())
    features = [frozen_features(model, x) for x in images]
    for i in range(len(episodes)):
        for j in range(i):
            if set(episodes[i]) & set(episodes[j]):
                raise ValueError("Probe split episode overlap")
    wm_episodes = set(load_data(data_root, mode, "world_train")["episode_ids"].tolist())
    if any(wm_episodes & set(group) for group in episodes):
        raise ValueError("World-model/probe episode overlap")
    if raw_baseline is None:
        # B0: fixed 16x16 RGB observations, with training-only PCA.
        # This is a practical reference, NOT a proven information upper bound.
        raw_baseline = pixel_reference(images, identities, seed)
    identity = probe_suite(features, identities, 30, seed)
    position = probe_suite(features, positions, 16, seed)
    b0 = raw_baseline["selected_test_top1"]
    centered = features[2] - features[2].mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    prob = singular ** 2 / max(float((singular ** 2).sum()), 1e-12)
    rank = float(np.exp(-(prob[prob > 0] * np.log(prob[prob > 0])).sum()))
    result = {"mode": mode, "identity": identity, "position": position, "B0_pixel_reference": raw_baseline,
              "normalized_identity": identity["selected_test_top1"] / b0 if b0 else None,
              "latent_mean_std": float(features[2].std(0).mean()), "effective_rank": rank,
              "probe_episode_ids": episodes, "visible_samples": [len(x) for x in images],
              "scope": "Visible frames only. No policy success result; probes do not measure control ability.",
              "statistical_unit": "episode; frame scores descriptive, no independent-frame confidence intervals",
              "information_claim": "Successful decoding demonstrates retained information; failure is not absence."}
    Path(output, "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, raw_baseline
