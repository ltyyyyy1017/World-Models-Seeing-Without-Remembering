import torch
from torch import nn
from torch.nn import functional as F


class WorldModel(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(512, latent))
        self.dynamics = nn.Sequential(nn.Linear(latent + 5, 128), nn.ReLU(), nn.Linear(128, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 512), nn.ReLU(), nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, 2, 1), nn.Sigmoid())
        self.inverse = nn.Sequential(nn.Linear(latent * 2, 64), nn.ReLU(), nn.Linear(64, 5))

    def predict(self, z, action):
        return self.dynamics(torch.cat([z, F.one_hot(action, 5).float()], 1))


def variance_covariance(z, covariance_weight=0.01):
    """VICReg-inspired safeguard, shared by A/B/C and separately ablatable."""
    centered = z - z.mean(0)
    std = torch.sqrt(z.var(0, unbiased=False) + 1e-4)
    variance = F.relu(1 - std).mean()
    cov = centered.T @ centered / max(1, len(z) - 1)
    off_diagonal = cov - torch.diag(torch.diagonal(cov))
    return variance + covariance_weight * off_diagonal.square().sum() / z.shape[1]


def loss_terms(model, x, nxt, actions, kind, regularization=0.1, covariance_weight=0.01,
               prediction_objective="mse", temperature=0.2, reconstruction_weight=50.0,
               foreground_weight=0.0):
    z, target = model.encoder(x), model.encoder(nxt)
    predicted = model.predict(z, actions)
    if prediction_objective == "mse":
        pred = F.mse_loss(predicted, target.detach())
    elif prediction_objective == "contrastive":
        logits = F.normalize(predicted, dim=1) @ F.normalize(target.detach(), dim=1).T
        pred = F.cross_entropy(logits / temperature, torch.arange(len(z), device=z.device))
    else:
        raise ValueError(f"Unknown prediction objective: {prediction_objective}")
    if kind == "A":
        error = (model.decoder(z) - x).square()
        # Saturated pixels approximate the colorful person and folder in this
        # renderer. This is image-derived, not an identity or person label.
        saturation = x.amax(dim=1, keepdim=True) - x.amin(dim=1, keepdim=True)
        weights = 1 + foreground_weight * (saturation > 0.2).to(x.dtype)
        rec = (error * weights).sum() / (weights.sum() * x.shape[1])
    else:
        rec = z.new_zeros(())
    inv = F.cross_entropy(model.inverse(torch.cat([z, target], 1)), actions) if kind == "C" else z.new_zeros(())
    if prediction_objective == "mse":
        base = {"A": 0.5 * rec + 0.5 * pred, "B": pred, "C": 0.5 * pred + 0.5 * inv}[kind]
    else:
        base = {"A": pred + reconstruction_weight * rec, "B": pred, "C": pred + inv}[kind]
    reg = (variance_covariance(z, covariance_weight) +
           variance_covariance(target, covariance_weight)) / 2 if regularization else z.new_zeros(())
    total = base + regularization * reg
    return total, {"loss": total.item(), "prediction": pred.item(), "reconstruction": rec.item(),
                   "inverse": inv.item(), "regularizer": reg.item(), "latent_std": z.std(0).mean().item()}
