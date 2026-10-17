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


def variance_covariance(z):
    """VICReg-inspired safeguard, shared by A/B/C and separately ablatable."""
    centered = z - z.mean(0)
    std = torch.sqrt(z.var(0, unbiased=False) + 1e-4)
    variance = F.relu(1 - std).mean()
    cov = centered.T @ centered / max(1, len(z) - 1)
    off_diagonal = cov - torch.diag(torch.diagonal(cov))
    return variance + 0.01 * off_diagonal.square().sum() / z.shape[1]


def loss_terms(model, x, nxt, actions, kind, regularization=0.1):
    z, target = model.encoder(x), model.encoder(nxt)
    pred = F.mse_loss(model.predict(z, actions), target.detach())
    rec = F.mse_loss(model.decoder(z), x) if kind == "A" else z.new_zeros(())
    inv = F.cross_entropy(model.inverse(torch.cat([z, target], 1)), actions) if kind == "C" else z.new_zeros(())
    base = {"A": 0.5 * rec + 0.5 * pred, "B": pred, "C": 0.5 * pred + 0.5 * inv}[kind]
    reg = (variance_covariance(z) + variance_covariance(target)) / 2 if regularization else z.new_zeros(())
    total = base + regularization * reg
    return total, {"loss": total.item(), "prediction": pred.item(), "reconstruction": rec.item(),
                   "inverse": inv.item(), "regularizer": reg.item(), "latent_std": z.std(0).mean().item()}
