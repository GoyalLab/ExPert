import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import nbinom

# -------------------------------
# Credit to Ben Kutznetz-Speck :>
# -------------------------------
# ZINB Log-Likelihood Loss
# -------------------------------
class ZINBLoss(nn.Module):
    def forward(self, x, mu, theta, pi):
        eps = 1e-6
        mu = torch.clamp(F.softplus(mu), min=eps, max=1e3)
        theta = torch.clamp(F.softplus(theta), min=eps, max=1e3)
        pi = torch.clamp(pi, min=1e-4, max=1-1e-4)

        negbinom = (
            torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1)
            + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
            + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
        )

        zero_case = torch.log(pi + (1 - pi) * torch.exp(negbinom) + eps)
        nonzero_case = torch.log(1 - pi + eps) + negbinom

        loss = -torch.where(x < eps, zero_case, nonzero_case).mean()
        return loss


# -------------------------------
# Encoder with Noise Injection
# -------------------------------
class NoisyEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, latent_dim * 2)

    def forward(self, x):
        noise = torch.randn_like(x) * 0.05  # Input noise
        x = x + noise
        h = F.relu(self.ln1(self.fc1(x)))
        mean, logvar = torch.chunk(self.fc2(h), 2, dim=-1)
        mean = mean.clamp(min=-5, max=5)
        logvar = F.softplus(logvar).clamp(min=1e-4, max=2)
        return mean, logvar


# -------------------------------
# Decoder with LayerNorm
# -------------------------------
class ZINBDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2_mu = nn.Linear(128, output_dim)
        self.fc2_theta = nn.Linear(128, output_dim)
        self.fc2_pi = nn.Linear(128, output_dim)

    def forward(self, z):
        h = F.relu(self.ln1(self.fc1(z)))
        mu = torch.exp(self.fc2_mu(h))
        theta = torch.exp(self.fc2_theta(h)) + 1e-4
        pi = torch.sigmoid(self.fc2_pi(h)).clamp(1e-3, 0.95)
        return mu, theta, pi


# -------------------------------
# VAE Model with ZINB Prior
# -------------------------------
class ZINBVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = ZINBDecoder(latent_dim, input_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar).clamp(1e-3, 5)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        mu, theta, pi = self.decoder(z)
        return mu, theta, pi, mean, logvar
