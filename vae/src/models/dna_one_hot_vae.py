import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import pdb


class DNASequenceDataset(Dataset):
    def __init__(self, matrix):
        self.one_hot = torch.tensor(matrix)
        
    def __len__(self):
        return len(self.one_hot)
    
    def __getitem__(self, idx):
        return self.one_hot[idx]

class DNASequenceDataModule(pl.LightningDataModule):
    def __init__(self, matrix, batch_size: int = 32, train_val_split: float = 0.8, shuffle: bool = True):
        super().__init__()
        self.matrix = matrix
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.shuffle = shuffle

    def setup(self, stage=None):
        dataset = DNASequenceDataset(self.matrix)
        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class DNASequenceVAE(pl.LightningModule):
    def __init__(
            self, 
            seq_length: int,
            n_nucleotides: int = 6,
            latent_dim: int = 32, 
            hidden_dim: int = 128, 
            kernel_size: int = 3, 
            padding: int = 1, 
            learning_rate: float = 1e-3,
        ):

        super().__init__()
        self.save_hyperparameters()
        
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        first_cvd = int(latent_dim / 2)
        
        # Encoder layers
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(n_nucleotides, first_cvd, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(first_cvd, latent_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.flatten = nn.Flatten()
        conv_output_size = latent_dim * (seq_length // 2)
        self.encoder_fc = nn.Linear(conv_output_size, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, conv_output_size),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, first_cvd, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(first_cvd, n_nucleotides, kernel_size=kernel_size, padding=padding),
        )
    
    def encode(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = F.relu(self.encoder_fc(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_fc(z)
        batch_size = x.size(0)
        x = x.view(batch_size, self.latent_dim, self.seq_length // 2)
        x = self.decoder_conv(x)
        x = x.permute(0, 2, 1)
        return F.softmax(x, dim=2)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
    
    def training_step(self, batch, batch_idx):
        reconstructed, mu, logvar = self(batch)
        recon_loss = F.mse_loss(reconstructed, batch, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon_loss', recon_loss, prog_bar=True)
        self.log('train_kl_loss', kl_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        reconstructed, mu, logvar = self(batch)
        recon_loss = F.mse_loss(reconstructed, batch, reduction='sum')
        if torch.isnan(recon_loss):
            pdb.set_trace()
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_kl_loss', kl_loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def probabilities_to_sequences(probs):
    nucleotides = ['N', 'A', 'C', 'G', 'T']
    sequences = []
    for seq_prob in probs:
        indices = torch.argmax(seq_prob, dim=1).cpu().numpy()
        seq = ''.join([nucleotides[idx] for idx in indices])
        sequences.append(seq)
    return sequences

def generate_sequences(model, num_sequences=10, device='cuda'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_sequences, model.latent_dim).to(device)
        probs = model.decode(z)
        sequences = probabilities_to_sequences(probs)
    return sequences

if __name__ == "__main__":
    # Example usage
    dataset = DNASequenceDataset(matrix)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create and train VAE
    model = DNASequenceVAE(seq_length=max_length, latent_dim=16, hidden_dim=64)
    trainer = pl.Trainer(max_epochs=100, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, dataloader)
    
    # Generate new sequences
    new_sequences = generate_sequences(model, num_sequences=5, device='cuda' if torch.cuda.is_available() else 'cpu')
    for seq in new_sequences:
        print(seq)
