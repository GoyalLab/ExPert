import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import scipy.sparse as sp
from src.models.base import Encoder, Decoder
from typing import List

import pdb
from sklearn.model_selection import train_test_split


class DNASequenceDataset(Dataset):
    def __init__(self, matrix):
        if not isinstance(matrix, sp.csr_matrix):
            self.data = sp.csr_matrix(matrix)
        else:
            self.data = matrix
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx: int):
        # Get expression data for the cell
        data = self.data[idx].toarray() if sp.issparse(self.data) else self.data[idx]
        data = torch.FloatTensor(data).squeeze()
        return data
    

class DNASequenceDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            matrix, 
            batch_size: int = 32, 
            train_val_split: float = 0.8, 
            stratify_by: List | None = None,
            shuffle: bool = True
        ):
        super().__init__()
        self.matrix = matrix
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.shuffle = shuffle
        self.stratify = stratify_by

    def setup(self, stage=None):
        dataset = DNASequenceDataset(self.matrix)
        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size

        if self.stratify is not None:
            indices = list(range(len(dataset)))
            train_indices, val_indices = train_test_split(
                indices, 
                train_size=train_size, 
                stratify=self.stratify
            )
            self.train_idc = train_indices
            self.val_idc = val_indices
            self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
            self.val_dataset = torch.utils.data.Subset(dataset, val_indices)
        else:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
            )

    def sparse_collate_fn(self, batch):
        # batch is a list of sparse tensors, one per sample
        return batch  # keep as list for your model to handle

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )

class DNASequenceVAE(pl.LightningModule):
    def __init__(
            self, 
            input_dim: int,
            latent_dim: int = 32, 
            hidden_dim: int = 128, 
            learning_rate: float = 1e-3,
        ):

        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        # Encoder layers
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=[hidden_dim],
            latent_dim=latent_dim
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=[hidden_dim],
            output_dim=input_dim
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
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
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_kl_loss', kl_loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def generate_sequences(model, num_sequences=10, device='cuda'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_sequences, model.latent_dim).to(device)
        sequences = model.decode(z)
    return sequences

def get_latent_space(model, dataloader, device='cuda'):
    model.eval()
    latent_space = []
    with torch.no_grad():
        model.to(device)
        for batch in dataloader:
            batch = batch.to(device)
            mu, logvar = model.encoder(batch)
            z = model.reparameterize(mu, logvar)
            latent_space.append(z.cpu())
    return torch.cat(latent_space, dim=0)

if __name__ == "__main__":
    # Example usage
    matrix = [[0, 1, 0, 1], [1, 0, 1, 0]]  # Replace with your 2D matrix
    dataset = DNASequenceDataset(matrix)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create and train VAE
    input_dim = len(matrix[0])
    model = DNASequenceVAE(input_dim=input_dim, latent_dim=16, hidden_dim=64)
    trainer = pl.Trainer(max_epochs=100, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, dataloader)
    
    # Generate new sequences
    new_sequences = generate_sequences(model, num_sequences=5, device='cuda' if torch.cuda.is_available() else 'cpu')
    for seq in new_sequences:
        print(seq)
