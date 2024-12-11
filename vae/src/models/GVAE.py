from tkinter.tix import Tree
from regex import T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap


class MPNNEncoder(MessagePassing):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MPNNEncoder, self).__init__(aggr='mean')  # Aggregation via mean
        
        # Define layers for node update
        self.node_update = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x, edge_index, edge_weight):
        # Message passing
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        print(out)
        # After message passing, update node features
        node_embeddings = self.node_update(out)
        return node_embeddings

    def message(self, x_j, edge_weight):
        # Edge feature used in message passing (weighted by edge weight)
        return edge_weight.view(-1, 1) * x_j  # Weighted node features

    def update(self, aggr_out):
        # Aggregated output from message passing
        return aggr_out

class EdgeEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EdgeEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)  # For concatenating node embeddings of both ends
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        h_i = x[edge_index[0]]  # Node embeddings for the source node of each edge
        h_j = x[edge_index[1]]  # Node embeddings for the target node of each edge
        edge_features = torch.cat([h_i, h_j], dim=-1)  # Concatenate node embeddings of both ends
        edge_embeddings = F.relu(self.fc1(edge_features))
        return self.fc2(edge_embeddings)


class GraphVAE(pl.LightningModule):
    def __init__(self, input_dim, num_classes,
                 hidden_dim=64, 
                 latent_dim=32,
                 lr=1e-3,
                 dropout_rate=0.1,
                 kl_weight=0.2,
                 node_weight=1,
                 edge_weight=1,
                 classifier_weight=1,
                 encoder='GCN'):
        super(GraphVAE, self).__init__()
        
        self.save_hyperparameters()
        
        # Encoder layers
        if encoder == 'GCN':
            self.encoder = nn.Sequential(
                GCNConv(input_dim, hidden_dim),
                GCNConv(hidden_dim, hidden_dim)
            )
        elif encoder == 'MPNN':
            self.encoder = MPNNEncoder(input_dim, hidden_dim, latent_dim)
        elif encoder == 'edge':
            self.encoder = EdgeEncoder(input_dim, hidden_dim, latent_dim)
        else:
            raise ValueError(f'Encoder has to be one of GCN, MPNN, or edge, got {encoder}')

        # VAE components
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers for node and edge reconstruction
        self.node_decoder = nn.Linear(hidden_dim, input_dim)
        
        # Optimized edge decoder
        self.edge_decoder_w1 = nn.Linear(hidden_dim, hidden_dim)
        self.edge_decoder_w2 = nn.Linear(hidden_dim, hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        # Edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        # Initialize metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        
        self.lr = lr
        self.kl_weight = kl_weight
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.classifier_weight = classifier_weight

    def encode(self, x, edge_index, edge_weight=None):
        # remove node features from latent space
        x = torch.zeros_like(x)
        h = F.relu(self.encoder(x, edge_index, edge_weight))
        
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, num_nodes):
        # Decode node features
        h1 = F.relu(self.decoder_fc1(z))
        h2 = F.relu(self.decoder_fc2(h1))
        node_features = self.node_decoder(h2)
        
        # Optimized edge reconstruction using matrix operations
        h_i = self.edge_decoder_w1(h1)  # [num_nodes, hidden_dim]
        h_j = self.edge_decoder_w2(h2)  # [num_nodes, hidden_dim]
        
        # Compute adjacency using matrix multiplication
        edge_weights = torch.mm(h_i, h_j.t())  # [num_nodes, num_nodes]
        
        return node_features, edge_weights

    def classify_edge(self, edge_weight, batch):
        """
        z: Node embeddings [total_nodes, latent_dim]
        batch: PyTorch Geometric batch assignment tensor [total_nodes]
        """
        graph_embedding = global_mean_pool(edge_weight, batch)

        return self.edge_classifier(graph_embedding)  # [num_graphs, num_classes]

    def classify(self, z, batch):
        """
        z: Node embeddings [total_nodes, latent_dim]
        batch: PyTorch Geometric batch assignment tensor [total_nodes]
        """
        # Initialize the aggregated edge weights for each graph
        graph_embedding = global_mean_pool(z, batch)
        # Pass the aggregated edge weights to the classifier
        return self.classifier(graph_embedding)  # [num_graphs, num_classes]
    
    def forward(self, x, edge_index, batch, edge_weight=None):
        mu, log_var = self.encode(x, edge_index, edge_weight)
        z = self.reparameterize(mu, log_var)
        reconstructed_nodes, reconstructed_edges = self.decode(z, x.size(0))
        class_logits = self.classify(z, batch)
        return reconstructed_nodes, reconstructed_edges, mu, log_var, class_logits

    def _get_losses(self, batch, mode):
        x, edge_index, edge_weight, y, b_idx = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
        # reformat all tensors to git batch size
        batch_size = x.size(0)
        
        # Create adjacency matrix
        adj_matrix = torch.zeros(batch_size, batch_size, device=self.device)
        adj_matrix[edge_index[0], edge_index[1]] = edge_weight.squeeze()
        
        # Forward pass with batch information
        reconstructed_nodes, reconstructed_edges, mu, log_var, class_logits = self(
            x, edge_index, b_idx, edge_weight
        )
        # Reconstruction losses
        node_recon_loss = F.mse_loss(reconstructed_nodes, x)
        edge_recon_loss = F.mse_loss(reconstructed_edges, adj_matrix.float())
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        
        # Classification loss
        pred_y = torch.sigmoid(class_logits)
        targets = y.view((class_logits.shape[0], class_logits.shape[1]))

        classification_loss = F.binary_cross_entropy_with_logits(targets, pred_y)
        
        # Total loss
        total_loss = self.node_weight * node_recon_loss + self.edge_weight * edge_recon_loss + self.kl_weight * kl_loss + self.classifier_weight * classification_loss
        
        # Calculate accuracy
        predictions = torch.argmax(pred_y, dim=1)
        true_labels = torch.argmax(targets, dim=1)
        if mode == 'train':
            accuracy = self.train_acc(predictions, true_labels)
        elif mode == 'val':
            accuracy = self.val_acc(predictions, true_labels)
        else:
            raise ValueError(f'Loss mode must be one of "train", "val", got "{mode}"')
        
        return {
            'loss': total_loss,
            'node_recon_loss': node_recon_loss,
            'edge_recon_loss': edge_recon_loss,
            'kl_loss': kl_loss,
            'classification_loss': classification_loss,
            'accuracy': accuracy
        }

    def training_step(self, batch, batch_idx):
        loss_dict = self._get_losses(batch, 'train')
        
        # Log all metrics
        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, prog_bar=True)
        
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        loss_dict = self._get_losses(batch, 'val')
        
        # Log all metrics
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, prog_bar=True)
        
        return loss_dict['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
def plot_latent_space(model, dataloader):
    model.eval()  # Set model to evaluation mode

    # List to store all node embeddings and corresponding labels
    node_embeddings = []
    labels = []

    # Loop through DataLoader to get batches
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Loading data', unit='batch', leave=True, position=0):
            # Send data to the same device as your model (e.g., GPU)
            data = data.to(model.device)

            # Get node-level embeddings
            reconstructed_nodes, reconstructed_edges, mu, log_var, class_logits = model(data.x, data.edge_index, data.batch, data.edge_attr)  # [num_nodes, latent_dim]

            # Store the embeddings and labels
            node_embeddings.append(reconstructed_nodes.cpu())  # Move to CPU if necessary
            labels.append(data.y.cpu())  # Node-level labels

    # Concatenate all node embeddings and labels
    node_embeddings = torch.cat(node_embeddings, dim=0)  # Shape: [num_nodes, latent_dim]
    labels = torch.cat(labels, dim=0)  # Shape: [num_nodes]

    # Apply dimensionality reduction (UMAP or t-SNE)
    umap_model = umap.UMAP(n_components=2, random_state=42)
    reduced_embeddings = umap_model.fit_transform(node_embeddings)  # Reduce to 2D

    # Plot the 2D UMAP representation
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral')
    plt.colorbar()
    plt.title("UMAP of Node Embeddings")
    plt.show()
