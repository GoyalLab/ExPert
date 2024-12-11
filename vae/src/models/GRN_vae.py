import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class GRNVariationalAutoencoder(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        use_gat: bool = True,
        gpu_mem: int = 50_000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize accuracy metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        
        # First layer
        if use_gat:
            self.encoder_layers.append(GATConv(num_features, hidden_dim, edge_dim=hidden_dim))
        else:
            self.encoder_layers.append(GCNConv(num_features, hidden_dim))
            
        # Middle layers
        for _ in range(num_layers - 1):
            if use_gat:
                self.encoder_layers.append(GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            else:
                self.encoder_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Latent space projections for node features
        self.mu_node = nn.Linear(hidden_dim, latent_dim)
        self.log_var_node = nn.Linear(hidden_dim, latent_dim)
        
        # Latent space projections for edge weights
        self.mu_edge = nn.Linear(hidden_dim, latent_dim)
        self.log_var_edge = nn.Linear(hidden_dim, latent_dim)
        
        # Classifier in latent space
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Decoder for node features
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features)
        )
        
        # Decoder for edge weights
        self.edge_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Loss weights
        self.node_reconstruction_weight = 1.0
        self.edge_reconstruction_weight = 1.0
        self.kl_weight = 0.1
        self.classification_weight = 1.0
        self.gpu_mem = gpu_mem
        
    def encode(self, x, edge_index, edge_attr, batch):
        """Graph encoding"""
        # Free up memory
        torch.cuda.empty_cache()
        
        # Process in chunks if the graph is too large
        if x.size(0) > self.gpu_mem: 
            chunk_size = self.gpu_mem
            chunks = torch.split(x, chunk_size)
            edge_chunks = torch.split(edge_index, chunk_size, dim=1)
            outputs = []
            
            for x_chunk, edge_chunk in zip(chunks, edge_chunks):
                # Process chunk
                edge_embedding = self.edge_encoder(edge_attr)
                
                # Create edge batch tensor for this chunk
                edge_batch = batch[edge_chunk[0]]
                
                # Process through layers
                for layer in self.encoder_layers:
                    if isinstance(layer, GATConv):
                        x_chunk = layer(x_chunk, edge_chunk, edge_attr=edge_embedding)
                    else:
                        x_chunk = layer(x_chunk, edge_chunk)
                    x_chunk = F.relu(x_chunk)
                    x_chunk = F.dropout(x_chunk, p=self.hparams.dropout, training=self.training)
                
                outputs.append(x_chunk)
            
            # Combine chunks
            x = torch.cat(outputs, dim=0)
        else:
            # Original encoding logic for smaller graphs
            edge_embedding = self.edge_encoder(edge_attr)
            edge_batch = batch[edge_index[0]]
            
            for layer in self.encoder_layers:
                if isinstance(layer, GATConv):
                    x = layer(x, edge_index, edge_attr=edge_embedding)
                else:
                    x = layer(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        
        # Efficient pooling
        node_embedding = global_mean_pool(x, batch)
        edge_embedding = global_mean_pool(edge_embedding, edge_batch)
        
        # Get latent parameters
        mu_node = self.mu_node(node_embedding)
        log_var_node = self.log_var_node(node_embedding)
        mu_edge = self.mu_edge(edge_embedding)
        log_var_edge = self.log_var_edge(edge_embedding)
        
        return mu_node, log_var_node, mu_edge, log_var_edge
    
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode_node_features(self, z_node, batch):
        """
        Decode node features from latent space.
        
        Args:
            z_node (torch.Tensor): Latent vectors for graphs [num_graphs, latent_dim]
            batch (torch.Tensor): Batch tensor mapping nodes to graphs [num_nodes]
        """
        # Get the number of unique graphs
        num_graphs = len(torch.unique(batch))
        
        # If z_node is not already expanded, expand it
        if z_node.shape[0] != num_graphs:
            z_node = z_node.view(num_graphs, -1)
        
        # Expand graph embeddings to node embeddings
        z_expanded = z_node[batch]  # [num_nodes, latent_dim]
        
        # Decode to node features
        return self.node_decoder(z_expanded)
    
    def decode_edge_weights(self, z_node, edge_index):
        """
        Decode edge weights from node embeddings.
        
        Args:
            z_node (torch.Tensor): Node embeddings [num_nodes, latent_dim]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
        """
        # Safety checks
        device = z_node.device
        num_nodes = z_node.size(0)
        
        # Ensure edge indices are valid
        if torch.max(edge_index) >= num_nodes:
            raise ValueError(f"Edge index {torch.max(edge_index)} >= number of nodes {num_nodes}")
            
        # Process in chunks if the number of edges is too large
        chunk_size = self.gpu_mem
        num_edges = edge_index.size(1)
        
        if num_edges > chunk_size:
            outputs = []
            for i in range(0, num_edges, chunk_size):
                end_idx = min(i + chunk_size, num_edges)
                edge_chunk = edge_index[:, i:end_idx]
                
                # Get source and target embeddings for this chunk
                src_idx = edge_chunk[0]
                dst_idx = edge_chunk[1]
                
                src_embedding = z_node[src_idx]
                dst_embedding = z_node[dst_idx]
                
                # Concatenate and decode
                edge_input = torch.cat([src_embedding, dst_embedding], dim=-1)
                chunk_output = self.edge_decoder(edge_input)
                outputs.append(chunk_output)
                
            # Combine all chunks
            return torch.cat(outputs, dim=0)
        else:
            # Get source and target embeddings
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
                
            src_embedding = z_node[src_idx]
            dst_embedding = z_node[dst_idx]
                
            # Concatenate and decode
            edge_input = torch.cat([src_embedding, dst_embedding], dim=-1)
            return self.edge_decoder(edge_input)
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass with memory-efficient processing.
        """
        # Clear any lingering memory
        torch.cuda.empty_cache()
        
        try:
            # Encode
            mu_node, log_var_node, mu_edge, log_var_edge = self.encode(x, edge_index, edge_attr, batch)
            
            # Sample latent vectors
            z_node = self.reparameterize(mu_node, log_var_node)
            z_edge = self.reparameterize(mu_edge, log_var_edge)
            
            # Make sure z_node is properly expanded for decoding
            num_nodes = x.size(0)
            z_node_expanded = z_node[batch]  # [num_nodes, latent_dim]
            
            # Decode node features and edge weights
            reconstructed_features = self.decode_node_features(z_node_expanded, batch)
            reconstructed_weights = self.decode_edge_weights(z_node_expanded, edge_index)
            
            # Classification using concatenated embeddings
            z_combined = torch.cat([z_node, z_edge], dim=-1)
            predictions = self.classifier(z_combined)
            
            return (reconstructed_features, reconstructed_weights, 
                    mu_node, log_var_node, mu_edge, log_var_edge, 
                    predictions)
                    
        except RuntimeError as e:
            print(f"Forward pass failed with error: {str(e)}")
            print(f"Shapes: x={x.shape}, edge_index={edge_index.shape}, batch={batch.shape}")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            raise
    
    def training_step(self, batch, batch_idx):
        # Get predictions
        (reconstructed_features, reconstructed_weights,
         mu_node, log_var_node, mu_edge, log_var_edge,
         predictions) = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Node feature reconstruction loss
        node_recon_loss = F.mse_loss(reconstructed_features, batch.x)
        
        # Edge weight reconstruction loss
        edge_recon_loss = F.mse_loss(reconstructed_weights.squeeze(), batch.edge_attr)
        
        # KL divergence for both node and edge latent spaces
        kl_loss_node = -0.5 * torch.mean(1 + log_var_node - mu_node.pow(2) - log_var_node.exp())
        kl_loss_edge = -0.5 * torch.mean(1 + log_var_edge - mu_edge.pow(2) - log_var_edge.exp())
        kl_loss = kl_loss_node + kl_loss_edge
        
        # Format labels and compute classification loss
        y = batch.y.float().view((predictions.shape[0], predictions.shape[1]))
        class_loss = F.binary_cross_entropy_with_logits(predictions, y)
        
        # Calculate accuracy
        pred_probs = torch.softmax(predictions, dim=1)
        targets = torch.argmax(y, dim=1)
        accuracy = self.train_acc(pred_probs, targets)
        
        # Total loss
        total_loss = (self.node_reconstruction_weight * node_recon_loss +
                     self.edge_reconstruction_weight * edge_recon_loss +
                     self.kl_weight * kl_loss +
                     self.classification_weight * class_loss)
        
        # Log metrics
        self.log('train_total_loss', total_loss, prog_bar=True)
        self.log('train_node_recon_loss', node_recon_loss, prog_bar=True)
        self.log('train_edge_recon_loss', edge_recon_loss, prog_bar=True)
        self.log('train_kl_loss', kl_loss, prog_bar=True)
        self.log('train_class_loss', class_loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Get predictions
        (reconstructed_features, reconstructed_weights,
         mu_node, log_var_node, mu_edge, log_var_edge,
         predictions) = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Node feature reconstruction loss
        node_recon_loss = F.mse_loss(reconstructed_features, batch.x)
        
        # Edge weight reconstruction loss
        edge_recon_loss = F.mse_loss(reconstructed_weights.squeeze(), batch.edge_attr)
        
        # Format labels and compute classification loss
        y = batch.y.float().view((predictions.shape[0], predictions.shape[1]))
        class_loss = F.binary_cross_entropy_with_logits(predictions, y)
        
        # Calculate accuracy
        pred_probs = torch.softmax(predictions, dim=1)
        targets = torch.argmax(y, dim=1)
        accuracy = self.val_acc(pred_probs, targets)
        
        # Log metrics
        self.log('val_node_recon_loss', node_recon_loss, prog_bar=True)
        self.log('val_edge_recon_loss', edge_recon_loss, prog_bar=True)
        self.log('val_class_loss', class_loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        
        return {'val_loss': class_loss, 'val_acc': accuracy}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def visualize_latent_space(model, dataloader, device='cuda'):
    """
    Visualize the latent space using UMAP
    """
    import umap
    import numpy as np
    import matplotlib.pyplot as plt
    
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mu_node, _, mu_edge, _ = model.encode(
                batch.x, 
                batch.edge_index, 
                batch.edge_attr, 
                batch.batch
            )
            # Concatenate node and edge embeddings
            latent = torch.cat([mu_node, mu_edge], dim=1)
            latent_vectors.append(latent.cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    
    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    
    # Reduce dimensionality for visualization
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latent_vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    return plt.gcf()
