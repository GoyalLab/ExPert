import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.nn import FCLayers
import logging

class EmbeddingClassifier(nn.Module):
    """Classifier where latent attends to class embeddings, optionally externally provided,
    using dot product or cosine similarity. Compatible with external CE loss logic.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_labels: int = 5,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        logits: bool = True,  # <- for CE loss, return logits
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        class_embed_dim: int = 128,
        use_multihead: bool = False,
        use_cosine_similarity: bool = True,
        n_heads: int = 4,
        temperature: float = 0.05,
        return_latents: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.logits = logits
        self.use_multihead = use_multihead
        self.use_cosine_similarity = use_cosine_similarity
        self.class_embed_dim = class_embed_dim
        self.n_labels = n_labels
        self.temperature = temperature
        self.return_latents = return_latents

        # 1. Base encoder
        if n_hidden > 0 and n_layers > 0:
            self.encoder = FCLayers(
                n_in=n_input,
                n_out=class_embed_dim,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                activation_fn=activation_fn,
                **kwargs,
            )
        else:
            self.encoder = nn.Identity()
            class_embed_dim = n_input

        # 2. Check compatibility
        if use_multihead and class_embed_dim % n_heads != 0:
            fallback = 'cosine similarity' if use_cosine_similarity else 'dot product'
            logging.warning(f"class_embed_dim must be divisible by n_heads. Falling back to {fallback}.")
            self.use_multihead = False

        # 3. Class embeddings (used if external not provided)
        self.learned_class_embeds = nn.Parameter(torch.randn(n_labels, class_embed_dim))

        # 4. Optional attention layer
        if self.use_multihead:
            self.attn = nn.MultiheadAttention(
                embed_dim=class_embed_dim, num_heads=n_heads, batch_first=True
            )
        else:
            self.attn = None

    def forward(self, x, class_embeds=None, **kwargs):
        """
        Parameters
        ----------
        x : Tensor
            Input latent features, shape (batch, n_input)
        class_embeds : Optional[Tensor]
            External class embeddings, shape (n_labels, class_embed_dim)
        """
        z = self.encoder(x)  # (batch, d)

        if class_embeds is None:
            class_embeds = self.learned_class_embeds
        class_embeds = class_embeds.to(z.device)

        if self.use_cosine_similarity:
            z_norm = F.normalize(z, dim=-1)            # (batch, d)
            c_norm = F.normalize(class_embeds, dim=-1) # (n_labels, d)
            logits = torch.matmul(z_norm, c_norm.T) / self.temperature    # (batch, n_labels)
            _z = z_norm
        elif self.attn is not None:
            z_seq = z.unsqueeze(1)  # (batch, 1, d)
            class_seq = class_embeds.unsqueeze(0).expand(z.size(0), -1, -1)  # (batch, n_labels, d)
            attn_out, _ = self.attn(z_seq, class_seq, class_seq)  # (batch, 1, d)
            attn_out = attn_out.squeeze(1)                        # (batch, d)
            logits = torch.matmul(attn_out, class_embeds.T)       # (batch, n_labels)
            _z = attn_out
        else:
            z = z.unsqueeze(1)  # (batch, 1, d)
            class_seq = class_embeds.unsqueeze(0).expand(z.size(0), -1, -1)  # (batch, n_labels, d)
            logits = torch.bmm(z, class_seq.transpose(1, 2)).squeeze(1)      # (batch, n_labels)
            _z = z
        l = logits if self.logits else F.softmax(logits, dim=-1)
        if self.return_latents:
            return l, _z
        else:
            return l
