import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Optional


class FeatureAttention(nn.Module):
    """
    Per-sample self-attention mechanism over input features.
    For each sample in the batch, computes attention between its features.
    """
    def __init__(self, n_input: int, head_size: int | None = None, dropout_rate: float = 0.1):
        super().__init__()
        self.head_size = n_input if head_size is None else head_size 
        self.query = nn.Linear(n_input, self.head_size, bias=False)
        self.key = nn.Linear(n_input, self.head_size, bias=False)
        self.value = nn.Linear(n_input, self.head_size, bias=False)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, feature_mask: torch.Tensor = None):
        """
        Args:
            x: Tensor of shape (batch, features)
            feature_mask: Optional mask of shape (batch, features) with 0s for features to mask

        Returns:
            attended: Tensor of shape (batch, features)
        """

        # Compute dot product between all features for each sample
        q = self.query(x)               # (batch, features)
        k = self.key(x)                 # (batch, features)
        v = self.value(x)               # (batch, features)

        # Compute attention: (batch, features, features)
        attn_weights = torch.bmm(q.unsqueeze(2), k.unsqueeze(1)) * self.head_size**-0.5     # per sample, scale by attention head size

        if feature_mask is not None:
            # Mask has shape (batch, features) --> (batch, 1, features)
            attn_weights = attn_weights.masked_fill(
                feature_mask.unsqueeze(1) == 0, -torch.inf
            )
        # Normalize attention weights to sum to 1
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch, features, features)
        # Apply dropout
        if self.dropout_rate > 0:
            attn_weights = self.dropout(attn_weights)
        # Apply attention: (batch, features, features) x (batch, features, 1) --> (batch, features, 1)
        attended = torch.bmm(attn_weights, v.unsqueeze(-1)).squeeze(-1)  # (batch, features)

        return attended
    

class MultiHeadAttentionIterative(nn.Module):
    """Multiple FeatureAttention heads in parallel."""

    def __init__(self, num_heads: int, n_input: int, head_size: int | None = None, dropout_rate: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([FeatureAttention(n_input=n_input, head_size=head_size) for _ in range(num_heads)])
        attn_dim = int(head_size * num_heads)
        self.proj = nn.Linear(attn_dim, n_input)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x
    

class AttentionHead(nn.Module):
    """Single attention head with its own projections"""
    def __init__(self, embedding_dim: int, head_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.head_size = head_size
        # Each head has its own projections
        self.q_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.k_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.v_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_emb: (B,F,E) embedded input
        Returns:
            out: (B,F,L) attention output
            attn: (B,F,F) attention weights
        """
        # Project to Q,K,V
        q = self.q_proj(x_emb)  # (B,F,L)
        k = self.k_proj(x_emb)  # (B,F,L)
        v = self.v_proj(x_emb)  # (B,F,L)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn = F.softmax(scores, dim=-1)  # (B,F,F)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)  # (B,F,L)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_input: int,
        num_heads: int,
        head_size: int | None = None,
        dropout_rate: float = 0.1,
        sequential: bool = True,
        emb_dim: int | None = None,
        buffer_n_step: int | None = None,
        min_buffer_steps: int = 5000,
        max_buffer_entries: int = 20,
        buffer_mode: Literal['train', 'val'] = 'train',
        use_ext_emb: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.n_input = n_input
        feature_input = n_input if emb_dim is None else emb_dim
        self.head_size = head_size if head_size is not None else feature_input // num_heads
        self.d_model = self.head_size * num_heads

        # learnable embeddings for each feature (gene)
        if not use_ext_emb:
            self.feature_emb = nn.Embedding(n_input, self.d_model)
        
        # Create separate attention heads
        self.heads = nn.ModuleList([
            AttentionHead(
                embedding_dim=self.d_model,
                head_size=self.head_size,
                dropout_rate=dropout_rate
            ) for _ in range(num_heads)
        ])

        # output projection: concat(H*L) -> 1
        self.out_proj = nn.Linear(self.d_model, 1, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.sequential = sequential
        # If not None and bigger than min steps, save a buffer of the attention layer
        self.record_each_n_step = buffer_n_step if buffer_n_step is not None and buffer_n_step > min_buffer_steps else None
        self.buffer_mode = buffer_mode
        self._buffer_idx = 0
        # Register buffer TODO: implement buffer_mode switch
        if self.record_each_n_step is not None:
            self.max_buffer_entries = max_buffer_entries
            self.register_buffer('attn_buffer', [], persistent=False)
        
    def forward(self, x: torch.Tensor, feature_emb: torch.Tensor | None = None, return_attn: bool = False):
        """
        Args:
            x: (B, F) expression matrix
            feature_emb (optional): (F, E) external feature embedding matrix
        Returns:
            out: (B, F)
            attn (optional): (B, H, F, F)  # only if return_attn=True
        """
        #self._buffer_idx += 1
        _record_step = False
        #_record_step = self.record_each_n_step is not None and self._buffer_idx % self.record_each_n_step == 0 and self.training
        # Get feature embeddings
        _feature_emb = feature_emb.T if feature_emb is not None else self.feature_emb.weight
        x_emb = torch.einsum("bf,fe->bfe", x, _feature_emb)  # (B,F,E)

        # Process each head
        out_heads = []
        attn_list = [] if return_attn else None

        # Run each attention head
        for head in self.heads:
            out_h, attn = head(x_emb)  # (B,F,L), (B,F,F)
            out_heads.append(out_h)
            if return_attn or _record_step:
                attn_list.append(attn.detach().cpu())

        # Concatenate outputs from all heads
        out = torch.cat(out_heads, dim=-1)  # (B,F,H*L)

        # Project back to output dimension
        out = self.out_proj(out).squeeze(-1)  # (B,F)
        out = self.dropout(out)

        # Save attention maps if requested
        if _record_step:
            if len(self.attn_buffer) > self.max_buffer_entries:
                self.register_buffer('attn_buffer', [], persistent=False)
            self.attn_buffer.append(torch.stack(attn_list, dim=1))  # (B,H,F,F)

        if return_attn:
            return out, (torch.stack(attn_list, dim=1) if attn_list else None)
        return out
    

class MultiHeadAttentionMM(nn.Module):
    """
    Multi-head feature self-attention (no sequence axis), loop-free via batched bmm.

    Shapes:
      x: (B, F) where F == n_input
      head_size = L  (defaults to n_input)
      q,k,v per head: (B, L)
      attention per head: (B, L, L)
      concat heads: (B, H*L)
      proj -> (B, n_input)

    Performance tips:
      - Keep record_attn=False during training (set True only when analyzing).
      - Use AMP/bfloat16 if possible.
      - Consider smaller head_size if L is large (cost ~ O(B*H*L^2)).
    """

    def __init__(
        self,
        num_heads: int,
        n_input: int,
        head_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        record_attn: bool = False,
        store_attn_on_cpu: bool = True,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.n_input = int(n_input)
        self.head_size = int(n_input if head_size is None else head_size)  # L
        self.scale = self.head_size ** -0.5

        H, L = self.num_heads, self.head_size

        # Single projections for all heads: (B, F) -> (B, H*L) -> view to (B, H, L)
        self.query = nn.Linear(self.n_input, H * L, bias=False)
        self.key   = nn.Linear(self.n_input, H * L, bias=False)
        self.value = nn.Linear(self.n_input, H * L, bias=False)

        # Concat heads -> project back to (B, n_input)
        self.proj = nn.Linear(H * L, self.n_input, bias=True)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

        # Attention recording controls
        self.record_attn = bool(record_attn)
        self.store_attn_on_cpu = bool(store_attn_on_cpu)
        self.register_buffer("last_attn_weights", torch.empty(0), persistent=False)

    @torch.no_grad()
    def _maybe_record(self, attn_bhll: torch.Tensor):
        if not self.record_attn:
            # keep an empty tensor to signal "not recording"
            self.last_attn_weights = attn_bhll.new_empty(0)
            return
        attn = attn_bhll.detach()
        if self.store_attn_on_cpu:
            attn = attn.to("cpu", non_blocking=True)
        self.last_attn_weights = attn  # shape: (B, H, L, L)

    def forward(
        self,
        x: torch.Tensor,                         # (B, F)
        feature_mask: Optional[torch.Tensor] = None,  # (B, L) with 1=keep, 0=mask (keys)
        return_attn: bool = False
    ):
        B, F = x.shape
        H, L = self.num_heads, self.head_size
        BH = B * H

        # Project all heads at once: (B, F) -> (B, H, L)
        q = self.query(x).view(B, H, L)
        k = self.key(x).view(B, H, L)
        v = self.value(x).view(B, H, L)

        # Flatten batch*heads for batched bmm
        q_bh = q.reshape(BH, L)                      # (BH, L)
        k_bh = k.reshape(BH, L)                      # (BH, L)
        v_bh = v.reshape(BH, L)                      # (BH, L)

        # Scores via outer products per (B,H): (BH, L, 1) @ (BH, 1, L) -> (BH, L, L)
        scores = torch.bmm(q_bh.unsqueeze(-1), k_bh.unsqueeze(-2))  # (BH, L, L)
        scores.mul_(self.scale)

        # Key mask: broadcast across query positions
        if feature_mask is not None:
            if feature_mask.shape != (B, L):
                raise ValueError(f"feature_mask must be (B, {L}) but got {tuple(feature_mask.shape)}")
            key_mask_bh = feature_mask.repeat_interleave(H, dim=0)          # (BH, L)
            # mask zeros (to -inf) along key axis
            scores = scores.masked_fill(key_mask_bh.unsqueeze(1) == 0,
                                        torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)          # (BH, L, L)
        if self.attn_dropout.p > 0:
            attn = self.attn_dropout(attn)

        # Weighted sum: (BH, L, L) @ (BH, L, 1) -> (BH, L)
        out_bh = torch.bmm(attn, v_bh.unsqueeze(-1)).squeeze(-1)  # (BH, L)

        # Restore (B, H, L) -> concat heads -> (B, H*L)
        out = out_bh.view(B, H, L).reshape(B, H * L)              # (B, H*L)

        # Final projection back to (B, n_input)
        out = self.proj(out)                                      # (B, n_input)
        if self.out_dropout.p > 0:
            out = self.out_dropout(out)

        # Record attention (reshape back to (B,H,L,L)) only if requested
        #self._maybe_record(attn.view(B, H, L, L))

        if return_attn:
            return out, self.last_attn_weights if self.last_attn_weights.numel() else attn.view(B, H, L, L)
        return out
