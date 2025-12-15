"""
addGPT: Teaching Transformers to Add Numbers

A transformer-based model that learns to perform integer addition, demonstrating
the capability of attention mechanisms to learn algorithmic tasks.

This implementation provides both custom attention mechanisms and PyTorch's
optimized MultiheadAttention for performance comparison.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class EmbeddingLayer(nn.Module):
    """
    Embedding layer combining token and positional embeddings.
    
    Args:
        dropout_emb: Dropout probability for embeddings
        n_emb: Embedding dimension
        vocab_size: Size of the vocabulary
        context_size: Maximum sequence length
    """
    def __init__(self, dropout_emb, n_emb, vocab_size, context_size):
        super().__init__()
        self.tok_emb_table = nn.Embedding(vocab_size, n_emb)
        self.pos_emb_table = nn.Embedding(context_size, n_emb)
        self.dropout = nn.Dropout(dropout_emb)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T) containing token indices
            
        Returns:
            Embedded representation of shape (B, T, n_emb)
        """
        seq_len = x.size(1)
        tok_emb = self.tok_emb_table(x)  # (B, T, n_emb)
        pos_emb = self.pos_emb_table(torch.arange(seq_len, device=x.device))  # (T, n_emb)
        pos_emb = pos_emb.unsqueeze(0)  # (1, T, n_emb) for broadcasting
        input_emb = tok_emb + pos_emb
        input_emb = self.dropout(input_emb)
        return input_emb


class SelfAttention(nn.Module):
    """
    Self-attention mechanism with causal masking for autoregressive generation.
    
    Supports both PyTorch's optimized MultiheadAttention and a custom implementation
    for educational/comparison purposes.
    
    Args:
        n_emb: Embedding dimension
        n_heads: Number of attention heads
        tgt_size: Maximum target sequence length
        dropout_att: Dropout probability for attention weights
        dropout_proj: Dropout probability for output projection
        use_pytorch_mha: If True, use nn.MultiheadAttention; otherwise use custom implementation
        use_flash: Only used with custom implementation; enables flash attention if True
    """
    def __init__(self, n_emb, n_heads, tgt_size, dropout_att, dropout_proj, use_pytorch_mha=True, use_flash=True):
        assert n_emb % n_heads == 0, f'n_emb needs to be divisible by n_heads'
        super().__init__()
        
        self.use_pytorch_mha = use_pytorch_mha
        self.n_emb = n_emb
        self.n_heads = n_heads
        self.head_size = n_emb // n_heads
        self.dropout_att_p = dropout_att
        self.flash = use_flash

        if use_pytorch_mha:
            # Use PyTorch's optimized MultiheadAttention
            self.mha = nn.MultiheadAttention(
                embed_dim=n_emb, 
                num_heads=n_heads, 
                dropout=dropout_att,
                batch_first=True
            )
            mask = torch.tril(torch.ones(tgt_size, tgt_size, dtype=torch.bool))
            self.register_buffer('causal_mask', mask)
        else:
            # Custom implementation
            self.Wqkv = nn.Linear(n_emb, 3 * n_emb)
            self.proj = nn.Linear(n_emb, n_emb)
            self.register_buffer('tril', torch.tril(torch.ones(tgt_size, tgt_size)))
            self.dropout_att = nn.Dropout(dropout_att)
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.Wqkv.weight, nonlinearity='linear')
            nn.init.zeros_(self.Wqkv.bias)
            nn.init.kaiming_uniform_(self.proj.weight, nonlinearity='linear')
            nn.init.zeros_(self.proj.bias)

        self.dropout_proj = nn.Dropout(dropout_proj)

    def forward(self, dec_in):
        """
        Args:
            dec_in: Input tensor of shape (B, T, C)
            
        Returns:
            out: Output tensor of shape (B, T, C)
            attn_weights: Attention weights (may be None for flash attention)
        """
        if self.use_pytorch_mha:
            return self._forward_pytorch(dec_in)
        else:
            return self._forward_custom(dec_in)
    
    def _forward_pytorch(self, dec_in):
        """Forward pass using PyTorch's MultiheadAttention"""
        B, T, C = dec_in.shape
        attn_mask = self.causal_mask[:T, :T]
        out, attn_weights = self.mha(
            dec_in, dec_in, dec_in,
            attn_mask=~attn_mask
        )
        out = self.dropout_proj(out)
        return out, attn_weights
    
    def _forward_custom(self, dec_in):
        """Forward pass using custom attention implementation"""
        B, T, C = dec_in.shape
        q, k, v = self.Wqkv(dec_in).chunk(3, dim=-1)  # (B, T, n_emb) -> 3 x (B, T, n_emb)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, T, head_size)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, T, head_size)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, T, head_size)

        if self.flash:
            # Use PyTorch's flash attention
            att = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout_att_p if self.training else 0, is_causal=True)
            weights = None
        else:
            # Manual attention computation
            weights = q @ k.transpose(-1, -2)  # (B, n_heads, T, T)
            weights *= self.head_size**-0.5
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            weights = F.softmax(weights, dim=-1)
            weights = self.dropout_att(weights)
            att = weights @ v  # (B, n_heads, T, head_size)
        
        # Concatenate heads
        att = att.transpose(1, 2).reshape(B, T, self.n_heads * self.head_size)  # (B, T, n_emb)
        out = self.proj(att)
        out = self.dropout_proj(out)
        return out, weights

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for attending to encoder outputs.
    
    Args:
        n_emb: Embedding dimension
        n_heads: Number of attention heads
        dropout_att: Dropout probability for attention weights
        dropout_proj: Dropout probability for output projection
        use_flash: If True, use PyTorch's flash attention; otherwise manual computation
    """
    def __init__(self, n_emb, n_heads, dropout_att, dropout_proj, use_flash=True):
        assert n_emb % n_heads == 0 # n_emb needs to be divisible by n_heads
        super().__init__()
        head_size = n_emb // n_heads
        self.n_emb = n_emb
        self.n_heads = n_heads
        self.head_size = head_size
        self.Wq = nn.Linear(n_emb, n_emb)
        self.Wkv = nn.Linear(n_emb, 2 * n_emb)
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout_att = nn.Dropout(dropout_att)
        self.dropout_proj = nn.Dropout(dropout_proj)
        self.flash = use_flash
        self.dropout_att_p = dropout_att

        # initializing weights
        nn.init.kaiming_uniform_(self.Wq.weight, nonlinearity='linear')
        nn.init.zeros_(self.Wq.bias)
        nn.init.kaiming_uniform_(self.Wkv.weight, nonlinearity='linear')
        nn.init.zeros_(self.Wkv.bias)
        nn.init.kaiming_uniform_(self.proj.weight, nonlinearity='linear')
        nn.init.zeros_(self.proj.bias)

    def forward(self, enc_out, dec_in):
        """
        Args:
            enc_out: Encoder output of shape (B, eT, C)
            dec_in: Decoder input of shape (B, dT, C)
            
        Returns:
            out: Output tensor of shape (B, dT, C)
            weights: Attention weights (may be None for flash attention)
        """
        B, eT, C = enc_out.shape
        _, dT, _ = dec_in.shape
        q = self.Wq(dec_in) # (B, dT, n_emb) @ (n_emb, n_emb) -> (B, dT, n_emb)
        q = q.view(B, dT, self.n_heads, self.head_size).transpose(1, 2) # (B, n_heads, dT, head_size)
        k, v = self.Wkv(enc_out).chunk(2, dim=-1) # (B, eT, n_emb) @ (n_emb, n_emb) -> (B, eT, n_emb)
        k = k.view(B, eT, self.n_heads, self.head_size).transpose(1, 2) # (B, n_heads, eT, head_size)
        v = v.view(B, eT, self.n_heads, self.head_size).transpose(1, 2) # (B, n_heads, eT, head_size)
        
        if self.flash:
            att = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout_att_p if self.training else 0, is_causal=False)
            weights = None
        else:
            weights = q @ k.transpose(-1, -2) # (B, n_heads, dT, head_size) @ (B, n_heads, head_size, eT) -> (B, n_heads, dT, eT)
            weights *= self.head_size**-0.5 # scaling for stability
            weights = F.softmax(weights, dim=-1) # normalize along encoder tokens - contribution to decoder tokens
            weights = self.dropout_att(weights)
            att = weights @ v # (B, n_heads, dT, eT) @ (B, n_heads, eT, head_size) -> (B, n_heads, dT, head_size)
        # concatenate attention heads
        att = att.transpose(1, 2).reshape(B, dT, self.n_emb) # (B, n_heads, dT, head_size) -> (B, dT, n_heads * head_size)
        # project to embedding space
        out = self.proj(att)
        out = self.dropout_proj(out)
        return out, weights

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Args:
        n_emb: Embedding dimension
        dropout_ff: Dropout probability
    """
    def __init__(self, n_emb, dropout_ff):
        super().__init__()
        self.n_emb = n_emb
        self.proj_up = nn.Linear(n_emb, 4 * n_emb)
        self.relu = nn.ReLU()
        self.proj_down = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(dropout_ff)

        nn.init.kaiming_uniform_(self.proj_up.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj_up.bias)
        nn.init.kaiming_uniform_(self.proj_down.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj_down.bias)

        self.net = nn.Sequential(
            self.proj_up, 
            self.relu, 
            self.proj_down,
            self.dropout
        )

    def forward(self, dec_in):
        return self.net(dec_in)

class Block(nn.Module):
    """
    Transformer decoder block with self-attention, cross-attention, and feed-forward layers.
    
    Args:
        n_emb: Embedding dimension
        n_heads: Number of attention heads
        tgt_size: Maximum target sequence length
        dropout_attn: Dropout for attention
        dropout_proj: Dropout for projection
        dropout_ff: Dropout for feed-forward network
    """
    def __init__(self, n_emb, n_heads, tgt_size, dropout_attn, dropout_proj, dropout_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_emb)
        self.self_attention = SelfAttention(n_emb, n_heads, tgt_size, dropout_attn, dropout_proj)
        self.ln2 = nn.LayerNorm(n_emb)
        self.cross_attention = CrossAttention(n_emb, n_heads, dropout_attn, dropout_proj)
        self.ln3 = nn.LayerNorm(n_emb)
        self.ff = FeedForward(n_emb, dropout_ff)

    def forward(self, enc_out, dec_in):
        """
        Args:
            enc_out: Encoder output of shape (B, eT, C)
            dec_in: Decoder input of shape (B, dT, C)
            
        Returns:
            Decoder output of shape (B, dT, C)
        """
        # Pre-norm architecture: normalize, then attention
        dec_in = self.ln1(dec_in)
        sa_dec_in, _ = self.self_attention(dec_in)  # Returns attention pattern
        dec_in = dec_in + sa_dec_in
        dec_in = self.ln2(dec_in)
        ca_dec_in, _ = self.cross_attention(enc_out, dec_in)
        dec_in = dec_in + ca_dec_in
        dec_in = self.ln3(dec_in)
        dec_in = dec_in + self.ff(dec_in)
        return dec_in
    
class Model(nn.Module):
    """
    Transformer encoder-decoder model for learning addition.
    
    The model learns to map addition problems (e.g., "00123+00456=") to their
    solutions (e.g., "975"), demonstrating that transformers can learn algorithmic tasks.
    
    Args:
        config: ModelConfig containing architecture parameters
    """
    def __init__(self, config):
        super().__init__()
        self.enc_out_emb_layer =EmbeddingLayer(
            config.dropout_emb, config.n_emb, config.vocab_size, config.max_src_tokens)
        self.dec_in_emb_layer = EmbeddingLayer(
            config.dropout_emb, config.n_emb, config.vocab_size, config.max_tgt_tokens)
        
        self.blocks = nn.ModuleList(
            [Block(config.n_emb, config.n_heads, config.max_tgt_tokens, config.dropout_att, 
                   config.dropout_proj, config.dropout_ff) for _ in range(config.n_blocks)])
        self.ln = nn.LayerNorm(config.n_emb)
        self.decoder_head = nn.Linear(config.n_emb, config.vocab_size)

    def encode(self, enc_out):
        """
        Encode the input sequence.
        
        Note: Current simplified implementation uses only embeddings.
        A full encoder would include self-attention and feed-forward layers.
        
        Args:
            enc_out: Input token indices of shape (B, eT)
            
        Returns:
            Encoded representation of shape (B, eT, n_emb)
        """
        enc_out = self.enc_out_emb_layer(enc_out)  # (B, eT, n_emb)
        return enc_out
    
    def decode(self, enc_out, dec_in):
        """
        Decode with cross-attention to encoder outputs.
        
        Args:
            enc_out: Encoder output of shape (B, eT, n_emb)
            dec_in: Decoder input token indices of shape (B, dT)
            
        Returns:
            Logits of shape (B, dT, vocab_size)
        """
        dec_in = self.dec_in_emb_layer(dec_in)  # (B, dT, n_emb)
        for block in self.blocks:
            dec_in = block(enc_out, dec_in)
        dec_in = self.ln(dec_in)
        logits = self.decoder_head(dec_in) # (B, dT, vocab_size)
        return logits

    def forward(self, enc_out, dec_in, dec_tgt=None):
        """
        Forward pass through the model.
        
        Args:
            enc_out: Encoder input token indices of shape (B, eT)
            dec_in: Decoder input token indices of shape (B, dT)
            dec_tgt: Target token indices of shape (B, dT) (optional, for training)
            
        Returns:
            logits: Predicted logits of shape (B, dT, vocab_size)
            loss: Cross-entropy loss (if dec_tgt provided, otherwise None)
        """
        enc_out = self.encode(enc_out)
        logits = self.decode(enc_out, dec_in)
        if dec_tgt is None:
            loss = None
        else:
            B, dT = dec_tgt.shape
            # compute loss
                # F.cross_entropy needs input to be shape (N, C), where N = batch size and C = channels
                # since it expects only N tokens (with C = vocab_size channels each)
                # we need to reshape to (B * dT, C)
                # For input with shape (N, C), target must be shape (N)
                # so both are flattened:
            loss = F.cross_entropy(logits.view(B * dT, -1), dec_tgt.view(B * dT)) 
        return logits, loss
    
    def generate(self, enc_out, out):
        """
        Generate output sequence autoregressively.
        
        Args:
            enc_out: Encoder input of shape (B, eT) with problem tokens
            out: Initial decoder input of shape (B, 1) with SOS token
            
        Returns:
            Generated sequence of shape (B, max_len) including SOS token
        """
        max_len = self.dec_in_emb_layer.pos_emb_table.num_embeddings
        self.eval()
        with torch.no_grad():
            for i in range(max_len - 1):
                logits, _ = self(enc_out, out)
                logits = logits[:, -1, :]  # Take last token logits
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                out = torch.cat((out, idx_next), dim=1)
            return out
