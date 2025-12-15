"""
Tests for model architecture.
"""

import pytest
import torch
from config import ModelConfig
from model import (
    EmbeddingLayer,
    SelfAttention,
    CrossAttention,
    FeedForward,
    Block,
    Model
)


class TestEmbeddingLayer:
    """Tests for EmbeddingLayer."""
    
    def test_initialization(self):
        """Test embedding layer initializes correctly."""
        layer = EmbeddingLayer(
            dropout_emb=0.1,
            n_emb=64,
            vocab_size=14,
            context_size=12
        )
        assert layer.tok_emb_table.num_embeddings == 14
        assert layer.tok_emb_table.embedding_dim == 64
        assert layer.pos_emb_table.num_embeddings == 12
    
    def test_forward_shape(self):
        """Test embedding layer output shape."""
        layer = EmbeddingLayer(0.1, 64, 14, 12)
        x = torch.randint(0, 14, (4, 8))  # (batch_size=4, seq_len=8)
        
        output = layer(x)
        assert output.shape == (4, 8, 64)  # (B, T, n_emb)
    
    def test_forward_different_seq_len(self):
        """Test embedding with different sequence lengths."""
        layer = EmbeddingLayer(0.1, 64, 14, 12)
        
        for seq_len in [1, 5, 12]:
            x = torch.randint(0, 14, (2, seq_len))
            output = layer(x)
            assert output.shape == (2, seq_len, 64)


class TestSelfAttention:
    """Tests for SelfAttention."""
    
    def test_initialization_pytorch_mha(self):
        """Test SelfAttention with PyTorch MultiheadAttention."""
        attn = SelfAttention(
            n_emb=64,
            n_heads=4,
            tgt_size=10,
            dropout_att=0.1,
            dropout_proj=0.1,
            use_pytorch_mha=True
        )
        assert attn.use_pytorch_mha is True
        assert hasattr(attn, 'mha')
    
    def test_initialization_custom(self):
        """Test SelfAttention with custom implementation."""
        attn = SelfAttention(
            n_emb=64,
            n_heads=4,
            tgt_size=10,
            dropout_att=0.1,
            dropout_proj=0.1,
            use_pytorch_mha=False
        )
        assert attn.use_pytorch_mha is False
        assert hasattr(attn, 'Wqkv')
        assert hasattr(attn, 'proj')
    
    def test_forward_shape_pytorch(self):
        """Test SelfAttention output shape with PyTorch implementation."""
        attn = SelfAttention(64, 4, 10, 0.1, 0.1, use_pytorch_mha=True)
        x = torch.randn(2, 7, 64)  # (B=2, T=7, C=64)
        
        output, weights = attn(x)
        assert output.shape == (2, 7, 64)
    
    def test_forward_shape_custom(self):
        """Test SelfAttention output shape with custom implementation."""
        attn = SelfAttention(64, 4, 10, 0.1, 0.1, use_pytorch_mha=False)
        x = torch.randn(2, 7, 64)
        
        output, weights = attn(x)
        assert output.shape == (2, 7, 64)
    
    def test_n_heads_divisibility(self):
        """Test that n_emb must be divisible by n_heads."""
        with pytest.raises(AssertionError):
            SelfAttention(65, 4, 10, 0.1, 0.1)  # 65 not divisible by 4


class TestCrossAttention:
    """Tests for CrossAttention."""
    
    def test_initialization(self):
        """Test CrossAttention initializes correctly."""
        attn = CrossAttention(
            n_emb=64,
            n_heads=4,
            dropout_att=0.1,
            dropout_proj=0.1
        )
        assert attn.n_emb == 64
        assert attn.n_heads == 4
        assert attn.head_size == 16
    
    def test_forward_shape(self):
        """Test CrossAttention output shape."""
        attn = CrossAttention(64, 4, 0.1, 0.1)
        enc_out = torch.randn(2, 10, 64)  # (B=2, eT=10, C=64)
        dec_in = torch.randn(2, 7, 64)    # (B=2, dT=7, C=64)
        
        output, weights = attn(enc_out, dec_in)
        assert output.shape == (2, 7, 64)  # Same shape as dec_in


class TestFeedForward:
    """Tests for FeedForward."""
    
    def test_initialization(self):
        """Test FeedForward initializes correctly."""
        ff = FeedForward(n_emb=64, dropout_ff=0.1)
        assert ff.n_emb == 64
    
    def test_forward_shape(self):
        """Test FeedForward output shape."""
        ff = FeedForward(64, 0.1)
        x = torch.randn(2, 7, 64)
        
        output = ff(x)
        assert output.shape == (2, 7, 64)  # Same as input
    
    def test_expansion(self):
        """Test FeedForward uses 4x expansion."""
        ff = FeedForward(64, 0.1)
        # proj_up should map to 4*n_emb
        assert ff.proj_up.out_features == 256  # 4 * 64


class TestBlock:
    """Tests for transformer Block."""
    
    def test_initialization(self):
        """Test Block initializes correctly."""
        block = Block(
            n_emb=64,
            n_heads=4,
            tgt_size=10,
            dropout_attn=0.1,
            dropout_proj=0.1,
            dropout_ff=0.1
        )
        assert hasattr(block, 'self_attention')
        assert hasattr(block, 'cross_attention')
        assert hasattr(block, 'ff')
    
    def test_forward_shape(self):
        """Test Block output shape."""
        block = Block(64, 4, 10, 0.1, 0.1, 0.1)
        enc_out = torch.randn(2, 10, 64)
        dec_in = torch.randn(2, 7, 64)
        
        output = block(enc_out, dec_in)
        assert output.shape == (2, 7, 64)  # Same as dec_in


class TestModel:
    """Tests for complete Model."""
    
    @pytest.fixture
    def model_config(self):
        """Create a small model config for testing."""
        return ModelConfig(
            max_src_tokens=12,
            max_tgt_tokens=7,
            vocab_size=14,
            n_emb=64,
            n_heads=4,
            n_blocks=2,
            dropout_emb=0.1,
            dropout_att=0.1,
            dropout_proj=0.1,
            dropout_ff=0.1
        )
    
    def test_initialization(self, model_config):
        """Test Model initializes correctly."""
        model = Model(model_config)
        assert len(model.blocks) == 2
        assert hasattr(model, 'enc_out_emb_layer')
        assert hasattr(model, 'dec_in_emb_layer')
    
    def test_forward_shape(self, model_config):
        """Test Model forward pass shapes."""
        model = Model(model_config)
        model.eval()
        
        batch_size = 2
        enc_out = torch.randint(0, 14, (batch_size, 12))  # (B, eT)
        dec_in = torch.randint(0, 14, (batch_size, 7))    # (B, dT)
        dec_tgt = torch.randint(0, 14, (batch_size, 7))   # (B, dT)
        
        logits, loss = model(enc_out, dec_in, dec_tgt)
        
        assert logits.shape == (batch_size, 7, 14)  # (B, dT, vocab_size)
        assert loss is not None
        assert loss.shape == ()  # Scalar
    
    def test_forward_without_target(self, model_config):
        """Test Model forward without target returns no loss."""
        model = Model(model_config)
        model.eval()
        
        enc_out = torch.randint(0, 14, (2, 12))
        dec_in = torch.randint(0, 14, (2, 7))
        
        logits, loss = model(enc_out, dec_in, dec_tgt=None)
        
        assert logits.shape == (2, 7, 14)
        assert loss is None
    
    def test_encode_shape(self, model_config):
        """Test encoder output shape."""
        model = Model(model_config)
        enc_input = torch.randint(0, 14, (2, 12))
        
        enc_out = model.encode(enc_input)
        assert enc_out.shape == (2, 12, 64)  # (B, eT, n_emb)
    
    def test_decode_shape(self, model_config):
        """Test decoder output shape."""
        model = Model(model_config)
        enc_out = torch.randn(2, 12, 64)
        dec_in = torch.randint(0, 14, (2, 7))
        
        logits = model.decode(enc_out, dec_in)
        assert logits.shape == (2, 7, 14)  # (B, dT, vocab_size)
    
    def test_generate_shape(self, model_config):
        """Test generation output shape."""
        model = Model(model_config)
        model.eval()
        
        enc_out = torch.randint(0, 14, (1, 12))
        start_token = torch.randint(0, 14, (1, 1))
        
        with torch.no_grad():
            output = model.generate(enc_out, start_token)
        
        # Should generate up to max_tgt_tokens
        assert output.shape[0] == 1
        assert output.shape[1] <= model_config.max_tgt_tokens
    
    def test_parameter_count(self, model_config):
        """Test model has reasonable parameter count."""
        model = Model(model_config)
        n_params = sum(p.numel() for p in model.parameters())
        
        # Small model should have between 100k and 500k params
        assert 50_000 < n_params < 1_000_000
    
    def test_trainable_parameters(self, model_config):
        """Test all parameters are trainable."""
        model = Model(model_config)
        
        for param in model.parameters():
            assert param.requires_grad is True


class TestModelCompatibility:
    """Tests for model compatibility and edge cases."""
    
    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        cfg = ModelConfig(12, 7, 14, n_emb=32, n_heads=4, n_blocks=1)
        model = Model(cfg)
        model.eval()
        
        for batch_size in [1, 2, 8, 16]:
            enc = torch.randint(0, 14, (batch_size, 12))
            dec = torch.randint(0, 14, (batch_size, 7))
            
            with torch.no_grad():
                logits, _ = model(enc, dec)
            
            assert logits.shape == (batch_size, 7, 14)
    
    def test_eval_vs_train_mode(self):
        """Test model behavior in eval vs train mode."""
        cfg = ModelConfig(12, 7, 14, n_emb=32, n_heads=4, n_blocks=1)
        model = Model(cfg)
        
        enc = torch.randint(0, 14, (2, 12))
        dec = torch.randint(0, 14, (2, 7))
        tgt = torch.randint(0, 14, (2, 7))
        
        # Train mode
        model.train()
        with torch.no_grad():
            logits_train, _ = model(enc, dec, tgt)
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(enc, dec, tgt)
        
        # Shapes should be the same
        assert logits_train.shape == logits_eval.shape
