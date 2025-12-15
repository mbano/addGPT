"""
Tests for configuration classes.
"""

import pytest
from config import DataConfig, ModelConfig, TrainingConfig, ExperimentConfig


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_initialization(self):
        """Test DataConfig with default values."""
        cfg = DataConfig()
        assert cfg.max_src_len == 5
        assert cfg.max_tgt_len == 6
        assert cfg.plus_token == '+'
        assert cfg.eq_token == '='
        assert cfg.sos_token == '<'
        assert cfg.eos_token == '>'
    
    def test_computed_fields(self):
        """Test auto-computed fields."""
        cfg = DataConfig(max_src_len=5)
        assert cfg.max_val == 99999  # 10^5 - 1
        assert cfg.max_src_tokens == 12  # 5*2 + 2 (two operands + + and =)
        assert cfg.max_tgt_tokens == 7   # 6 + 1 (answer + SOS/EOS)
    
    def test_custom_values(self):
        """Test DataConfig with custom values."""
        cfg = DataConfig(max_src_len=3, max_tgt_len=4)
        assert cfg.max_val == 999
        assert cfg.max_src_tokens == 8  # 3*2 + 2
        assert cfg.max_tgt_tokens == 5  # 4 + 1


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_initialization_required_params(self):
        """Test that required parameters must be provided."""
        with pytest.raises(TypeError):
            ModelConfig()  # Should fail without required params
    
    def test_valid_initialization(self):
        """Test ModelConfig with valid parameters."""
        cfg = ModelConfig(
            max_src_tokens=12,
            max_tgt_tokens=7,
            vocab_size=14
        )
        assert cfg.max_src_tokens == 12
        assert cfg.max_tgt_tokens == 7
        assert cfg.vocab_size == 14
        assert cfg.n_emb == 512  # default
        assert cfg.n_heads == 8   # default
    
    def test_custom_architecture(self):
        """Test custom architecture parameters."""
        cfg = ModelConfig(
            max_src_tokens=12,
            max_tgt_tokens=7,
            vocab_size=14,
            n_emb=256,
            n_heads=4,
            n_blocks=6
        )
        assert cfg.n_emb == 256
        assert cfg.n_heads == 4
        assert cfg.n_blocks == 6


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_default_initialization(self):
        """Test TrainingConfig with defaults."""
        cfg = TrainingConfig()
        assert cfg.seed == 42
        assert cfg.batch_size == 64
        assert cfg.max_iters == 3000
        assert len(cfg.data_split) == 3
        assert sum(cfg.data_split) == 1.0
    
    def test_computed_dataset_size(self):
        """Test dataset size computation."""
        cfg = TrainingConfig(batch_size=32)
        assert cfg.dataset_size == 100_000 * 32


class TestExperimentConfig:
    """Tests for ExperimentConfig."""
    
    def test_default_initialization(self):
        """Test ExperimentConfig creates sub-configs."""
        cfg = ExperimentConfig()
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.training, TrainingConfig)
    
    def test_custom_sub_configs(self):
        """Test ExperimentConfig with custom sub-configs."""
        data_cfg = DataConfig(max_src_len=3)
        train_cfg = TrainingConfig(batch_size=32)
        exp_cfg = ExperimentConfig(data=data_cfg, training=train_cfg)
        
        assert exp_cfg.data.max_src_len == 3
        assert exp_cfg.training.batch_size == 32
