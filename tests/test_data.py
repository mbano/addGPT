"""
Tests for data utilities.
"""

import pytest
import torch
from config import DataConfig
from data import (
    build_vocab,
    AdditionDataset,
    Tokenizer,
    collate_fn,
    make_sos,
    format_output
)


class TestVocabulary:
    """Tests for vocabulary building."""
    
    def test_build_vocab(self):
        """Test vocabulary construction."""
        cfg = DataConfig()
        char2tok, tok2char = build_vocab(cfg)
        
        # Check vocab size (10 digits + 4 special tokens)
        assert len(char2tok) == 14
        assert len(tok2char) == 14
        
        # Check digits
        for i in range(10):
            assert str(i) in char2tok
            assert char2tok[str(i)] == i
        
        # Check special tokens
        assert '+' in char2tok
        assert '=' in char2tok
        assert '<' in char2tok
        assert '>' in char2tok
    
    def test_vocab_symmetry(self):
        """Test char2tok and tok2char are inverses."""
        cfg = DataConfig()
        char2tok, tok2char = build_vocab(cfg)
        
        for char, tok in char2tok.items():
            assert tok2char[tok] == char


class TestAdditionDataset:
    """Tests for AdditionDataset."""
    
    def test_dataset_length(self):
        """Test dataset returns correct length."""
        cfg = DataConfig()
        n_examples = 100
        dataset = AdditionDataset(cfg, n_examples, seed=42)
        assert len(dataset) == n_examples
    
    def test_dataset_format(self):
        """Test dataset returns correct format."""
        cfg = DataConfig()
        dataset = AdditionDataset(cfg, 10, seed=42)
        problem, solution = dataset[0]
        
        # Check problem format: "a+b="
        assert '+' in problem
        assert problem.endswith('=')
        
        # Check solution is numeric string
        assert solution.isdigit()
    
    def test_dataset_correctness(self):
        """Test dataset generates correct additions."""
        cfg = DataConfig()
        dataset = AdditionDataset(cfg, 10, seed=42)
        
        for i in range(10):
            problem, solution = dataset[i]
            # Parse problem
            parts = problem.rstrip('=').split('+')
            a, b = int(parts[0]), int(parts[1])
            expected = a + b
            actual = int(solution)
            assert actual == expected
    
    def test_dataset_reproducibility(self):
        """Test same seed produces same data."""
        cfg = DataConfig()
        dataset1 = AdditionDataset(cfg, 10, seed=42)
        dataset2 = AdditionDataset(cfg, 10, seed=42)
        
        for i in range(10):
            assert dataset1[i] == dataset2[i]


class TestTokenizer:
    """Tests for Tokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initializes correctly."""
        cfg = DataConfig()
        char2tok, _ = build_vocab(cfg)
        tokenizer = Tokenizer(cfg, char2tok)
        
        assert tokenizer.max_src_len == cfg.max_src_len
        assert tokenizer.max_tgt_len == cfg.max_tgt_len
    
    def test_tokenize_shape(self):
        """Test tokenization produces correct shapes."""
        cfg = DataConfig()
        char2tok, _ = build_vocab(cfg)
        tokenizer = Tokenizer(cfg, char2tok)
        
        src, tgt = tokenizer.tokenize("123+456=", "579")
        
        assert len(src) == cfg.max_src_tokens
        assert len(tgt) == cfg.max_tgt_len
    
    def test_tokenize_padding(self):
        """Test tokenization pads correctly."""
        cfg = DataConfig(max_src_len=5)
        char2tok, _ = build_vocab(cfg)
        tokenizer = Tokenizer(cfg, char2tok)
        
        src, tgt = tokenizer.tokenize("1+2=", "3")
        
        # Source should be zero-padded: "00001+00002="
        assert len(src) == 12  # 5*2 + 2
        # Target should be zero-padded and possibly reversed
        assert len(tgt) == 6
    
    def test_target_reversal(self):
        """Test target reversal option."""
        cfg = DataConfig()
        char2tok, tok2char = build_vocab(cfg)
        
        # With reversal
        tokenizer_rev = Tokenizer(cfg, char2tok, reverse_target=True)
        _, tgt_rev = tokenizer_rev.tokenize("123+456=", "579")
        
        # Without reversal
        tokenizer_no_rev = Tokenizer(cfg, char2tok, reverse_target=False)
        _, tgt_no_rev = tokenizer_no_rev.tokenize("123+456=", "579")
        
        # Should be different
        assert tgt_rev != tgt_no_rev


class TestCollateFn:
    """Tests for batch collation."""
    
    def test_collate_shape(self):
        """Test collate_fn produces correct tensor shapes."""
        cfg = DataConfig()
        char2tok, _ = build_vocab(cfg)
        tokenizer = Tokenizer(cfg, char2tok)
        
        batch = [("123+456=", "579"), ("111+222=", "333")]
        enc_out, dec_in, dec_tgt = collate_fn(batch, tokenizer, cfg)
        
        batch_size = len(batch)
        assert enc_out.shape == (batch_size, cfg.max_src_tokens)
        assert dec_in.shape == (batch_size, cfg.max_tgt_tokens)
        assert dec_tgt.shape == (batch_size, cfg.max_tgt_tokens)
    
    def test_collate_types(self):
        """Test collate_fn produces correct tensor types."""
        cfg = DataConfig()
        char2tok, _ = build_vocab(cfg)
        tokenizer = Tokenizer(cfg, char2tok)
        
        batch = [("123+456=", "579")]
        enc_out, dec_in, dec_tgt = collate_fn(batch, tokenizer, cfg)
        
        assert enc_out.dtype == torch.long
        assert dec_in.dtype == torch.long
        assert dec_tgt.dtype == torch.long
    
    def test_collate_sos_eos(self):
        """Test SOS and EOS token placement."""
        cfg = DataConfig()
        char2tok, _ = build_vocab(cfg)
        tokenizer = Tokenizer(cfg, char2tok)
        
        batch = [("1+1=", "2")]
        enc_out, dec_in, dec_tgt = collate_fn(batch, tokenizer, cfg)
        
        # Dec_in should start with SOS
        assert dec_in[0, 0] == tokenizer.sos_id
        
        # Dec_tgt should end with EOS
        assert dec_tgt[0, -1] == tokenizer.eos_id


class TestMakeSOS:
    """Tests for SOS token generation."""
    
    def test_make_sos_shape(self):
        """Test make_sos produces correct shape."""
        cfg = DataConfig()
        char2tok, _ = build_vocab(cfg)
        device = torch.device('cpu')
        
        batch_size = 8
        sos = make_sos(batch_size, char2tok, device, cfg)
        
        assert sos.shape == (batch_size, 1)
        assert sos.dtype == torch.long
    
    def test_make_sos_value(self):
        """Test make_sos contains correct token."""
        cfg = DataConfig()
        char2tok, _ = build_vocab(cfg)
        device = torch.device('cpu')
        
        sos = make_sos(4, char2tok, device, cfg)
        expected_id = char2tok[cfg.sos_token]
        
        assert torch.all(sos == expected_id)


class TestFormatOutput:
    """Tests for output formatting."""
    
    def test_format_output_correct(self):
        """Test format_output with correct prediction."""
        cfg = DataConfig()
        char2tok, tok2char = build_vocab(cfg)
        
        # Create mock encoder output: "00123+00456="
        problem = "00123+00456="
        enc_out = torch.tensor([[char2tok[c] for c in problem]])
        
        # Create mock model output: "<975000>" (reversed, with SOS/EOS)
        # Note: actual output would be reversed "579" -> "975"
        output_str = "<975000>"
        output = torch.tensor([[char2tok[c] for c in output_str]])
        
        results = format_output(enc_out, output, tok2char, cfg)
        
        # Should return list of strings
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], str)
