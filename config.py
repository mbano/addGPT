"""
Configuration classes for the addGPT project.

Defines data, model, and training configurations using dataclasses
for type safety and clean organization.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """
    Configuration for dataset generation and tokenization.
    
    Attributes:
        max_src_len: Maximum length for each operand (e.g., 5 allows up to 99999)
        max_tgt_len: Maximum length for the result (should be max_src_len + 1)
        plus_token: Token representing addition operator
        eq_token: Token representing equals sign
        sos_token: Start-of-sequence token
        eos_token: End-of-sequence token
        max_val: Maximum value for operands (auto-computed)
        max_src_tokens: Total source sequence length (auto-computed)
        max_tgt_tokens: Total target sequence length (auto-computed)
    """
    max_src_len: int = 5
    max_tgt_len: int = 6
    plus_token: str = '+'
    eq_token: str = '='
    sos_token: str = '<'
    eos_token: str = '>'

    # the parameters below auto-fill

    max_val: int = field(init=False)
    max_src_tokens: int = field(init=False)
    max_tgt_tokens: int = field(init=False)

    def __post_init__(self):
        self.max_val: int = 10**self.max_src_len - 1
        # self.max_val: int = 50
        self.max_src_tokens = self.max_src_len * 2 + 2 # two operands, and the + and = chars
        self.max_tgt_tokens = self.max_tgt_len + 1 # answer, and either SOS or EOS tokens

@dataclass
class ModelConfig:
    """
    Configuration for transformer model architecture.
    
    Data-dependent parameters (must be provided):
        max_src_tokens: Maximum source sequence length
        max_tgt_tokens: Maximum target sequence length
        vocab_size: Size of the vocabulary
    
    Architectural hyperparameters:
        n_emb: Embedding dimension
        n_heads: Number of attention heads
        n_blocks: Number of transformer blocks
        dropout_emb: Dropout for embeddings
        dropout_att: Dropout for attention weights
        dropout_proj: Dropout for attention projection
        dropout_ff: Dropout for feed-forward layers
    """
    
    # Data-dependent parameters (must be provided)
    max_src_tokens: int
    max_tgt_tokens: int
    vocab_size: int

    # Architectural hyperparameters
    n_emb: int = 512
    n_heads: int = 8
    n_blocks: int = 4
    dropout_emb: float = 0.1
    dropout_att: float = 0.1
    dropout_proj: float = 0.1
    dropout_ff: float = 0.1

@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters and logistics.
    
    Attributes:
        seed: Random seed for reproducibility
        batch_size: Training batch size
        learning_rate: Peak learning rate for optimizer
        warm_up_steps: Number of warmup steps for learning rate schedule
        weight_decay: L2 regularization weight
        eval_interval: Steps between validation evaluations
        eval_iters: Number of batches to use for validation
        max_iters: Maximum training iterations
        max_norm: Gradient clipping norm
        early_stop_patience: Number of evaluations without improvement before stopping
        log_dir: Directory for TensorBoard logs
        data_split: Train/val/test split ratios
        dataset_size: Total dataset size (auto-computed)
    """
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 15e-5 # result from trials
    warm_up_steps: int = 300
    weight_decay: float = 1e-2 # 13e-7 from trials
    eval_interval: int = 500
    eval_iters: int = 200
    max_iters: int = 3000
    max_norm: float = 1.0
    early_stop_patience: int = 5
    log_dir: str = 'runs'
    data_split: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    dataset_size: int = field(init=False)

    def __post_init__(self):
        self.dataset_size = 100_000 * self.batch_size

@dataclass
class ExperimentConfig:
    """
    Top-level configuration combining data and training configs.
    
    Note: This class is maintained for backward compatibility but is not used
    with the Hydra configuration system. Use Hydra YAML configs instead.
    """
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
