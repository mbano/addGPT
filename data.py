"""
Data utilities for the addGPT project.

Handles vocabulary creation, dataset generation, tokenization, and batch collation
for training a transformer model to perform integer addition.
"""

import torch
from torch.utils.data import Dataset
from config import DataConfig
from torch import Tensor


def build_vocab(cfg: DataConfig) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build vocabulary mappings for the addition task.
    
    Vocabulary includes:
    - Digits 0-9
    - Special tokens: plus (+), equals (=), start-of-sequence (<), end-of-sequence (>)
    
    Args:
        cfg: Data configuration containing special token definitions
        
    Returns:
        char2tok: Dictionary mapping characters to token indices
        tok2char: Dictionary mapping token indices to characters
    """
    char2tok = {str(i):i for i in range(10)}
    char2tok[cfg.plus_token] = len(char2tok)
    char2tok[cfg.eq_token] = len(char2tok)
    char2tok[cfg.sos_token] = len(char2tok)
    char2tok[cfg.eos_token] = len(char2tok)
    tok2char = {v:k for k, v in char2tok.items()}

    return char2tok, tok2char

class AdditionDataset(Dataset):
    """
    Dataset for generating random addition problems.
    
    Generates examples like: "12345+67890=" -> "80235"
    Numbers are sampled uniformly up to max_val (determined by max_src_len).
    
    Args:
        cfg: Data configuration
        n_examples: Number of examples in the dataset
        seed: Random seed for reproducibility
    """

    def __init__(self, cfg: DataConfig, n_examples: int, seed: int) -> None:
        self.cfg = cfg
        self.n_examples = n_examples
        self.max_val = cfg.max_val
        self.seed = seed

    def __len__(self) -> int:
        return self.n_examples
    
    def __getitem__(self, idx: int) -> tuple[str, str]:
        """
        Generate a single addition example.
        
        Args:
            idx: Index of the example (used to seed RNG for reproducibility)
            
        Returns:
            problem: String like "12345+67890="
            solution: String like "80235"
        """
        g = torch.Generator().manual_seed(self.seed + idx)
        a = torch.randint(0, self.max_val+1, (), generator=g)
        b = torch.randint(0, self.max_val+1, (), generator=g)
        return f'{a}{self.cfg.plus_token}{b}{self.cfg.eq_token}', f'{a+b}'

class Tokenizer:
    """
    Tokenizer for addition problems.
    
    Handles formatting, padding, and tokenization of addition problems.
    Optionally reverses the target to help the model learn the addition algorithm
    from right to left (like manual addition).
    
    Args:
        cfg: Data configuration
        char2tok: Character to token mapping
        reverse_target: If True, reverse the target string to aid learning
    """
    def __init__(self, cfg: DataConfig, char2tok: dict[str, int], reverse_target: bool = True):
        self.char2tok = char2tok
        self.max_src_len = cfg.max_src_len
        self.max_tgt_len = cfg.max_tgt_len
        self.reverse_target = reverse_target
        self.sos_id = char2tok[cfg.sos_token]
        self.eos_id = char2tok[cfg.eos_token]
        self.plus_token = cfg.plus_token
        self.eq_token = cfg.eq_token

    def _format_example(self, src: str, tgt: str) -> tuple[str, str]:
        """
        Format and pad an addition example.
        
        Args:
            src: Source string like "123+456="
            tgt: Target string like "579"
            
        Returns:
            src_str: Formatted and padded source
            tgt_str: Formatted, padded, and optionally reversed target
        """
        
        a_str, b_str = src.rstrip(self.eq_token).split(self.plus_token)
        a_str = a_str.zfill(self.max_src_len)
        b_str = b_str.zfill(self.max_src_len)
        src_str = f'{a_str}{self.plus_token}{b_str}{self.eq_token}'
        tgt_str = tgt.zfill(self.max_tgt_len) # leaving more room for result
        if self.reverse_target:
            tgt_str = tgt_str[::-1] # reverse output to help learning summing algo
        return src_str, tgt_str
    
    def tokenize(self, src_str: str, tgt_str: str) -> tuple[list[int], list[int]]:
        """
        Tokenize a source-target pair.
        
        Args:
            src_str: Source string
            tgt_str: Target string
            
        Returns:
            src: List of token indices for source
            tgt: List of token indices for target
        """
        x_str, y_str = self._format_example(src_str, tgt_str)
        src = [self.char2tok.get(c) for c in x_str]
        tgt = [self.char2tok.get(c) for c in y_str]

        return src, tgt

def collate_fn(batch: list[tuple[str, str]], tokenizer: Tokenizer, cfg: DataConfig) -> tuple[Tensor, Tensor, Tensor]:
    """
    Collate function for batching examples.
    
    Prepares batches by:
    - Tokenizing source and target
    - Adding SOS token to decoder input
    - Adding EOS token to decoder target
    
    Args:
        batch: List of (source, target) string pairs
        tokenizer: Tokenizer instance
        cfg: Data configuration
        
    Returns:
        enc_out: Encoder input tensor of shape (B, max_src_tokens)
        dec_in: Decoder input tensor of shape (B, max_tgt_tokens) with SOS prepended
        dec_tgt: Decoder target tensor of shape (B, max_tgt_tokens) with EOS appended
    """

    batch_size = len(batch)
    enc_out = torch.zeros([batch_size, cfg.max_src_tokens], dtype=torch.long)
    dec_in = torch.zeros([batch_size, cfg.max_tgt_tokens], dtype=torch.long)
    dec_tgt = torch.zeros([batch_size, cfg.max_tgt_tokens], dtype=torch.long)

    for i, example in enumerate(batch):
        src, tgt = tokenizer.tokenize(example[0], example[1])
        enc_out[i] = torch.tensor(src)
        dec_in[i] = torch.tensor([tokenizer.sos_id] + tgt)
        dec_tgt[i] = torch.tensor(tgt + [tokenizer.eos_id])
        
    return (enc_out, dec_in, dec_tgt)

def make_sos(batch_size: int, char2tok: dict[str, int], device: torch.device, cfg: DataConfig) -> Tensor:
    """
    Create a batch of start-of-sequence tokens.
    
    Args:
        batch_size: Number of sequences
        char2tok: Character to token mapping
        device: Target device
        cfg: Data configuration
        
    Returns:
        Tensor of shape (batch_size, 1) containing SOS token indices
    """
    return torch.full((batch_size, 1), char2tok[cfg.sos_token], device=device, dtype=torch.long)

def format_output(enc_out: Tensor, output: Tensor, tok2char: dict[int, str], cfg: DataConfig) -> list[str]:
    """
    Format model outputs into human-readable strings and check correctness.
    
    Args:
        enc_out: Encoder input tensor of shape (B, T)
        output: Model output tensor of shape (B, T)
        tok2char: Token to character mapping
        cfg: Data configuration
        
    Returns:
        List of formatted strings showing problems, predictions, and correctness
    """
    B, _ = enc_out.shape
    out = []
    for i in range(B):
        problem = ''.join([tok2char[token] for token in enc_out[i].tolist()])
        solution = ''.join([tok2char[token] for token in output[i].tolist()])
        
        a, b = problem[:cfg.max_src_tokens].rstrip('=').split('+')
        a = int(a)
        b = int(b)
        c = solution[1:].rstrip('0')
        # c = solution[1:]

        if c != '':
            c = c[::-1]
        else:
            c = '0'

        # check addition
        sol = a + b
        try:
            c = int(c)
            out.append(f'{a}+{b}={c}')
            if sol == c:
                out.append('\tcorrect!')
            else:
                out.append(f'\twrong: {sol}, difference: {abs(sol - c)}')
        except ValueError:
            out.append(f'{a}+{b}={c}')
            out.append(f'\twrong: {sol}')

    return out
