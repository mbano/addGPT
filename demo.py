"""
Demo script for addGPT model inference.

Loads a trained model and performs addition on user-provided or random examples.
"""

import torch
import argparse
from pathlib import Path
from config import DataConfig, ModelConfig
from data import build_vocab, make_sos, format_output
from model import Model


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[Model, DataConfig]:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model in eval mode
        data_cfg: Data configuration from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['cfg']
    
    # Build vocab and model
    data_cfg = DataConfig(**cfg['data'])
    char2tok, tok2char = build_vocab(data_cfg)
    
    max_src_tokens = data_cfg.max_src_len * 2 + 2
    max_tgt_tokens = data_cfg.max_tgt_len + 1
    model_cfg = ModelConfig(
        max_src_tokens=max_src_tokens,
        max_tgt_tokens=max_tgt_tokens,
        vocab_size=len(char2tok),
        **cfg['model']
    )
    
    model = Model(model_cfg).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, data_cfg, char2tok, tok2char


def format_problem(a: int, b: int, data_cfg: DataConfig) -> str:
    """Format an addition problem as a string."""
    a_str = str(a).zfill(data_cfg.max_src_len)
    b_str = str(b).zfill(data_cfg.max_src_len)
    return f"{a_str}{data_cfg.plus_token}{b_str}{data_cfg.eq_token}"


def tokenize_problem(problem: str, char2tok: dict[str, int]) -> torch.Tensor:
    """Tokenize a problem string into a tensor."""
    tokens = [char2tok[c] for c in problem]
    return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension


def main():
    parser = argparse.ArgumentParser(description="Demo addGPT model inference")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/ckpt_step12000.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--problem',
        type=str,
        nargs=2,
        metavar=('A', 'B'),
        help='Two numbers to add (e.g., --problem 123 456)'
    )
    parser.add_argument(
        '--num-random',
        type=int,
        default=5,
        help='Number of random examples to generate (if --problem not specified)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        print("\nTo download the pretrained checkpoint:")
        print("  python download_checkpoint.py")
        return
    
    device = torch.device(args.device)
    print(f"Loading model from {args.checkpoint}...")
    model, data_cfg, char2tok, tok2char = load_checkpoint(args.checkpoint, device)
    print(f"Model loaded on {device}\n")
    
    # Generate problems
    if args.problem:
        # User-specified problem
        a, b = int(args.problem[0]), int(args.problem[1])
        problems = [(a, b)]
    else:
        # Random problems
        import random
        max_val = data_cfg.max_val
        problems = [(random.randint(0, max_val), random.randint(0, max_val)) 
                   for _ in range(args.num_random)]
    
    # Run inference
    print("=" * 60)
    for a, b in problems:
        # Format and tokenize problem
        problem_str = format_problem(a, b, data_cfg)
        enc_out = tokenize_problem(problem_str, char2tok).to(device)
        
        # Generate solution
        sos = make_sos(1, char2tok, device, data_cfg)
        with torch.no_grad():
            output = model.generate(enc_out, sos)
        
        # Format output
        results = format_output(enc_out.cpu(), output.cpu(), tok2char, data_cfg)
        for line in results:
            print(line)
        print("-" * 60)


if __name__ == '__main__':
    main()
