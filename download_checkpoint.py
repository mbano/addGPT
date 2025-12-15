"""
Download pretrained checkpoint from Hugging Face Hub.

This script downloads the trained model weights for addGPT from Hugging Face Hub.
"""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_checkpoint(repo_id: str, filename: str, cache_dir: str = None) -> Path:
    """
    Download checkpoint from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/addGPT')
        filename: Name of the checkpoint file
        cache_dir: Optional cache directory
        
    Returns:
        Path to downloaded checkpoint
    """
    print(f"Downloading {filename} from {repo_id}...")
    
    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True
        )
        print(f"✓ Downloaded to: {checkpoint_path}")
        
        # Create checkpoints directory and symlink
        checkpoints_dir = Path("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        
        local_path = checkpoints_dir / filename
        if not local_path.exists():
            local_path.symlink_to(checkpoint_path)
            print(f"✓ Created symlink: {local_path} -> {checkpoint_path}")
        
        return Path(checkpoint_path)
        
    except Exception as e:
        print(f"✗ Error downloading checkpoint: {e}")
        print("\nMake sure:")
        print("  1. The repository exists on Hugging Face Hub")
        print("  2. You have internet connection")
        print("  3. The checkpoint filename is correct")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download addGPT pretrained checkpoint from Hugging Face Hub"
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='m-bano/addGPT',
        help='Hugging Face repository ID (e.g., username/addGPT)'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='ckpt_step12500.pt',
        help='Checkpoint filename'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Cache directory for downloaded files'
    )
    
    args = parser.parse_args()
    
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface-hub")
        return
    
    # Download checkpoint
    try:
        checkpoint_path = download_checkpoint(
            args.repo_id,
            args.filename,
            args.cache_dir
        )
        print(f"\n✓ Success! Checkpoint ready at: {args.filename}")
        print(f"\nYou can now run inference with:")
        print(f"  python demo.py --checkpoint {args.filename}")
        
    except Exception as e:
        print(f"\n✗ Failed to download checkpoint")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
