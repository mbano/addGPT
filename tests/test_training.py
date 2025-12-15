"""
Smoke tests for training functionality.
"""

import torch
from train import get_lr
from config import TrainingConfig


def test_learning_rate_warmup():
    """Test learning rate warmup phase."""
    cfg = TrainingConfig(warm_up_steps=100, max_iters=1000, learning_rate=1e-3)
    
    # At step 0, lr should be 0
    assert get_lr(0, cfg) == 0.0
    
    # At step 50 (halfway), lr should be ~half of peak
    lr_50 = get_lr(50, cfg)
    assert 0.4e-3 < lr_50 < 0.6e-3
    
    # After warmup, lr should decay
    lr_200 = get_lr(200, cfg)
    lr_500 = get_lr(500, cfg)
    assert lr_200 > lr_500


def test_single_training_step():
    """Smoke test: single training step runs without error."""
    from omegaconf import OmegaConf
    from torch.utils.tensorboard import SummaryWriter
    from train import Trainer
    import tempfile
    
    cfg = OmegaConf.create({
        'data': {'max_src_len': 2, 'max_tgt_len': 3, 'plus_token': '+', 
                 'eq_token': '=', 'sos_token': '<', 'eos_token': '>'},
        'model': {'n_emb': 32, 'n_heads': 4, 'n_blocks': 1,
                  'dropout_emb': 0.0, 'dropout_att': 0.0, 
                  'dropout_proj': 0.0, 'dropout_ff': 0.0},
        'training': {'seed': 42, 'batch_size': 4, 'learning_rate': 1e-3,
                     'warm_up_steps': 10, 'weight_decay': 1e-2,
                     'eval_interval': 10, 'eval_iters': 5, 'max_iters': 50,
                     'max_norm': 1.0, 'early_stop_patience': 3,
                     'log_dir': 'runs', 'data_split': [0.8, 0.1, 0.1]}
    })
    
    device = torch.device('cpu')
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SummaryWriter(log_dir=tmpdir)
        trainer = Trainer(cfg, device, writer)
        
        # Get a batch and perform training step
        batch = next(iter(trainer.train_loader))
        loss = trainer.train_step(batch)
        
        # Check loss is valid
        assert isinstance(loss, float)
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))
        
        writer.close()
