import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
# from matplotlib import pyplot as plt
from functools import partial
from config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from data import build_vocab, AdditionDataset, Tokenizer, collate_fn, make_sos, format_output
from model import Model
import optuna
from optuna.integration import TensorBoardCallback
import gc
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
import os
from hydra.utils import to_absolute_path


def get_lr(step, cfg, lr=0., decay=True):

    warm_up_steps = cfg.warm_up_steps
    max_steps = cfg.max_iters

    if not lr:
        lr = cfg.learning_rate

    # warm-up
    if step <= warm_up_steps:
        lr = lr * step / warm_up_steps
    # decay
    elif decay:
        if step >= max_steps:
            return 0.0
        # cosine
        lr = 0.5 * lr * (1 + math.cos(math.pi * (step - warm_up_steps) / (max_steps - warm_up_steps)))
        # linear
        # lr = lr * (1 - (step - warm_up_steps) / (max_steps - warm_up_steps))
    return lr

class Trainer:
    def __init__(self, 
                 cfg: DictConfig, 
                 device: torch.device, 
                 writer: SummaryWriter, 
                 ):
        
        self.cfg = cfg
        self.device = device
        self.logger = writer

        # build vocab

        self.char2tok, self.tok2char = build_vocab(self.cfg.data)
        self.tokenizer                = Tokenizer(self.cfg.data, self.char2tok)

        # build data loaders

        data_cfg = DataConfig(**cfg.data)
        self.train_cfg = TrainingConfig(**cfg.training)
        collate = partial(collate_fn, tokenizer=self.tokenizer, cfg=data_cfg)
        dataset = AdditionDataset(data_cfg, 
                                  self.train_cfg.dataset_size, 
                                  self.train_cfg.seed)
        
        train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, self.train_cfg.data_split)

        self.train_loader = DataLoader(train_ds, 
                                       batch_size=self.train_cfg.batch_size, 
                                       shuffle=False, 
                                       collate_fn=collate)
        self.val_loader = DataLoader(val_ds, 
                                     batch_size=self.train_cfg.batch_size, 
                                     shuffle=False, 
                                     collate_fn=collate)
        
        # build model, optimizer, scheduler

        vocab_size = len(self.char2tok)
        max_src_tokens = cfg.data.max_src_len * 2 + 2
        max_tgt_tokens = cfg.data.max_tgt_len + 1
        model_cfg = ModelConfig(
            max_src_tokens=max_src_tokens, 
            max_tgt_tokens=max_tgt_tokens, 
            vocab_size=vocab_size, 
            **cfg.model)
        model_cfg.vocab_size = vocab_size
 

        self.model = Model(model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr = self.train_cfg.learning_rate, 
            weight_decay = self.train_cfg.weight_decay
        )
        self.scheduler = LambdaLR(self.optimizer, 
                             lr_lambda=lambda step: get_lr(step, self.train_cfg) / self.train_cfg.learning_rate)
        
        # Trainer state
        self.best_val = float('inf')
        self.best_train = float('inf')
        self.step = 0

    def train_step(self, batch):
        '''Train on a single batch'''
        self.model.train()
        enc_out, dec_in, dec_tgt = batch
        enc_out = enc_out.to(self.device)
        dec_in = dec_in.to(self.device)
        dec_tgt = dec_tgt.to(self.device)
        # pass data to model
        logits, loss = self.model(enc_out, dec_in, dec_tgt)
        # reset gradient
        self.optimizer.zero_grad(set_to_none=True)
        # calculate gradient
        loss.backward()
        # update parameters
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                        max_norm=self.train_cfg.max_norm)
        self.optimizer.step()
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.add_scalar('LR', current_lr, self.step)

        return loss.item()

    def validate(self):
        '''Calculate losses on training and validation data'''
        train_total_loss, val_total_loss, n  = 0.0, 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= self.train_cfg.eval_iters: break
                _, loss = self.model(*[t.to(self.device) for t in batch])
                train_total_loss += loss.item()

            for i, batch in enumerate(self.val_loader):
                if i >= self.train_cfg.eval_iters: break
                _, loss = self.model(*[t.to(self.device) for t in batch])
                val_total_loss += loss.item()

                n += 1
            
        train_avg_loss = train_total_loss / n
        val_avg_loss = val_total_loss / n
        return train_avg_loss, val_avg_loss

    def run(self, save_callback=None):
        '''Run training, validation and logging'''
        no_improve = 0
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            if i >= self.train_cfg.max_iters: 
                break

            loss = self.train_step(batch)
            self.step += 1

            if i % self.train_cfg.eval_interval == 0:
                train_loss, val_loss = self.validate()
                self.logger.add_scalar('Loss/train', train_loss, i)
                self.logger.add_scalar('Loss/val', val_loss, i)

                # PRUNING HOOK
                if hasattr(self, '_trial'):
                    # report intermediate value for this step
                    self._trial.report(val_loss, step=i)
                    # ask pruner whether to stop
                    if self._trial.should_prune():
                        raise optuna.exceptions.TrialPruned

                # early stopping
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self.best_train = train_loss
                    best_step = i
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.train_cfg.early_stop_patience:
                        break

        if save_callback is not None:
            save_callback(self.model, self.optimizer, self.cfg, best_step)

        return self.best_val, self.best_train
    
    def register_pruner(self, trial: optuna.trial.Trial):
        '''Enable pruning using an Optuna trial'''
        self._trial = trial


def save_checkpoint(model, optimizer, cfg, step):
    # Create checkpoints directory if it doesn't exist
    ckpt_dir = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_path = os.path.join(ckpt_dir, f'ckpt_step{step}.pt')
    torch.save({
        'model_state': model.state_dict(), 
        'opt_state': optimizer.state_dict(), 
        'step': step, 
        'cfg': OmegaConf.to_container(cfg)
    }, ckpt_path)
    return ckpt_path

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg):
    print('Resolved config:\n', OmegaConf.to_yaml(cfg))
    # setup device, rng, directories
    torch.manual_seed(cfg.training.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_logdir = os.path.join(os.getcwd(), 'trials')

    def objective(trial: optuna.trial.Trial) -> float:
        # 1. Sample hyperparameters
        lr              = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        batch_size      = trial.suggest_categorical('batch_size', [32, 64, 128])
        n_emb           = trial.suggest_categorical('n_emb', [64, 128, 256, 512])
        n_heads         = trial.suggest_categorical('n_heads', [4, 8, 16, 32])
        n_blocks        = trial.suggest_categorical('n_blocks', [4, 8, 12])
        weight_decay    = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        # build config and Trainer
        cfg.training.learning_rate      = lr
        cfg.training.batch_size         = batch_size
        cfg.model.n_emb                 = n_emb
        cfg.model.n_heads               = n_heads
        cfg.model.n_blocks              = n_blocks
        cfg.training.weight_decay       = weight_decay

        logdir = f'{base_logdir}/trial-{trial.number}'
        logger = SummaryWriter(log_dir=logdir)
        trainer = Trainer(cfg, device, logger)

        # run training with pruning
        trainer.register_pruner(trial)

        try:
            best_val, best_train = trainer.run(save_callback=save_checkpoint)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                trial.set_user_attr('oom', True)
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.exceptions.TrialPruned()

        logger.close()

        return best_val
    
    # training loop

    tb_cb = TensorBoardCallback(base_logdir, metric_name='val_loss')
    study = optuna.create_study(
        direction='minimize', 
        sampler=optuna.samplers.TPESampler(), 
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=1
        ), 
    )
    study.optimize(objective, n_trials=20, callbacks=[tb_cb])


    # best_train, best_val = trainer.run(
    #     save_callback=lambda step: save_checkpoint(trainer.model, trainer.optimizer, cfg, step)
    # )
    # print(f'Best val loss: {best_val:.4f}')

if __name__ == '__main__':
    main()
