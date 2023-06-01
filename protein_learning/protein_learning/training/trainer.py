"""Train, Test, and Validate a Model"""
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, AdamW

from protein_learning.common.data.datasets.dataset import ProteinDatasetABC
from protein_learning.common.data.model_data import ModelInput, ModelOutput, ModelLoss
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, default
from protein_learning.common.model_config import ModelConfig
from protein_learning.models.utils.model_abc import ProteinModel
from protein_learning.training.train_utils import (
    check_nan_in_grad,
    EvalTy,
    get_grad_norm,
    safe_round
)
import os

logger = get_logger(__name__)


class Trainer:
    """Model Trainer"""

    def __init__(self,
                 config: ModelConfig,
                 model: ProteinModel,
                 train_data: ProteinDatasetABC,
                 valid_data: Optional[ProteinDatasetABC] = None,
                 test_data: Optional[ProteinDatasetABC] = None,
                 ):

        self.config = config
        data_keys = [EvalTy.TRAINING.value, EvalTy.VALIDATION.value, EvalTy.TESTING.value]
        self.datasets = {k: v for k, v in zip(data_keys, [train_data, valid_data, test_data])}
        self.sample_idx, self.checkpoint, self.batch_idx = 0, 0, 0
        self.epoch, self.n_processed, self.n_valid_batch_samples = 0, 0, 0

        # set up model and optimizer
        self.model = model.to(config.device)
        self.optim = self._get_optim(config)
        self.epoch_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lambda x: config.decrease_lr_by ** max(1, x), last_epoch=-1)

        # potentially load from previous state
        if config.load_state:
            self.load_state(path=config.model_load_path)

        # optimize over per-sample gradients or average over batch
        self.clipped_grad = None
        self.use_amp = not config.clip_grad_per_sample
        self.scaler = GradScaler(enabled=self.use_amp)
        self.decrease_lr_every = config.decrease_lr_every if config.decrease_lr_every > 0 \
            else config.epochs + 1

    def train(self):
        """Train Model"""
        logger.info("Beginning Model Training")
        self._init_model()
        config = self.config
        start = time.time()
        cum_time, _n_processed = 0, 0
        for epoch in range(config.epochs):
            logger.info(f"Beginning training epoch : {epoch}")
            data = self.datasets[EvalTy.TRAINING.value]
            data.shuffle()
            train_data_loader = data.get_dataloader(num_workers=config.data_workers,
                                                    prefetch_factor=16,
                                                    batch_size=min(config.batch_size, 16),
                                                    shuffle=config.shuffle,
                                                    )
            for _, samples in enumerate(train_data_loader):
                self.batch_idx = self.n_processed // config.batch_size
                logger.info(f'[batch {self.batch_idx}], loaded {len(samples)} samples')
                for sample in samples:
                    with autocast(enabled=self.use_amp):
                        model_out, model_loss = self.process_sample(
                            sample,
                            eval_ty=EvalTy.TRAINING,
                            compute_zero_wt_loss=(self.n_processed + 1) % (config.batch_size // 2) == 0,
                        )
                    if model_out is None:
                        continue
                    self.n_processed += 1

                    # display loss
                    if self.n_processed % (config.batch_size // 2) == 0:
                        self._display_loss(model_loss, eval_ty=EvalTy.TRAINING)

                    # save pdbs (Optional)
                    if self.n_processed % (config.batch_size // 2) == 0:
                        if config.save_pdbs:
                            native = model_out.native_protein
                            native.to_pdb(os.path.join(config.pdb_dir, "native"))
                            native.to_pdb(os.path.join(config.pdb_dir, "decoy"),
                                          coords=model_out.predicted_coords[0])

                    model_loss, model_out = None, None  # Free memory
                    _n_processed += 1

                    # update model
                    if self.n_processed % config.batch_size == 0:
                        self.update_model()
                        cum_time += time.time() - start
                        avg_time = np.round(cum_time / _n_processed, 3)
                        print("##### average time per sample :", avg_time, "#####")
                        start = time.time()

                    # save the model
                    if self.n_processed % int(config.save_every * config.batch_size) == 0:
                        self.save_state()

                    # checkpoint model
                    if self.n_processed % int(config.checkpoint_every * config.batch_size) == 0:
                        self.checkpoint_model()

                    # run on validation targets
                    if self.n_processed % int(config.validate_every * config.batch_size) == 0:
                        if self.datasets[EvalTy.VALIDATION.value] is not None:
                            self._process_dataset(
                                eval_ty=EvalTy.VALIDATION,
                                max_samples=config.max_val_samples,
                            )
                            start = time.time()

                    # run on test targets
                    if self.n_processed % int(config.test_every * config.batch_size) == 0:
                        if self.datasets[EvalTy.TESTING.value] is not None:
                            self._process_dataset(
                                eval_ty=EvalTy.TESTING,
                                max_samples=config.max_test_samples
                            )
                            start = time.time()

            self.epoch = self.epoch + 1
            if (self.epoch % self.decrease_lr_every) == 0:
                self.epoch_scheduler.step()
            self.sample_idx = 0

    def update_model(self) -> None:
        """Update model (backward pass and optim.step)"""
        config = self.config
        updated = 0
        if config.clip_grad_per_sample:
            for name, param in self.model.named_parameters():
                if name not in self.clipped_grad:
                    continue
                param_grad = self.clipped_grad[name]
                if param_grad is not None:
                    if param.size()!=param_grad.size():
                        print(f"mismatch param {name}, param size : {param.size()}, grad size {param_grad.size()}")
                    param.grad = param_grad / max(1, self.n_valid_batch_samples)
            logger.info(f"Updating model on {self.n_valid_batch_samples}/"
                        f"{config.batch_size} batch samples, gradients for {updated} params")

        # global clip
        if not config.clip_grad_per_sample:
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.grad_norm_clip)

        gradient_norm = get_grad_norm(self.model)
        print(f"[INFO] global gradient norms {safe_round(gradient_norm)} at batch : {self.batch_idx}")
        has_nan = check_nan_in_grad(self.model)
        if not has_nan and not torch.isnan(gradient_norm):
            self.scaler.step(self.optim)
        else:
            logger.warn('caught NaN in optim.step()! skipping...')
        self.clipped_grad = None
        self.scaler.update()
        self.optim.zero_grad()
        self.n_valid_batch_samples = 0
        self.batch_idx += 1

    def process_sample(self,
                       sample: ModelInput,
                       eval_ty: EvalTy,
                       compute_zero_wt_loss: bool = True
                       ) -> Tuple[Optional[ModelOutput], Optional[ModelLoss]]:
        """Run model on a single sample"""
        model_out = self.safe_eval(sample, eval_ty=eval_ty)
        model_loss = self.get_model_loss(
            model_out=model_out,
            eval_ty=eval_ty,
            compute_zero_wt_loss=compute_zero_wt_loss,
        ) if exists(model_out) else None

        if eval_ty == EvalTy.TRAINING:
            self.accum_grad(model_loss)

        return model_out, model_loss

    """Model evaluation methods"""

    def safe_eval(self, sample: ModelInput, eval_ty: EvalTy, raise_exceptions: Optional[bool] = None) -> ModelOutput:
        """Evaluate model and (optionally) catch exceptions"""
        with torch.set_grad_enabled(eval_ty == eval_ty.TRAINING):
            try:
                return self.model(sample.to(self.config.device).crop(self.config.max_len))
            except Exception as e:
                logger.error(f'Caught exception {e} evaluating model,'
                             f' eval_ty :{eval_ty.value}, sample_idx : {self.n_processed}')
                if default(raise_exceptions, self.config.raise_exceptions):
                    raise e

    def get_model_loss(self,
                       model_out: ModelOutput,
                       eval_ty: EvalTy,
                       compute_zero_wt_loss: bool = True
                       ) -> Optional[ModelLoss]:
        """Compute model loss"""
        logger.info("Getting Model Loss")
        with torch.set_grad_enabled(eval_ty == eval_ty.TRAINING):
            try:
                return self.model.compute_loss(
                    output=model_out,
                    compute_zero_wt_loss=compute_zero_wt_loss,
                )
            except Exception as e:
                logger.error(f"Error {e} calculating model loss, eval_ty : {eval_ty.value}")
                if self.config.raise_exceptions:
                    raise e
                return None

    """Optimization Specific Functions"""

    def accum_grad(self, model_loss: Optional[ModelLoss]) -> None:
        """Accumulate gradient"""
        logger.info("Accumulating Model Gradients")
        if model_loss is None:
            return
        loss = model_loss.get_loss()
        config = self.config
        # check for nan in gradient and skip if found
        if check_nan_in_grad(self.model) or torch.any(torch.isnan(loss)):
            logger.warn("found NaN in gradients. skipping update")
            self.model.zero_grad()
            return

        # self.scaler used for mixed precision training
        if not config.clip_grad_per_sample:
            self.scaler.scale(loss / config.batch_size).backward()
            return
        else:
            self.scaler.scale(loss).backward()

        self.n_valid_batch_samples += 1
        if self.clipped_grad is None:
            self.clipped_grad = {name: 0 for name, param in self.model.named_parameters()
                                 if (param is not None and param.grad is not None)}
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.grad_norm_clip)
        grad_params = filter(lambda x: x[1] is not None and x[1].grad is not None,
                             self.model.named_parameters())
        for name, param in grad_params:
            self.clipped_grad[name] += param.grad
        self.model.zero_grad()

    """I/O Methods"""

    def checkpoint_model(self):
        logger.info("Checkpointing Model State")
        """hard-checkpoint of model state - in case things go awry"""
        self.save_state(path=self.config.checkpoint_path(self.checkpoint))
        self.checkpoint += 1

    def save_state(self, path=None) -> None:
        """Save model and training state"""
        logger.info("Saving Model State")
        torch.save(
            {
                'model': self.model.state_dict(),
                'optim': self.optim.state_dict(),
                'epoch_scheduler': self.epoch_scheduler.state_dict(),
                'epoch': self.epoch,
                'processed': self.n_processed,
                'sample_idx': self.sample_idx,
                'checkpoint': self.checkpoint,
                'batch_idx': self.batch_idx,
            },
            default(path, self.config.save_path)
        )

    def load_state(self, path: Optional[str] = None):
        """Load from previous state"""
        config, path = self.config, default(path, self.config.model_load_path)
        print(f"[INFO] LOADING MODEL \nCheckpoint = {config.load_from_checkpoint}")
        logger.info(f"LOADING MODEL, Checkpoint = {config.load_from_checkpoint}")
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f'[ERROR] Failed to load model! {e}\n path : {path}')
            raise e

        self.model.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
        self.epoch = checkpoint['epoch']
        self.n_processed = checkpoint['processed']
        self.sample_idx = checkpoint['sample_idx']
        self.checkpoint = checkpoint['checkpoint']
        self.batch_idx = checkpoint['batch_idx']

    """Private Helper Methods"""

    def _display_loss(self, loss: ModelLoss, eval_ty: EvalTy):
        """Display model loss"""
        if not (exists(loss)):
            return
        print(f"###### BATCH_IDX {self.batch_idx}, EPOCH {self.epoch},"
              f" TY {eval_ty.value}, PROCESSED {self.n_processed} ######")
        loss.display_loss()

    def _process_dataset(
            self,
            eval_ty: EvalTy = EvalTy.VALIDATION,
            dataset: ProteinDatasetABC = None,
            max_samples: int = -1,
    ):
        """
        Run model on all samples in a dataset - does not compute gradients.
        """
        logger.info(f"Processing {eval_ty.value} dataset")
        dataset = default(dataset, self.datasets[eval_ty.value])
        data = dataset.get_dataloader(
            batch_size=min(len(dataset), self.config.batch_size),
            num_workers=self.config.data_workers,
            shuffle=False
        )
        n_processed, start = 0, time.time()

        with torch.no_grad():
            for idx, batch_data in enumerate(data):
                if n_processed > max_samples > 0:
                    break
                for sample in batch_data:
                    model_out, model_loss = self.process_sample(sample, eval_ty=eval_ty)
                    if model_loss is None:
                        continue
                    n_processed += 1
                    self._display_loss(model_loss, eval_ty=eval_ty)
        avg_time = (time.time() - start) / max(n_processed, 1)
        logger.info(f"[INFO] finished running on {eval_ty} set, average time / sample :", {avg_time})

    def _get_optim(self, config: ModelConfig):
        """Get optimizer"""
        logger.info("initializing optimizer")
        if config.weight_decay <= 0:
            return Adam(self.model.parameters(), lr=config.lr)
        else:
            return AdamW(
                self.model.parameters(),
                lr=config.weight_decay,
                weight_decay=config.weight_decay
            )

    def _init_model(self):
        """Initialize model with dummy forward pass"""
        print("[INFO] initializing model...")
        # make sure all modules in model are initialized via a dummy forward pass
        max_tries = 3
        dataset = self.datasets[EvalTy.TRAINING.value]
        for i in range(max_tries + 2):
            print(f"[INFO] Running Dummy forward pass {i}")
            try:
                out = self.safe_eval(sample=dataset[i], eval_ty=EvalTy.TESTING, raise_exceptions=True)
            except Exception as e:  # noqa
                out = None
            if i > max_tries:
                # probably an error, so raise the exception
                print("[ERROR] unable to initialize model")
                assert out is None
                # raise the exception
                self.safe_eval(sample=dataset[i], eval_ty=EvalTy.TESTING, raise_exceptions=True)
            if out is None:
                continue
            else:
                break
        print("[INFO] Successfully initialized model...")
