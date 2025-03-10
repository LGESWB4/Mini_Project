import cv2
import time
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score

from trainer.base_trainer import BaseTrainer
from utils.util import MetricTracker, inf_loop

class Trainer(BaseTrainer):
    """
    Trainer class for 3-class classification
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        save_dir,
        args,
        device,
        train_loader,
        val_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, optimizer, save_dir, args)
        self.args = args
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.do_validation = self.val_loader is not None
        self.lr_scheduler = lr_scheduler
        
        # Determine epoch length
        if len_epoch is None:
            self.len_epoch = len(self.train_loader)
        else:
            self.train_loader = inf_loop(train_loader)
            self.len_epoch = len_epoch

        # Metric Tracking
        self.train_metrics = MetricTracker("Loss", *[c.__class__.__name__ for c in self.criterion], *["Accuracy", "F1"])
        self.valid_metrics = MetricTracker("Loss", *["Accuracy", "F1"])
        print(self.train_metrics._data.index) #['loss', 'CrossEntropyLoss', 'Accuracy', 'F1']
        self.scaler = GradScaler()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        start = time.time()
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # # save train images
            # dt = data[0].numpy()
            # dt = dt.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            # dt = (dt * 255).clip(0, 255).astype(np.uint8)  # Ensuring valid range
            # dt = cv2.cvtColor(dt, cv2.COLOR_RGB2BGR)
            # print(f"Saving image {batch_idx}: shape={dt.shape}, max={dt.max()}, min={dt.min()}")
            # cv2.imwrite(f"./example/ex_{batch_idx}.png", dt)
            
            data, target = data.to(self.device), target.to(self.device)
            total_loss = 0

            self.optimizer.zero_grad()
            with autocast(device_type=self.device):
                output = self.model(data)
                for loss_fn in self.criterion:  # [bce_with_logit, ...]
                    loss = loss_fn(output, target)
                    # print(f"{loss_fn} : ", loss)
                    self.train_metrics.update(loss_fn.__class__.__name__, loss.item())  # metric_fn마다 값 update
                    total_loss += loss
                
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Compute predictions
            preds = torch.argmax(output, dim=1).cpu().numpy()
            target_np = target.cpu().numpy()

            # Compute Accuracy & F1-score
            acc = accuracy_score(target_np, preds)
            macro_f1 = f1_score(target_np, preds, average="macro")
            self.train_metrics.update("Accuracy", acc)
            self.train_metrics.update("F1", macro_f1)
            self.train_metrics.update("Loss", total_loss.item()) # Loss per batch

            if batch_idx % self.args.log_interval == 0:
                print(f"Train Epoch: {epoch}/{self.args.epochs} {self._progress(batch_idx)} Loss: {total_loss.item():.6f}")
                log_dict = self.train_metrics.result()
                if self.args.is_wandb:
                    wandb.log({"Iter_train_" + k: v for k, v in log_dict.items()})

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        # print(log)
        if self.args.is_wandb:
            wandb.log({"Epoch_Train_Loss": log["Loss"], "Epoch_Train_Accuracy": log["Accuracy"], "Epoch_Train_F1": log["F1"]})

        print(f"Train Epoch: {epoch}, Loss: {log['Loss']:.4f}, Accuracy: {log['Accuracy']*100:.1f}, F1: {log['F1']:.4f}")
        print(f"Train time per epoch: {time.time()-start:.3f}s\n")

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})
            if self.args.is_wandb:
                wandb.log({"Epoch_val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if self.args.is_wandb:
                wandb.log({"lr": self.optimizer.param_groups[0]["lr"]})
            if self.args.lr_scheduler["type"] == "ReduceLROnPlateau":
                self.lr_scheduler.step(log["val_Loss"])
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        """
        print("Validation Start!")
        self.model.eval()
        self.valid_metrics.reset()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):        
                total_val_loss = 0
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                for loss_fn in self.criterion:  # [bce_with_logit, ...]
                        total_val_loss += loss_fn(output, target)
                        
                total_val_loss += total_val_loss.item()

                # Compute predictions
                preds = torch.argmax(output, dim=1).cpu().numpy()
                target_np = target.cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(target_np)

                # Compute Accuracy & F1-score
                acc = accuracy_score(all_targets, all_preds)
                macro_f1 = f1_score(all_targets, all_preds, average="macro")

                # Update validation metrics
                self.valid_metrics.update("Loss", total_val_loss)
                self.valid_metrics.update("Accuracy", acc)
                self.valid_metrics.update("F1", macro_f1)

        val_log_dict = self.valid_metrics.result()
        print(f"Validation Epoch: {epoch}, Loss: {val_log_dict['Loss']:.4f}, Accuracy: {val_log_dict['Accuracy']*100:.1f}, F1: {val_log_dict['F1']:.4f}\n")

        return val_log_dict

    def _progress(self, batch_idx):
        base = "[{:>3d}/{}]"
        if hasattr(self.train_loader, "n_samples"):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total)
