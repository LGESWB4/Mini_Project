import torch
import numpy as np
from abc import abstractmethod

class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, optimizer, save_dir, args):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.save_dir = save_dir

        # Early stopping 설정
        self.early_stop = args.early_stop

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        patience = 0
        best_val_F1 = 0
        best_val_loss = np.inf

        print("Training Start")
        for epoch in range(1, self.epochs + 1):
            result = self._train_epoch(epoch)

            # Save logged information
            log = {"epoch": epoch}
            log.update(result)

            print("--- Log print ---")
            for key, value in log.items():
                print(f"{str(key)}: {value:.5f}" if key != "epoch" else f"{str(key)}: {value}")

            # 모델 성능 평가 후 저장
            if log["val_F1"] > best_val_F1:
                print(f"New best model for val F1: {log['val_F1']:.4f}! Saving the best model..")
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_epoch.pth")
                best_val_F1 = log["val_F1"]
                best_val_loss = log["val_Loss"]
                patience = 0
            else:
                patience += 1

            torch.save(self.model.state_dict(), f"{self.save_dir}/latest.pth")
            print(
                f"[Val] F1: {log['val_F1']:.4f}, Loss: {log['val_Loss']:.4f} || "
                f"Best F1: {best_val_F1:.4f}, Best Loss: {best_val_loss:.4f} || Patience: {patience}\n"
            )

            # Early stopping
            if self.early_stop < patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best Performance - F1: {best_val_F1:.4f}, Loss: {best_val_loss:.4f}")
                break
