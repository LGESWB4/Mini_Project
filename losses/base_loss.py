import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.5, 0.25, 0.5], gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (list or float): 클래스별 가중치 (예: [0.25, 0.5, 0.25])
            gamma (float): Hard Example 가중치 (일반적으로 2.0 사용)
            reduction (str): "mean", "sum", "none" 중 선택
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = torch.tensor([0.25, 0.25, 0.25])
        elif isinstance(alpha, (list, tuple)):  
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = torch.tensor([alpha] * 3)  

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  

        p_t = torch.softmax(inputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = torch.clamp(p_t, min=1e-8, max=1.0)
        alpha_t = self.alpha.to(inputs.device)[targets]

        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_labels * log_probs).sum(dim=1)
        return loss.mean()

_loss_entrypoints = {
    "cross_entropy": CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingCrossEntropy,
}

def get_loss_fn(loss_name, **kwargs):
    if loss_name in _loss_entrypoints:
        return _loss_entrypoints[loss_name](**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# class ClassificationMetrics:
#     def __init__(self, num_classes=3):
#         self.num_classes = num_classes

#     def compute(self, y_pred, y_true):
#         """
#         y_pred: (batch_size, num_classes) → 확률값 (Softmax 적용 전)
#         y_true: (batch_size,) → 정답 클래스 인덱스
#         """
#         y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
#         y_true = y_true.cpu().numpy()

#         acc = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
#         recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
#         f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
#         cm = confusion_matrix(y_true, y_pred)

#         return {
#             "accuracy": acc,
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1,
#             "confusion_matrix": cm
#         }


if __name__ == "__main__":
    batch_size = 8
    num_classes = 3
    y_true = torch.randint(0, num_classes, (batch_size,))  # 0~2 class
    y_pred = torch.randn(batch_size, num_classes)  # logit

    loss_fn = get_loss_fn("cross_entropy") 
    loss = loss_fn(y_pred, y_true)
    print(f"Loss ({loss_fn.__class__.__name__}): {loss.item():.4f}")

    # metrics = ClassificationMetrics(num_classes=3)
    # results = metrics.compute(y_pred, y_true)

    # print("\n🔹 Classification Metrics 🔹")
    # for key, value in results.items():
    #     if key == "confusion_matrix":
    #         print(f"{key}:\n{value}")
    #     else:
    #         print(f"{key}: {value:.4f}")
