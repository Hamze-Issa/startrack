from torchmetrics import JaccardIndex, Accuracy, R2Score, F1Score
from torchmetrics.regression import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy

class R2Wrapper:
    def __init__(self, metric):
        self.metric = metric
    def __call__(self, preds, target):
        # Flatten from [B, C, H, W] to [B*H*W, C] or [N] if C=1
        if preds.dim() == 4 and target.dim() == 4:
            B, C, H, W = preds.shape
            preds = preds.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.permute(0, 2, 3, 1).reshape(-1, C)
            if C == 1:
                preds = preds.view(-1)
                target = target.view(-1)
        return self.metric(preds, target)

def custom_r2score(**kwargs):
    r2 = R2Score(**kwargs)
    return R2Wrapper(r2)

METRIC_FUNCTIONS = {
    'iou': lambda **kwargs: JaccardIndex(**kwargs),
    'accuracy': lambda **kwargs: Accuracy(**kwargs),
    'multiclass_acc': lambda **kwargs: MulticlassAccuracy(**kwargs),
    'r2_metric' : custom_r2score, #lambda **kwargs: R2Score(**kwargs),
    'mse' : lambda **kwargs: MeanSquaredError(**kwargs),
    'f1' : lambda **kwargs: F1Score(**kwargs),
    # Add other torchmetrics or custom metrics here
}
