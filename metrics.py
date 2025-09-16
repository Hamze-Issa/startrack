from torchmetrics import JaccardIndex, Accuracy
from torchmetrics.classification import MulticlassAccuracy

METRIC_FUNCTIONS = {
    'iou': lambda **kwargs: JaccardIndex(**kwargs),
    'accuracy': lambda **kwargs: Accuracy(**kwargs),
    'multiclass_acc': lambda **kwargs: MulticlassAccuracy(**kwargs),
    # Add other torchmetrics or custom metrics here
}
