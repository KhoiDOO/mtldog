from torchmetrics import Accuracy
from torch import Tensor    

def metric_cls_acc(preds: Tensor, label: Tensor)->float:
    
    if len(preds.size()) > 1:
        B, C = preds.size()
        accuracy = Accuracy(task="multiclass", num_classes=C).to(preds.device)
    else:
        accuracy = Accuracy(task="binary").to(preds.device)
    
    acc: Tensor = accuracy(preds.argmax(1), label)
    
    return acc.item()