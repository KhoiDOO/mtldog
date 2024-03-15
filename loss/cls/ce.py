import torch.nn.functional as F

def loss_cls_ce(logits, labels):
    return F.cross_entropy(logits, labels)