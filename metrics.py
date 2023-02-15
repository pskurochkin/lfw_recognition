import torch
import torch.nn.functional as F


def triplet_accuracy(anchor, positive, negative):
    pos_distance = F.pairwise_distance(anchor, positive)
    neg_distance = F.pairwise_distance(anchor, negative)

    return torch.mean((pos_distance < neg_distance).to(torch.float32))
