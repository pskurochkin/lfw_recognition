import torch
from torch.utils.tensorboard import SummaryWriter

from metrics import triplet_accuracy


class Trainer:
    def __init__(self, loss_fn, optimizer, logdir=None):
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.writer = SummaryWriter(logdir)

    def train_step(self, embedder, anchor_img, positive_img, negative_img):
        anchor = embedder(anchor_img)
        positive = embedder(positive_img)
        negative = embedder(negative_img)

        loss = self.loss_fn(anchor, positive, negative)
        accuracy = triplet_accuracy(anchor, positive, negative)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item(), accuracy

    def eval_step(self, embedder, anchor_img, positive_img, negative_img):
        with torch.no_grad():
            anchor = embedder(anchor_img)
            positive = embedder(positive_img)
            negative = embedder(negative_img)

        loss = self.loss_fn(anchor, positive, negative)
        accuracy = triplet_accuracy(anchor, positive, negative)

        return loss.item(), accuracy
