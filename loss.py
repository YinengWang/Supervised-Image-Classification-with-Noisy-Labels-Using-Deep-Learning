import torch.nn.functional as F
import torch
import torch.nn as nn


class ELR_loss(nn.Module):
    def __init__(self, n_examples, n_classes=10, beta=0.3, lam=3):
        super(ELR_loss, self).__init__()
        self.beta = beta  # "beta": 0.7 in authors' cifar10 config file
        self.lam = lam  # "lambda": 3 in authors' cifar10 config file
        self.n_classes = n_classes

        # target probabilities for each example
        if torch.cuda.is_available():
            self.stored_targets = torch.zeros(n_examples, self.n_classes).cuda()
        else:
            self.stored_targets = torch.zeros(n_examples, self.n_classes)

    def forward(self, indices, output, label):
        pred = F.softmax(output)
        pred = torch.clamp(pred, 1e-4, 1.0-1e-4)

        ce_loss = F.cross_entropy(output, label)

        # detach to update running average, need to detach so torch does not try to compute gradients for this
        pred_detached = pred.data.detach()
        # update running average of target probabilities
        self.stored_targets[indices, :] = self.beta * self.stored_targets[indices, :] + (1 - self.beta) * (
                pred_detached / pred_detached.sum(dim=1, keepdim=True))

        # ELR regularization - maximizes dot product between model output and stored targets to penalize memorization.
        elr_regularization_term = ((1 - (self.stored_targets[indices] * pred).sum(dim=1)).log()).mean()

        loss_tot = ce_loss + self.lam * elr_regularization_term

        return loss_tot
