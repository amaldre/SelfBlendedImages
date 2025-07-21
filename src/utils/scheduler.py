import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler
import math
class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch=self.n_epoch
        b_lr=self.base_lrs[0]
        start_decay=self.start_decay
        if last_epoch>start_decay:
            lr=b_lr-b_lr/(n_epoch-start_decay)*(last_epoch-start_decay)
        else:
            lr=b_lr
        return [lr]


class LinearDecayLR_LaaNet(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1, booster=2):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        self.booster = booster
        super(LinearDecayLR_LaaNet, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch = self.n_epoch
        b_lr = self.base_lrs[-1]

        if last_epoch > 0:
            try:
                cur_lr = self.get_last_lr()
            except:
                cur_lr = b_lr * self.booster
        start_decay = self.start_decay

        if last_epoch >= start_decay:
            lr = b_lr * self.booster - (b_lr * self.booster)/(n_epoch - start_decay) * (last_epoch - start_decay)
        else:
            if last_epoch < start_decay:
                lr = b_lr + (b_lr * self.booster - b_lr)/start_decay * last_epoch
            else:
                lr = cur_lr
                
        self._last_lr = lr
        print(f'Active Learning Rate --- {lr}')
        return [lr]

    import math
from torch.optim.lr_scheduler import _LRScheduler

class FlatCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, flat_epochs, eta_min=0.0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_epochs (int): Total number of training epochs.
            flat_epochs (int): Number of initial epochs with constant LR.
            eta_min (float): Minimum learning rate.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.total_epochs = total_epochs
        self.flat_epochs = flat_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.flat_epochs:
            return [base_lr for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.flat_epochs) / max(1, self.total_epochs - self.flat_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]


if __name__=='__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, 0.001)
    s=LinearDecayLR(optimizer, 100, 75)
    ss=[]
    for epoch in range(100):
        optimizer.step()
        s.step()
        ss.append(s.get_lr()[0])

    print(ss)