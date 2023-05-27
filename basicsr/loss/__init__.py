import os
from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
from basicsr.loss.hard_example_mining import HEM
import matplotlib

# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

class Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        device = torch.device('cuda')

        # self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        loss_type = "1*L1+2*HEM"
        for loss in loss_type.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'HEM':
                loss_function = HEM(device=device)
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')()
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found'.format(loss_type))

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        self.loss_module.to(device)


    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # self.log[-1, i] += effective_loss.item()
            #elif l['type'] == 'DIS':
            #    self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        # if len(self.loss) > 1:
        #     self.log[-1, -1] += loss_sum.item()
        return loss_sum
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class Loss2(nn.modules.loss._Loss):
    def __init__(self, loss_type = "1*L1+2*HEM"):
        super(Loss2, self).__init__()
        print('Preparing loss function:')

        device = torch.device('cuda')

        # self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        # loss_type = "1*L1+2*HEM"
        for loss in loss_type.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == "PSNR":
                loss_function = PSNRLoss()
            elif loss_type == 'HEM':
                loss_function = HEM(device=device)
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')()
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found'.format(loss_type))

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        self.loss_module.to(device)


    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # self.log[-1, i] += effective_loss.item()
            #elif l['type'] == 'DIS':
            #    self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        # if len(self.loss) > 1:
        #     self.log[-1, -1] += loss_sum.item()
        return loss_sum