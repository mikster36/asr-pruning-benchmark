import os

import torch
from torchaudio.models import Wav2Letter

from pruning.sensitivity import prune_sensitivity
from pruning.gradient import prune_gradient
from labels import labels
from train import device, test_loader, criterion, train, n_iter, lm
from logger import Logger

model = Wav2Letter(num_classes=len(labels)).to(device)
model.load_state_dict(torch.load('/Users/michael/PycharmProjects/deep-learning-playground/states_fused.pth'))

prune_ratio = 0.5
prune_method = 'gradient'
logger = Logger('wav2letter_log.txt')

print(f"Pruning LibriSpeech with {prune_ratio*100}% using method {prune_method}...")
for name, module in model.named_modules():
    print(f"{name}: {module}")

if prune_method == 'sensitivity':
    model = prune_sensitivity(model, test_loader, criterion=criterion, prune_ratio=prune_ratio,
                              method='filter', evaluation="batch", batch_size=1, logger=logger, lm=lm)

    masks = {name: module.weight_mask for name, module in model.named_modules() if hasattr(module, 'weight_mask')}
    torch.save(masks, 'sensitivity_masks.pth')

    print(f"Retraining LibriSpeech after pruning...")
    train(model, n_iter, lm)
elif prune_method == 'gradient':
    path = '/Users/michael/PycharmProjects/deep-learning-playground/gradient_masks.pt'
    if os.path.exists(path):
        mask = torch.load(path)
        print(mask)
        print('Applying loaded mask')
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param.data *= mask
                    param.requires_grad = False
    else:
        model = prune_gradient(model=model, data_loader=test_loader, criterion=criterion, prune_ratio=prune_ratio,
                               device=device, logger=logger)

    print(f"Retraining LibriSpeech after pruning...")
    train(model, n_iter, lm)
