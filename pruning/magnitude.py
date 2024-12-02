import os

import torch
import sys
from tqdm import tqdm

from logger import Logger


def prune_magnitude(
    model: torch.nn.Module, prune_ratios=None, logger=None
):
    if prune_ratios is None:
        prune_ratios = [0.3]
    if logger:
        sys.stdout = logger
    else:
        sys.stdout = Logger('magnitude_pruning_log.txt')

    model.eval()  # Set the model to evaluation mode to avoid updating BatchNorm stats
    total = 0
    weights = []
    masks = {}

    for i in range(len(prune_ratios) - 1, 0, -1):
        prune_ratio = prune_ratios[i]
        mask_path = f'masks/magnitude/{prune_ratio}_masks.pth'
        if os.path.exists(mask_path):
            mask = torch.load(mask_path)
            print(f'Loaded masks from {mask_path}')
            masks[str(prune_ratio)] = mask
            prune_ratios.remove(prune_ratio)

    if len(prune_ratios) == 0:
        return model, masks

    # Collect all weights and their magnitudes
    with torch.no_grad():
        for name, param in tqdm(list(model.named_parameters()), 'Collecting weights', file=sys.stdout):
            if param.requires_grad and param.data is not None:
                total += param.numel()
                weight_magnitude = param.data.abs()
                weights.append((weight_magnitude, name))

    # Concatenate all magnitudes for sorting
    all_weights = torch.cat([w.flatten() for w, _ in weights])
    for prune_ratio in prune_ratios:
        num_to_prune = int(prune_ratio * total)

        # Get the threshold value for pruning
        threshold, _ = torch.topk(all_weights, num_to_prune, largest=False)

        # Apply pruning by creating masks for each parameter
        mask = {}
        with torch.no_grad():
            for _, name, idx in tqdm(to_prune, 'Pruning weights', file=sys.stdout):
                param = dict(model.named_parameters())[name]
                param.view(-1)[idx] = 0.0
                # Do not set requires_grad to False; keep it True for future gradient updates

                # Optionally create a mask
                module = dict(model.named_modules()).get(name, None)
                if module and not hasattr(module, 'weight_mask'):
                    module.weight_mask = torch.ones_like(param)
                if module:
                    module.weight_mask.view(-1)[idx] = 0.0

                if name not in mask:
                    mask[name] = torch.ones_like(param)
                mask[name].view(-1)[idx] = 0.0

        if not os.path.exists('./masks'):
            os.mkdir('./masks/')
        if not os.path.exists('./masks/magnitude'):
            os.mkdir('./masks/magnitude')
        torch.save(masks, f'./masks/magnitude/{prune_ratio}_wav2letter_masks.pth')

        print(f"Pruned {num_to_prune} weights out of {total} weights total")
        masks[str(prune_ratio)] = mask

    return model, masks
