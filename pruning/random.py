import os
import torch
import sys
from tqdm import tqdm

from logger import Logger

torch.random.manual_seed(43)


def prune_random(model: torch.nn.Module, prune_ratios=None, logger=None):
    if prune_ratios is None:
        prune_ratios = [0.3]
    if logger:
        sys.stdout = logger
    else:
        sys.stdout = Logger('random_pruning_log.txt')
    model.eval()  # Set the model to evaluation mode to avoid updating BatchNorm stats
    total = 0
    weights = []
    masks = {}
    for i in range(len(prune_ratios) - 1, 0, -1):
        prune_ratio = prune_ratios[i]
        mask_path = f'masks/gradient/{prune_ratio}_masks.pth'
        if os.path.exists(mask_path):
            mask = torch.load(mask_path)
            print(f'Loaded masks from {mask_path}')
            masks[str(prune_ratio)] = mask
            prune_ratios.remove(prune_ratio)
    if len(prune_ratios) == 0:
        return model, masks

    # Collect all weights
    with torch.no_grad():
        for name, param in tqdm(list(model.named_parameters()), 'Collecting weights', file=sys.stdout):
            if param.requires_grad and param.data is not None:
                total += param.numel()
                weights.append((param, name))

    for prune_ratio in prune_ratios:
        # Determine the total number of weights to prune
        num_to_prune = int(prune_ratio * total)

        # Generate a random selection of indices to prune
        prune_indices = torch.randperm(total, device=next(model.parameters()).device)[:num_to_prune]

        # Create masks for each parameter
        mask = {name: torch.ones_like(param, device=param.device) for param, name in weights}

        # Apply pruning manually across all parameters
        with torch.no_grad():
            offset = 0
            for param, name in tqdm(weights, 'Pruning weights', file=sys.stdout):
                num_elements = param.numel()
                current_indices = prune_indices[
                    (prune_indices >= offset) & (prune_indices < offset + num_elements)
                ] - offset

                # Zero out the selected weights in the parameter
                param.view(-1)[current_indices] = 0.0
                mask[name].view(-1)[current_indices] = 0.0

                offset += num_elements

        if not os.path.exists('./masks'):
            os.mkdir('./masks/')
        if not os.path.exists('./masks/random'):
            os.mkdir('./masks/random')
        torch.save(mask, f'./masks/random/{prune_ratio}_wav2letter_masks.pth')

        print(f"Randomly pruned {num_to_prune} weights out of {total} weights total")
        masks[str(prune_ratio)] = mask

    return model, masks