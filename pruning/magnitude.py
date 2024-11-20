import torch
import sys
from tqdm import tqdm

from logger import Logger


def prune_magnitude(
    model: torch.nn.Module, prune_ratio=0.3, logger=None
):
    if logger:
        sys.stdout = logger
    else:
        sys.stdout = Logger('magnitude_pruning_log.txt')

    model.eval()  # Set the model to evaluation mode to avoid updating BatchNorm stats
    total = 0
    weights = []

    # Collect all weights and their magnitudes
    with torch.no_grad():
        for name, param in tqdm(list(model.named_parameters()), 'Collecting weights', file=sys.stdout):
            if param.requires_grad and param.data is not None:
                total += param.numel()
                weight_magnitude = param.data.abs()
                weights.append((weight_magnitude, name))

    # Concatenate all magnitudes for sorting
    all_weights = torch.cat([w.flatten() for w, _ in weights])
    num_to_prune = int(prune_ratio * total)

    # Get the threshold value for pruning
    threshold, _ = torch.topk(all_weights, num_to_prune, largest=False)

    # Apply pruning by creating masks for each parameter
    masks = {}
    with torch.no_grad():
        for weight_magnitude, name in tqdm(weights, 'Pruning weights', file=sys.stdout):
            mask = (weight_magnitude >= threshold[-1]).float()
            param = dict(model.named_parameters())[name]
            param.data.mul_(mask)

            masks[name] = mask

    # Save the masks for later use
    torch.save(masks, './masks/magnitude_masks.pth')

    print(f"Pruned {num_to_prune} weights out of {total} weights total")

    return model, masks
