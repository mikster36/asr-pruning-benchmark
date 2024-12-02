import torch
import sys
from tqdm import tqdm

from logger import Logger

torch.random.manual_seed(43)


def prune_random(model: torch.nn.Module, prune_ratio=0.3, logger=None):
    if logger:
        sys.stdout = logger
    else:
        sys.stdout = Logger('sensitivity_pruning_log.txt')
    model.eval()  # Set the model to evaluation mode to avoid updating BatchNorm stats
    total = 0
    weights = []

    # Collect all weights
    with torch.no_grad():
        for name, param in tqdm(list(model.named_parameters()), 'Collecting weights', file=sys.stdout):
            if param.requires_grad and param.data is not None:
                total += param.numel()
                weights.append((param, name))

    # Determine the total number of weights to prune
    num_to_prune = int(prune_ratio * total)

    # Generate a random selection of indices to prune
    prune_indices = torch.randperm(total, device=next(model.parameters()).device)[:num_to_prune]

    # Create masks for each parameter
    masks = {name: torch.ones_like(param, device=param.device) for param, name in weights}

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
            masks[name].view(-1)[current_indices] = 0.0

            offset += num_elements

    # Save the masks for later use
    torch.save(masks, './masks/random_masks.pth')

    print(f"Randomly pruned {num_to_prune} weights out of {total} weights total")

    return model, masks