import torch
import sys
from tqdm import tqdm

from logger import Logger


def prune_gradient(
    model: torch.nn.Module, data_loader, device, logger, criterion, prune_ratio=0.3
):
    if logger:
        sys.stdout = logger
    else:
        sys.stdout = Logger('gradient_pruning_log.txt')
    model.train()  # Set the model to training mode
    total = 0
    gradients = []

    # Get a batch of data
    inputs, labels, input_lengths, target_lengths = next(iter(data_loader))
    inputs = inputs.to(device)

    # Generate output from model
    outputs = model(inputs)

    model.zero_grad()
    loss = criterion(
        outputs.permute(2, 0, 1),
        labels,
        input_lengths,
        target_lengths,
    )
    loss.backward()

    # Collect gradients
    with torch.no_grad():
        for name, param in tqdm(list(model.named_parameters()), 'Collecting gradients', file=sys.stdout):
            if param.requires_grad and param.grad is not None:
                total += param.numel()
                gradient_magnitude = param.grad.abs()
                # Flatten gradients and collect with their indices
                for idx in range(param.numel()):
                    gradients.append(
                        (gradient_magnitude.view(-1)[idx].item(), name, idx)
                    )

    # Sort gradients and determine pruning threshold
    gradients.sort()
    num_to_prune = int(prune_ratio * len(gradients))
    to_prune = gradients[:num_to_prune]

    masks = {}
    # Apply pruning based on gradient magnitude
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

            if name not in masks:
                masks[name] = torch.ones_like(param)
            masks[name].view(-1)[idx] = 0.0

    torch.save(masks, './masks/gradient_masks.pth')

    print(f"Pruned {num_to_prune} weights out of {total} weights total")

    return model, masks
