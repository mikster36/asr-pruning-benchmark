import sys
import os

import torch
import torchaudio
from tqdm import tqdm

from logger import Logger


@torch.inference_mode()
def prune_test(model, test_loader, criterion, lm, batch_size, evaluation):

    count = 0
    test_loss_sum = 0
    c_ldist_sum, c_ref_len_sum = 0, 0
    w_ldist_sum, w_ref_len_sum = 0, 0

    for batch in test_loader:
        waves, labels, input_lens, output_lens = batch
        waves, labels = waves.cuda(
            non_blocking=True), labels.cuda(non_blocking=True)

        out = model(waves)  # (batch, n_class, time)

        loss = criterion(out.permute(2, 0, 1), labels, input_lens, output_lens)
        test_loss_sum += loss.item()

        decoded_preds = lm.decode_ctc(out)
        decoded_targets = lm.decode_ids(labels)
        decoded_targets = [t[:len]
                           for t, len in zip(decoded_targets, output_lens)]

        for hypo, ref in zip(decoded_preds, decoded_targets):
            c_ldist_sum += torchaudio.functional.edit_distance(ref, hypo)
            c_ref_len_sum += len(ref)

            hypo_words = ''.join(hypo).split()
            ref_words = ''.join(ref).split()
            w_ldist_sum += torchaudio.functional.edit_distance(ref_words, hypo_words)
            w_ref_len_sum += len(ref_words)

        count += 1
        if evaluation == 'batch' and count >= batch_size:
            break

    test_loss = test_loss_sum / len(test_loader)
    cer = c_ldist_sum / c_ref_len_sum
    wer = w_ldist_sum / w_ref_len_sum

    return test_loss, cer, wer


def prune_sensitivity(model, data_loader, criterion, prune_ratios=None,
                      method='filter', evaluation='full', batch_size=1, logger=None, lm=None):
    """
    Prune weights based on sensitivity criterion. We observe the change in loss induced
    by pruning a weight or weight group and use this as the criterion to prune.

    Args:
        model: PyTorch neural network model.
        data_loader: Data loader for computing gradients.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        prune_ratio: Fraction of weights to prune.
        method: Method for sensitivity calculation: 'filter', 'gradient', 'weight'. Filter is pruning by an entire filter
        layer. Gradient is an approximate of the sensitivity (equivalent to the first time of the Taylor expansion for
        loss sensitivity). Weight uses individual weights.
        evaluation: Evaluation type: 'mini_batch' or 'full'.
        batch_size: Number of batches to use for mini-batch methods.
        logger: Logger

    Returns:
        model: Pruned model.
    """
    if prune_ratios is None:
        prune_ratios = [0.2]
    if logger:
        sys.stdout = logger
    else:
        sys.stdout = Logger('sensitivity_pruning_log.txt')
    try:
        model.eval()
        sensitivities = []
        total_weights = 0
        device = next(model.parameters()).device
        masks = {}

        for i in range(len(prune_ratios) - 1, 0, -1):
            prune_ratio = prune_ratios[i]
            mask_path = f'masks/sensitivity/{prune_ratio}_masks.pth'
            if os.path.exists(mask_path):
                mask = torch.load(mask_path)
                print(f'Loaded masks from {mask_path}')
                masks[str(prune_ratio)] = mask
                prune_ratios.remove(prune_ratio)

        if len(prune_ratios) == 0:
            return model, masks

        if method == 'gradient':
            inputs, targets = next(iter(data_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            with torch.no_grad():
                for name, param in tqdm(list(model.named_parameters()), 'Approximating sensitivity using gradient',
                                        file=sys.stdout):
                    total_weights += param.numel()
                    if param.requires_grad:
                        sensitivity = (param.grad * param).abs()  # |∂L/∂w_j * w_j|
                        sensitivities.extend([(sensitivity.view(-1)[i].item(), name, i)
                                              for i in range(param.numel())])

        else:

            baseline_loss, _, _ = prune_test(model, data_loader, criterion, lm, batch_size, evaluation)

            with torch.no_grad():
                for name, param in tqdm(list(model.named_parameters()), "Calculating sensitivities", file=sys.stdout):
                    if not param.requires_grad:
                        continue

                    if method == 'filter' and len(param.shape) == 4:
                        total_weights += param.numel()
                        for i in range(param.shape[0]):
                            original_filter = param[i].clone()
                            param[i] = 0.0
                            pruned_loss, _, _ = prune_test(model, data_loader, criterion, lm, batch_size, evaluation)
                            sensitivity = abs(pruned_loss - baseline_loss)
                            sensitivities.append(
                                (sensitivity, name, i, param[i].numel()))
                            param[i] = original_filter

                    else:
                        param_flat = param.view(-1)
                        total_weights += param_flat.numel()

                        for i in tqdm(range(param_flat.numel())):
                            original_weight = param_flat[i].item()
                            param_flat[i] = 0.0
                            pruned_loss, _, _ = prune_test(model, data_loader, criterion, lm, batch_size, evaluation)
                            sensitivities.append((abs(pruned_loss - baseline_loss), name, i, 1))
                            param_flat[i] = original_weight

        total_sensitivity = sum(s[0] for s in sensitivities)
        sensitivities = [(s[0] / total_sensitivity, s[1], s[2], s[3]) for s in sensitivities]

        sensitivities.sort()
        for prune_ratio in prune_ratios:
            num_to_prune = int(prune_ratio * len(sensitivities))
            to_prune = sensitivities[:num_to_prune]
            total_pruned_weights = sum(prune_info[3] for prune_info in to_prune)

            mask = {}
            with torch.no_grad():
                for _, name, idx, _ in tqdm(to_prune, "Pruning weights", file=sys.stdout):
                    param = dict(model.named_parameters())[name]
                    param.view(-1)[idx] = 0.0
                    # param.requires_grad = False

                    module = dict(model.named_modules()).get(name)
                    if module is not None and not hasattr(module, 'weight_mask'):
                        module.weight_mask = torch.ones_like(param)
                    if module is not None:
                        module.weight_mask.view(-1)[idx] = 0.0

                    if name not in mask:
                        mask[name] = torch.ones_like(param)
                    mask[name].view(-1)[idx] = 0.0

            if not os.path.exists('./masks'):
                os.mkdir('./masks/')
            if not os.path.exists('./masks/sensitivity'):
                os.mkdir('./masks/sensitivity')
            torch.save(mask, f'./masks/sensitivity/{prune_ratio}_wav2letter_masks.pth')

            print(f"Pruned {total_pruned_weights} weights out of {total_weights} weights total")
            masks[str(prune_ratio)] = mask

    finally:
        if logger:
            logger.flush()
            logger.close()
        sys.stdout = sys.__stdout__

    return model, masks
