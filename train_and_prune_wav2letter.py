import argparse

import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torchaudio.models import wav2letter
import os
import sys
import matplotlib.pyplot as plt

from logger import Logger
from pruning.masking import apply_masking
from pruning.random import prune_random
from pruning.sensitivity import prune_sensitivity
from pruning.gradient import prune_gradient
from pruning.magnitude import prune_magnitude

MAX_SEQ_LENGTH = 800

labels = [
    " ",
    *"abcdefghijklmnopqrstuvwxyz",
    "'",
    "*"
]


# Wav2Letter model setup
class Wav2Letter(nn.Module):
    def __init__(self, num_classes=29):  # 29 for standard English phoneme classes + blank
        super(Wav2Letter, self).__init__()
        self.model = wav2letter.Wav2Letter(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def encode_labels(transcript):
    return [ord(c) - ord('a') + 1 for c in transcript.lower() if c.isalpha()]  # Example: 'a' -> 1, 'b' -> 2, etc.\


@torch.inference_mode()
def test(model, test_loader, criterion, lm):
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

    test_loss = test_loss_sum / len(test_loader)
    cer = c_ldist_sum / c_ref_len_sum
    wer = w_ldist_sum / w_ref_len_sum

    return test_loss, cer, wer


def collate_fun(batch, encode_fn, mode='train'):
    waves = []
    text_ids = []
    input_lengths = []
    output_lengths = []

    if mode == 'train':
        shifts = torch.randn(len(batch)) > 0.

    for i, (wave, _, text, *_) in enumerate(batch):
        if mode == 'train' and shifts[i]:
            wave = wave[:, 160:]
        waves.append(wave[0])
        ids = torch.LongTensor(encode_fn(text))
        text_ids.append(ids)
        input_lengths.append(wave.size(1) // 320)
        output_lengths.append(len(ids))

    waves = nn.utils.rnn.pad_sequence(waves, batch_first=True).unsqueeze(1)
    labels = nn.utils.rnn.pad_sequence(text_ids, batch_first=True)

    return waves, labels, input_lengths, output_lengths


class GreedyLM:
    def __init__(self, vocab, blank_label='*'):
        self.vocab = vocab
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.blank_label = blank_label

    def encode(self, text):
        return [self.char_to_id[c] for c in text.lower()]

    def decode_ids(self, ids):
        if ids.ndim == 2:  # batch|steps
            return [self.decode_ids(t) for t in ids]

        decoded_text = ''.join([self.vocab[id] for id in ids])

        return decoded_text

    def decode_ctc(self, emissions):
        if emissions.ndim == 3:  # batch|labels|steps
            return [self.decode_ctc(t) for t in emissions]

        amax_ids = emissions.argmax(0)
        amax_ids_collapsed = torch.unique_consecutive(amax_ids)
        decoded_text = ''.join([self.vocab[id] for id in amax_ids_collapsed])
        decoded_text = decoded_text.replace(self.blank_label, '')

        return decoded_text


def get_norm(parameters, norm_type=2.0):
    total_norm = torch.norm(torch.stack([torch.norm(
        p.grad.detach(), norm_type) for p in parameters if p.grad is not None]), norm_type)
    return total_norm

lm = GreedyLM(labels)


# Custom function to preprocess and load LibriSpeech data
def get_data_loaders(dataset_path, batch_size):
    train_dataset = LIBRISPEECH(dataset_path, url='train-clean-100', download=True)
    test_dataset = LIBRISPEECH(dataset_path, url='test-clean', download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fun(x, lm.encode, 'train')
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fun(x, lm.encode, 'valid')
    )

    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs, lr=0.01, save_best=True,
                save_path="best_model.pth", criterion=nn.CTCLoss(), loss_plot_path='loss_plot_wav2letter.png',
                cer_wer_plot_path='cer_wer_wav2letter.png'):
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5)

    best_val_loss = float('inf')
    train_loss_list, val_loss_list = [], []
    cer_list, wer_list = [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            waves, labels, input_lens, output_lens = batch
            waves, labels = waves.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            out = model(waves)  # (batch, n_class, time)
            out = out.permute(2, 0, 1)  # (time, batch, n_class)

            optimizer.zero_grad()
            loss = criterion(out, labels, input_lens, output_lens)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        train_loss_list.append(train_loss)

        # Validation
        model.eval()
        test_loss, cer, wer = test(model, test_loader, criterion, lm)
        val_loss_list.append(test_loss)
        cer_list.append(cer)
        wer_list.append(wer)

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {test_loss:.4f} | "
              f"CER: {cer:.4f} | "
              f"WER: {wer:.4f}")

        # Learning rate scheduler step
        scheduler.step(test_loss)

        # Save the best model
        if save_best and test_loss < best_val_loss:
            best_val_loss = test_loss
            torch.save(model.state_dict(), save_path)

    plt.figure(figsize=(15, 6))
    # Plotting loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_loss_list, label='Train Loss', marker='o')
    plt.plot(range(1, epochs+1), val_loss_list, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path)
    plt.close()

    # Plotting CER and WER
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), cer_list, label='Character Error Rate (CER)', marker='o')
    plt.plot(range(1, epochs+1), wer_list, label='Word Error Rate (WER)', marker='o')
    plt.title('Character Error Rate (CER) and Word Error Rate (WER)')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(cer_wer_plot_path)
    plt.close()

    print(f"Plots saved as {loss_plot_path} and {cer_wer_plot_path}")

    return train_loss_list, val_loss_list, cer_list, wer_list


def prune_and_train(dataset_path, prune_method, prune_ratios, epochs=10, batch_size=8, lr=0.01,
         save_path="best_model.pth", log_file="log.txt"):
    logger = Logger(log_file)
    sys.stdout = logger

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")

        model = Wav2Letter(num_classes=len(labels)).cuda()
        train_loader, test_loader = get_data_loaders(dataset_path, batch_size)
        criterion = nn.CTCLoss(blank=len(labels) - 1, zero_infinity=True).to(device)

        if os.path.exists(save_path):
            print(f"Loading existing best weights from {save_path}")
            state_dict = torch.load(save_path)

            new_state_dict = {}
            for key in state_dict.keys():
                new_key = 'model.' + key
                new_state_dict[new_key] = state_dict[key]
            # Load the adjusted state dict into the model
            model.load_state_dict(new_state_dict)
        else:
            print(f"Training LibriSpeech without pruning...")
            train_model(model=model, train_loader=train_loader, test_loader=test_loader, lr=lr, epochs=epochs, save_path=save_path,
                        criterion=criterion, save_best=True)

        print(f"Pruning LibriSpeech using method {prune_method}...")
        masks = []
        if prune_method == 'sensitivity':
            model, masks = prune_sensitivity(model=model, data_loader=test_loader, criterion=criterion, prune_ratios=prune_ratios,
                                      method='filter', evaluation="batch", batch_size=1, logger=logger, lm=lm)
        elif prune_method == 'gradient':
            model, masks = prune_gradient(model=model, data_loader=test_loader, criterion=criterion, prune_ratios=prune_ratios,
                                   device=device, logger=logger)
        elif prune_method == 'magnitude':
            model, masks = prune_magnitude(model=model, prune_ratios=prune_ratios, logger=logger)
        elif prune_method == 'random':
            model, masks = prune_random(model=model, prune_ratios=prune_ratios, logger=logger)

        rd_data = []
        for prune_ratio, mask in masks.items():
            model = apply_masking(model, mask)

            print(f"Retraining LibriSpeech after pruning...")
            if not os.path.exists('./weights/'):
                os.mkdir('./weights')
            if not os.path.exists(f'./weights/{prune_method}'):
                os.mkdir(f'./weights/{prune_method}')
            _, _, cer, wer = train_model(model=model, train_loader=train_loader, test_loader=test_loader, lr=lr, epochs=epochs,
                        criterion=criterion, save_best=True, save_path=f'{prune_method}_{prune_ratio}_wav2letter_best.pth',
                        cer_wer_plot_path=f'{prune_method}_{prune_ratio}_cer_wer_plot.png',
                        loss_plot_path=f'{prune_method}_{prune_ratio}_loss_plot.png')

            rd_data.append((prune_ratio, min(cer), min(wer)))
        return rd_data

    finally:
        logger.flush()
        logger.close()
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and prune the Wav2Letter model on LibriSpeech data.")
    parser.add_argument('--dataset_path', type=str, default="./data", help="Path to the dataset.")
    parser.add_argument('--prune_method', type=str, default="gradient", choices=['sensitivity', 'gradient', 'magnitude', 'random'], help="Pruning method to use.")
    parser.add_argument('--prune_ratio', type=float, default=0.5, help="Ratio of pruning (e.g., 0.5 for 50%).")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--save_path', type=str, default="states_fused.pth", help="Path to save the best model state.")
    parser.add_argument('--log_file', type=str, default="log_wav2letter_librispeech.txt", help="File to save the training log.")
    parser.add_argument('--all_ratios', action=argparse.BooleanOptionalAction, help="Whether to prune using all ratios (use to make rate-distortion curve)")

    args = parser.parse_args()

    if args.all_ratios:
        ratios = [0.1, 0.2, 0.5, 0.7, 0.9]
        rd_data = prune_and_train(
                    dataset_path=args.dataset_path,
                    prune_method=args.prune_method,
                    prune_ratios=ratios,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    save_path=args.save_path,
                    log_file=f"{args.prune_method}_{args.log_file}"
                )

        prune_ratios, cer_values, wer_values = zip(*rd_data)
        plt.figure(figsize=(8, 5))
        plt.plot(prune_ratios, cer_values, marker='o', label='CER (Character Error Rate)')
        plt.plot(prune_ratios, wer_values, marker='s', label='WER (Word Error Rate)')

        plt.xlabel('Prune Ratio')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{args.prune_method}_rd_curve_wav2letter_librispeech.png')
        plt.close()
    else:
        prune_and_train(
            dataset_path=args.dataset_path,
            prune_method=args.prune_method,
            prune_ratios=[args.prune_ratio],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.save_path,
            log_file=f"{args.prune_method}_{args.log_file}"
        )
