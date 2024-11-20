import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
import sys

from logger import Logger
from pruning.sensitivity import prune_sensitivity


# Use Predefined ResNet50 Model
class ResNet50(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)  # Initialize ResNet-50 from scratch
        # Adjust input layer for grayscale images if needed
        if input_channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust final layer for dataset classes

    def forward(self, x):
        return self.model(x)


# trains the model and evaluates validation accuracy, saving the best weights
def train_model(model, train_loader, test_loader, device, epochs, lr=0.01, step_size=8, gamma=0.1, save_best=True,
                save_path="best_model.pth", criterion=nn.CrossEntropyLoss()):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loss_list, val_loss_list = [], []
    train_accuracy_list, val_accuracy_list = [], []

    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        correct_train, total_train, running_train_loss = 0, 0, 0.0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        train_accuracy = (correct_train / total_train) * 100
        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        train_accuracy_list.append(train_accuracy)

        scheduler.step()

        # Validation loop (no gradient update for validation)
        model.eval()
        correct_val, total_val, running_val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        val_accuracy = (correct_val / total_val) * 100
        avg_val_loss = running_val_loss / len(test_loader)
        val_loss_list.append(avg_val_loss)
        val_accuracy_list.append(val_accuracy)

        print(f'Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.2f}%, '
              f'Validation Loss: {avg_val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%')

        if save_best and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, save_path)
            print(f"New best model found and saved with validation accuracy: {best_val_accuracy:.2f}%")

    # plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_loss_list, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_loss_list, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss_plot.png")

    # plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accuracy_list, label="Training Accuracy")
    plt.plot(range(1, epochs + 1), val_accuracy_list, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot.png")

    print("Plots saved as 'loss_plot.png' and 'accuracy_plot.png'")


# Load datasets based on their structure
def get_data_loaders(dataset_name, batch_size):
    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    elif dataset_name == "CIFAR-10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main(dataset_name, prune_method, prune_ratio, epochs=10, batch_size=64, lr=0.01, step_size=8, save_path="best_model.pth", log_file="log.txt"):
    logger = Logger(log_file)
    sys.stdout = logger

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")

        input_channels = 1 if dataset_name == "MNIST" else 3
        model = ResNet50(num_classes=10, input_channels=input_channels).to(device)
        train_loader, test_loader = get_data_loaders(dataset_name, batch_size)
        criterion = nn.CrossEntropyLoss()

        if os.path.exists(save_path):
            print(f"Loading existing best weights from {save_path}")
            model.load_state_dict(torch.load(save_path, map_location=device))
        else:
            print(f"Training {dataset_name} without pruning...")
            train_model(model, train_loader, test_loader, device, epochs, lr=lr, save_path=save_path,
                        step_size=step_size)

        print(f"Pruning {dataset_name} with {prune_ratio*100}% using method {prune_method}...")
        for name, module in model.named_modules():
            print(f"{name}: {module}")

        if prune_method == 'sensitivity':
            model = prune_sensitivity(model, test_loader, criterion=criterion, prune_ratio=prune_ratio,
                                      method='filter', evaluation="batch", batch_size=1, logger=logger)

            masks = {name: module.weight_mask for name, module in model.named_modules() if hasattr(module, 'weight_mask')}
            torch.save(masks, 'sensitivity_masks.pth')

        print(f"Retraining {dataset_name} after pruning...")
        # don't update best weights
        train_model(model, train_loader, test_loader, device, epochs, lr, step_size=step_size, save_best=False)

    finally:
        logger.flush()
        logger.close()
        sys.stdout = sys.__stdout__

if __name__ == '__main__':
    main("CIFAR-10", prune_ratio=0.5, epochs=90, batch_size=256, step_size=30,
         save_path="resnet50_cifar10_best_model.pth", log_file="resnet50_cifar10_training_log.txt",
         prune_method='sensitivity')
