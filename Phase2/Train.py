#!/usr/bin/env python3
import os
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomCrop
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Network.Network import DenseNet121  # Import custom DenseNet121

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_metric(values, title, ylabel, xlabel, output_path, color='blue'):
    """
    Plots a metric with the given color.
    """
    plt.figure()
    plt.plot(values, marker='o', color=color, linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_confusion_matrix(true_labels, pred_labels, output_path, cmap='Blues'):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap=cmap, values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved at: {output_path}")

def evaluate_model(loader, model):
    """
    Evaluates the model and returns accuracy, true labels, and predicted labels.
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100.0 * correct / total
    return accuracy, all_labels, all_preds

def train_operation(train_loader, test_loader, model, optimizer, scheduler, loss_fn, epochs, output_dir, batch_size):
    ensure_dir(output_dir)

    train_accuracies = []
    test_accuracies = []
    losses = []

    writer = SummaryWriter(log_dir='./Logs')

    dummy_input = torch.randn(1, 3, 32, 32).cuda()
    writer.add_graph(model, dummy_input)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        train_accuracies.append(train_accuracy)
        losses.append(total_loss)

        test_accuracy, test_labels, test_preds = evaluate_model(test_loader, model)
        test_accuracies.append(test_accuracy)

        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Loss: {total_loss:.4f}")

    writer.close()

    # Save graphs
    plot_metric(train_accuracies, "Train Accuracy over Epochs", "Accuracy (%)", "Epoch", os.path.join(output_dir, "train_accuracy.png"))
    plot_metric(test_accuracies, "Test Accuracy over Epochs", "Accuracy (%)", "Epoch", os.path.join(output_dir, "test_accuracy.png"))
    plot_metric(losses, "Loss over Epochs", "Loss", "Epoch", os.path.join(output_dir, "loss.png"))

    # Save optimizer and batch size details
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open(os.path.join(output_dir, "model_parameters.txt"), "w") as f:
        f.write(f"Number of parameters: {param_count}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Batch Size: {batch_size}\n")
    print(f"Model parameters and training details saved to {os.path.join(output_dir, 'model_parameters.txt')}")

    # Save confusion matrices
    train_accuracy, train_labels, train_preds = evaluate_model(train_loader, model)
    plot_confusion_matrix(train_labels, train_preds, os.path.join(output_dir, "train_confusion_matrix.png"))

    plot_confusion_matrix(test_labels, test_preds, os.path.join(output_dir, "test_confusion_matrix.png"))

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    print("Model saved to:", os.path.join(output_dir, "final_model.pth"))

def main():
    batch_size = 32  # Adjusted for potential memory limitations
    transform = Compose([
        RandomHorizontalFlip(),
        RandomCrop(32, padding=4),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = DenseNet121(num_classes=10).cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()

    output_dir = './output'
    train_operation(train_loader, test_loader, model, optimizer, scheduler, loss_fn, epochs=20, output_dir=output_dir, batch_size=batch_size)

if __name__ == '__main__':
    main()
