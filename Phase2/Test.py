#!/usr/bin/env python3
import os
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Network.Network import DenseNet121  # Import custom DenseNet121

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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

def evaluate_model(test_loader, model):
    """
    Evaluates the model on the test data and collects predictions and ground truth labels.
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100.0 * correct / total
    return accuracy, all_labels, all_preds

def main():
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_set = CIFAR10(root='./data', train=False, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    model = DenseNet121(num_classes=10).cuda()
    checkpoint_path = './output/final_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
    model.load_state_dict(torch.load(checkpoint_path))

    test_accuracy, true_labels, pred_labels = evaluate_model(test_loader, model)

    output_dir = './output'
    ensure_dir(output_dir)

    confusion_matrix_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(true_labels, pred_labels, confusion_matrix_path, cmap='Blues')

    print(f"Test evaluation completed. Confusion matrix saved.\nFinal Test Accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
    main()
