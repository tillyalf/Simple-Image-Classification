import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import WeightedRandomSampler


def remap_dataset(subset, mapping):
    """
    Remaps labels based on the inputted version, disregarding unmentioned labels
    Returns (img, new_label) pairs list
    """
    new_data = []
    for img, old_label in subset:
        if old_label in mapping: 
            new_data.append((img, mapping[old_label]))
    return new_data

def evaluate_and_plot(model, loader, device, title_prefix="Model"):
    """
    This function performs the same accuracy and matrices steps that have been inputed seperately throughout the assignment so far
    This has been introduced for part 5 to make the process more efficient, it really just repeats the same code already discussed
    """
    model.eval()
    pred_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, pred_labels)
    print(f"{title_prefix} Accuracy: {acc:.4f}")

    cm_raw  = confusion_matrix(true_labels, pred_labels)
    cm_norm = confusion_matrix(true_labels, pred_labels, normalize='true')

    plt.figure(figsize=(7,5))
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="viridis")
    plt.title(f"{title_prefix} - Raw Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(7,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"{title_prefix} - Normalised Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return acc, cm_raw, cm_norm

def weighted_sampler(y,device):
    """
    This is a function for compensating for the class imbalance by using WeightedRandomSampler
    This was provided by the lab exercises!
    """
    classes, counts = y.unique(return_counts=True)
    weights = 1.0 / counts.float()
    sample_weights = weights[y.squeeze().long()]
    generator = torch.Generator() # this changed to cpu to fix error when outputting the batch frequencies
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        generator=generator,
        replacement=True
    )
    return sampler