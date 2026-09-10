import os
import time
import torch
import torch.nn as nn
import torchvision.models as models

def build_resnet18_baseline(num_classes=6, pretrained=True):
    """
    Builds ResNet-18 baseline model for Fruit Classifier comparison.
    """
    if hasattr(models, 'ResNet18_Weights'):
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(pretrained=pretrained)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def train_baseline_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

if __name__ == "__main__":
    print("=== FruitLearn AI: ResNet-18 Baseline Classifier ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_baseline(num_classes=6, pretrained=True).to(device)
    print(f"ResNet-18 Baseline initialized on device: {device}")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
