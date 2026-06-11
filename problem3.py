import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 10

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-10 data
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_train = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

train_dataset, val_dataset = random_split(full_train, [45000, 5000])
val_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # skip connection
        out += identity
        out = self.relu(out)

        return out

# ResNet for CIFAR-10
class CIFARResNet(nn.Module):
    def __init__(self, block, layers, dropout_p=0.0):
        super().__init__()

        self.in_channels = 64

        # CIFAR version: 3x3 conv, no 7x7 conv, no maxpool
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []

        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out


def ResNet11(dropout_p=0.0):
    # smaller baseline: 1 block per stage
    return CIFARResNet(BasicBlock, [1, 1, 1, 1], dropout_p)


def ResNet18(dropout_p=0.0):
    # standard ResNet-18: 2 blocks per stage
    return CIFARResNet(BasicBlock, [2, 2, 2, 2], dropout_p)


def ResNet34(dropout_p=0.0):
    # bonus
    return CIFARResNet(BasicBlock, [3, 4, 6, 3], dropout_p)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Train / test
def train_model(model_name, model):
    model = model.to(device)

    print(f"\nTraining {model_name}")
    print("Parameters:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[25, 40],
        gamma=0.1
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }

    for epoch in range(EPOCHS):
        start = time.time()

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        epoch_time = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}% "
            f"Time: {epoch_time:.2f}s"
        )

    return model, history


def test_model(model):
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    return test_acc, all_labels, all_preds


def plot_confusion(labels, preds, title, filename):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def plot_curves(histories):
    plt.figure()
    for name, h in histories.items():
        plt.plot(h["train_loss"], label=f"{name} train")
        plt.plot(h["val_loss"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("resnet_loss_curves.png")
    plt.show()

    plt.figure()
    for name, h in histories.items():
        plt.plot(h["val_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("ResNet Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("resnet_val_accuracy_curves.png")
    plt.show()


def plot_accuracy_bar(results):
    names = list(results.keys())
    accs = [results[n]["test_acc"] for n in names]

    plt.figure(figsize=(10, 5))
    plt.bar(names, accs)
    plt.ylabel("Test Accuracy (%)")
    plt.title("ResNet Test Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("resnet_test_accuracy_bar.png")
    plt.show()

# Run experiments
experiments = {
    "resnet11_baseline": ResNet11(dropout_p=0.0),
    "resnet11_dropout_0.3": ResNet11(dropout_p=0.3),
    "resnet11_dropout_0.5": ResNet11(dropout_p=0.5),

    "resnet18_baseline": ResNet18(dropout_p=0.0),
    "resnet18_dropout_0.3": ResNet18(dropout_p=0.3),
    "resnet18_dropout_0.5": ResNet18(dropout_p=0.5),

    # Uncomment for bonus:
    # "resnet34_baseline": ResNet34(dropout_p=0.0),
}

models = {}
histories = {}
results = {}

for name, model in experiments.items():
    trained_model, history = train_model(name, model)

    test_acc, labels, preds = test_model(trained_model)

    models[name] = trained_model
    histories[name] = history

    results[name] = {
        "params": count_parameters(trained_model),
        "test_acc": test_acc,
        "avg_epoch_time": sum(history["epoch_time"]) / len(history["epoch_time"])
    }

    print(f"{name} Test Accuracy: {test_acc:.2f}%")

    plot_confusion(
        labels,
        preds,
        title=f"{name} Confusion Matrix",
        filename=f"{name}_confusion_matrix.png"
    )

plot_curves(histories)
plot_accuracy_bar(results)

# Results table
df = pd.DataFrame(results).T
print(df)

df.to_csv("resnet_results.csv")
print("Saved resnet_results.csv")