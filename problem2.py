import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np

from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Hyperparameters - same as Problem 1
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
WEIGHT_DECAY = 0.0
NUM_CLASSES = 10

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Same CIFAR-10 transforms
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

# Adapted VGGNet for CIFAR-10
# Similar parameter count to Problem 1 AlexNet
class CIFARVGG(nn.Module):
    def __init__(self, dropout_p=0.0, use_batchnorm=False):
        super().__init__()

        def conv_block(in_channels, out_channels, num_convs):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return layers

        # Adapted VGG-11 style
        # 32 -> 16 -> 8 -> 4 -> 2
        self.features = nn.Sequential(
            *conv_block(3, 64, 1),
            *conv_block(64, 128, 1),
            *conv_block(128, 256, 2),
            *conv_block(256, 256, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(name, dropout_p=0.0, use_batchnorm=False):
    model = CIFARVGG(dropout_p=dropout_p, use_batchnorm=use_batchnorm).to(device)

    print(f"\nTraining {name}")
    print("Trainable parameters:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }

    for epoch in range(EPOCHS):
        start_time = time.time()

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
        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

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


def plot_curves(histories):
    plt.figure()
    for name, h in histories.items():
        plt.plot(h["train_loss"], label=f"{name} train")
        plt.plot(h["val_loss"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VGG Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("vgg_loss_curves.png")
    plt.show()

    plt.figure()
    for name, h in histories.items():
        plt.plot(h["val_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("VGG Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("vgg_val_accuracy_curves.png")
    plt.show()


def plot_confusion(labels, preds, title, filename):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


# Run VGG experiments
experiments = {
    "vgg_baseline": {"dropout_p": 0.0, "use_batchnorm": False},
    "vgg_dropout_0.3": {"dropout_p": 0.3, "use_batchnorm": False},
    "vgg_dropout_0.5": {"dropout_p": 0.5, "use_batchnorm": False},

    # Bonus
    "vgg_batchnorm": {"dropout_p": 0.0, "use_batchnorm": True}
}

models = {}
histories = {}
test_results = {}

for name, config in experiments.items():
    model, history = train_model(
        name=name,
        dropout_p=config["dropout_p"],
        use_batchnorm=config["use_batchnorm"]
    )

    models[name] = model
    histories[name] = history

    test_acc, labels, preds = test_model(model)
    test_results[name] = test_acc

    print(f"{name} Test Accuracy: {test_acc:.2f}%")

    plot_confusion(
        labels,
        preds,
        title=f"{name} Confusion Matrix",
        filename=f"{name}_confusion_matrix.png"
    )

plot_curves(histories)

print("\nFinal VGG Test Results:")
for name, acc in test_results.items():
    avg_time = sum(histories[name]["epoch_time"]) / len(histories[name]["epoch_time"])
    params = count_parameters(models[name])
    print(f"{name}: Test Acc = {acc:.2f}%, Params = {params}, Avg epoch time = {avg_time:.2f}s")