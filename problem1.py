import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import random_split, DataLoader
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
WEIGHT_DECAY = 0.0
NUM_CLASSES = 10

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Data augmentation / normalization
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

train_size = 45000
val_size = 5000
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

val_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Modified AlexNet for CIFAR-10
class CIFARAlexNet(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()

        self.features = nn.Sequential(
            # Original AlexNet used 11x11 stride 4.
            # CIFAR-10 is only 32x32, so use smaller 3x3 filters.
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Pool less aggressively than original AlexNet.
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)   # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
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


# Train / evaluate functions
def train_one_model(dropout_p=0.0):
    model = CIFARAlexNet(dropout_p=dropout_p).to(device)
    print(f"\nDropout p = {dropout_p}")
    print("Trainable parameters:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(EPOCHS):
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

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}%"
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

    acc = 100 * correct / total
    return acc, all_labels, all_preds


def plot_history(histories):
    plt.figure()
    for name, h in histories.items():
        plt.plot(h["train_loss"], label=f"{name} train")
        plt.plot(h["val_loss"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")
    plt.show()

    plt.figure()
    for name, h in histories.items():
        plt.plot(h["val_acc"], label=f"{name}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_accuracy_curves.png")
    plt.show()


def plot_first_layer_filters(model, filename="first_layer_filters.png"):
    first_conv = model.features[0]
    weights = first_conv.weight.data.cpu()

    # Normalize filters for visualization
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    grid = make_grid(weights[:32], nrow=8, padding=2)
    np_grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(np_grid)
    plt.axis("off")
    plt.title("First Convolutional Layer Filters")
    plt.savefig(filename)
    plt.show()


def plot_confusion(labels, preds, title, filename):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


# Run experiments
experiments = {
    "baseline": 0.0,
    "dropout_0.3": 0.3,
    "dropout_0.5": 0.5
}

models = {}
histories = {}
test_results = {}

for name, p in experiments.items():
    model, history = train_one_model(dropout_p=p)
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

plot_first_layer_filters(models["baseline"])

plot_history(histories)

print("\nFinal Test Accuracies:")
for name, acc in test_results.items():
    print(f"{name}: {acc:.2f}%")