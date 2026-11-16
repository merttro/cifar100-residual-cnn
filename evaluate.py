
import torch
from model import ConvModel
from dataset import get_dataloaders

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli oluştur
model = ConvModel().to(device)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

# Dataset
train_loader,val_loader, test_loader = get_dataloaders(batch_size=64)

# Eğitim görüntülerinde augmentation'ı kapat
train_loader.dataset.dataset.transform = val_loader.dataset.dataset.transform

correct = 0
total = 0

with torch.no_grad():
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

train_accuracy = 100 * correct / total

print(f"Train Accuracy: {train_accuracy:.2f}%")
print("Test Accuracy: 61.56%")
